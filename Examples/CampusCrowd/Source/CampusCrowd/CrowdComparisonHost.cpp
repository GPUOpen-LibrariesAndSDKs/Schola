// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "CrowdComparisonHost.h"
#include "CrowdProcessors.h"
#include "CampusScene.h"
#include "EngineUtils.h"
#include "DrawDebugHelpers.h"
#include "UnrealClient.h"
#include "Materials/MaterialInterface.h"
#include "Camera/CameraActor.h"
#include "Camera/CameraComponent.h"
#include "Components/DirectionalLightComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/SkyLightComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/DirectionalLight.h"
#include "Engine/SkyLight.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Components/SkyAtmosphereComponent.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/GameModeBase.h"
#include "HAL/FileManager.h"
#include "HAL/IConsoleManager.h"
#include "Inference/BackgroundInferenceScheduler.h"
#include "Inference/NNEBatchBackendProvider.h"
#include "MassEntityManager.h"
#include "MassEntitySubsystem.h"
#include "MassExecutor.h"
#include "MassProcessingContext.h"
#include "Misc/CommandLine.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "RHIGlobals.h"
#include "ScholaInferenceFragments.h"
#include "ScholaMassInferenceSubsystem.h"
#include "Serialization/JsonSerializer.h"
#include "Spaces/BoxSpace.h"
#include "UObject/ConstructorHelpers.h"
#include "Async/ParallelFor.h"

namespace
{
double Quantile(TArray<double> Values, double Q)
{
    if (Values.IsEmpty()) return 0;
    Values.Sort();
    return Values[FMath::Clamp(FMath::CeilToInt(Q * Values.Num()) - 1, 0, Values.Num() - 1)];
}

double Mean(const TArray<double>& Values)
{
    double Sum = 0;
    for (double Value : Values) Sum += Value;
    return Values.IsEmpty() ? 0 : Sum / Values.Num();
}

double RowsPerCall(const FNPCBackgroundStats& Now, const FNPCBackgroundStats& Before)
{
    const int64 Calls = Now.BackendCalls - Before.BackendCalls;
    return Calls > 0 ? double(Now.BackendRows - Before.BackendRows) / Calls : 0;
}

double ServiceMs(const FNPCBackgroundStats& Now, const FNPCBackgroundStats& Before)
{
    return (Now.BackendServiceSeconds - Before.BackendServiceSeconds) * 1000;
}

int64 Dropped(const FNPCBackgroundStats& Now, const FNPCBackgroundStats& Before)
{
    return (Now.BusyRejections - Before.BusyRejections) + (Now.ExpiredRows - Before.ExpiredRows) +
        (Now.FailedRows - Before.FailedRows);
}

/** Delivered fraction below which a window or capture counts as missing the target rate. */
constexpr double TargetTolerance = .95;

FMassEntityManager& EntityManager(UWorld* World)
{
    return World->GetSubsystem<UMassEntitySubsystem>()->GetMutableEntityManager();
}

/** One key of the comparison flight: camera position and the point it looks at. */
struct FFlightKey
{
    double Time;
    FVector Location;
    FVector Target;
};

// A flight (keys in seconds at the default ten-second length) checked against
// the CAD roof heights: it starts 22 m over the front lawn crowd, crosses the
// road towards Queen's Park, banks over the park and pulls back until the
// whole campus model fills the frame below the HUD panels.
const FFlightKey FlightKeys[] = {
    {0.0, FVector(-3000, 17000, 2200), FVector(-11000, 9000, 0)},
    {3.0, FVector(4000, 8000, 4500), FVector(22000, 2000, 0)},
    {6.0, FVector(27000, 12000, 14000), FVector(4000, -14000, 0)},
    {10.0, FVector(38800, 164600, 152700), FVector(-13400, 21200, 0)},
};

FVector CatmullRom(const FVector& P0, const FVector& P1, const FVector& P2, const FVector& P3, double U)
{
    return .5 * (2 * P1 + (P2 - P0) * U + (2 * P0 - 5 * P1 + 4 * P2 - P3) * U * U +
        (3 * P1 - P0 - 3 * P2 + P3) * U * U * U);
}

void EvaluateFlight(double Seconds, FVector& OutLocation, FVector& OutTarget)
{
    constexpr int32 Last = UE_ARRAY_COUNT(FlightKeys) - 1;
    const double Time = FMath::Clamp(Seconds, 0.0, FlightKeys[Last].Time);
    int32 I = 0;
    while (I < Last - 1 && Time >= FlightKeys[I + 1].Time) ++I;
    const double U = (Time - FlightKeys[I].Time) / (FlightKeys[I + 1].Time - FlightKeys[I].Time);
    const FFlightKey& K0 = FlightKeys[FMath::Max(I - 1, 0)];
    const FFlightKey& K1 = FlightKeys[I];
    const FFlightKey& K2 = FlightKeys[I + 1];
    // Mirroring the previous key past the end gives zero velocity there, so the flight settles on the overview.
    const FFlightKey& K3 = FlightKeys[I + 2 <= Last ? I + 2 : I];
    OutLocation = CatmullRom(K0.Location, K1.Location, K2.Location, K3.Location, U);
    OutTarget = CatmullRom(K0.Target, K1.Target, K2.Target, K3.Target, U);
}
}

ACrowdComparisonHost::ACrowdComparisonHost()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PrePhysics;
    Crowd = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("Crowd"));
    RootComponent = Crowd;
    Crowd->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    Crowd->SetCastShadow(true);
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Cylinder(TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
    Crowd->SetStaticMesh(Cylinder.Object);
}

void ACrowdComparisonHost::BeginPlay()
{
    Super::BeginPlay();
    const TCHAR* Command = FCommandLine::Get();
    const bool bMigrations = FParse::Param(Command, TEXT("Migrations"));
    const bool bInferenceBenchmark = FParse::Param(Command, TEXT("InferenceBenchmark"));
    FString Preset;
    const bool bExplicitDemo = FParse::Value(Command, TEXT("Demo="), Preset);
    const auto* GameMode = GetWorld()->GetAuthGameMode();
    const FString Selected = GameMode ? UGameplayStatics::ParseOption(GameMode->OptionsString, TEXT("CampusDemo")) : FString();
    if (!Selected.IsEmpty()) Preset = Selected;
    int32 StudentFlag = 0;
    const bool bStudentFlag = FParse::Value(Command, TEXT("Students="), StudentFlag);
    const bool bLegacyNavigation = FParse::Param(Command, TEXT("NavigationRL")) ||
        FCString::Stristr(Command, TEXT("-NavigationRoutes=")) != nullptr;
    const bool bTurnFlag = FParse::Param(Command, TEXT("Turns"));
    const bool bDemoRequested = bExplicitDemo || !Selected.IsEmpty();
    if (bDemoRequested && !Preset.Equals(TEXT("Four"), ESearchCase::IgnoreCase) &&
        !Preset.Equals(TEXT("Scale"), ESearchCase::IgnoreCase))
    {
        Fail(TEXT("Use -Demo=Four or -Demo=Scale."));
        return;
    }
    bSplitDemo = bDemoRequested || bTurnFlag || (!bMigrations && !bInferenceBenchmark && !bLegacyNavigation);
    bFourDemo = bSplitDemo && (bDemoRequested ? Preset.Equals(TEXT("Four"), ESearchCase::IgnoreCase) :
        !bTurnFlag && !bStudentFlag);
    const bool bTurns = bSplitDemo;
    const bool bNavigation = !bTurns && bLegacyNavigation;
    int32 AgentCount = bTurns ? 650 : bNavigation ? 24 : 2000, Workers = 2, MaxBatch = 64;
    float DecisionHz = 10, MaxFPS = 60;
    FString ModeName = TEXT("Background");
    FParse::Value(Command, TEXT("View="), CameraPreset);
    FParse::Value(Command, TEXT("NavigationGateView="), NavigationGateView);
    FParse::Value(Command, TEXT("FlightSeconds="), FlightSeconds);
    FlightSeconds = FMath::Max(FlightSeconds, 1.0);
    CameraPreset = FMath::Clamp(CameraPreset, 0, 3);
    FParse::Value(Command, TEXT("Agents="), AgentCount);
    if (bSplitDemo && bStudentFlag) AgentCount = StudentFlag;
    if (bFourDemo) AgentCount = 4;
    FParse::Value(Command, TEXT("DecisionHz="), DecisionHz);
    FParse::Value(Command, TEXT("MaxFPS="), MaxFPS);
    FParse::Value(Command, TEXT("Workers="), Workers);
    FParse::Value(Command, TEXT("MaxBatch="), MaxBatch);
    FParse::Value(Command, TEXT("Mode="), ModeName);
    FParse::Value(Command, TEXT("CaptureSeconds="), CaptureSeconds);
    FParse::Value(Command, TEXT("WarmupSeconds="), WarmupSeconds);
    FParse::Value(Command, TEXT("Report="), ReportPath);
    constexpr int32 MaxCrowdAgents = 100000;
    constexpr int32 MaxScalingStudents = 500000;
    AgentCount = FMath::Clamp(AgentCount, 1, bTurns && !bFourDemo ? MaxScalingStudents : MaxCrowdAgents);
    bShowRoutes = bNavigation && AgentCount <= 32 && CameraPreset == 0 && NavigationGateView < 0;
    PeriodSeconds = 1.0 / FMath::Clamp(DecisionHz, .1f, 60.f);
    // Bound the scheduler queue to about 50 ms of target work: a longer queue makes
    // results older than the action age limit, so they would all be discarded.
    // One frame's submission is a single request and must fit in the queue, so admit
    // at most half of it per frame. A larger share leaves too little room for the
    // next frame, and a submission that does not fit is rejected whole. Overload then
    // shows up as reported deferrals rather than a silent collapse.
    MaxPendingRows = FMath::Clamp(FMath::CeilToInt32(AgentCount / PeriodSeconds * .05), FMath::Min(4096, AgentCount), AgentCount);
    FParse::Value(Command, TEXT("MaxPendingRows="), MaxPendingRows);
    MaxPendingRows = FMath::Max(1, MaxPendingRows);
    MaxRowsPerFrame = FMath::Max(1, MaxPendingRows / 2);
    FParse::Value(Command, TEXT("MaxRowsPerFrame="), MaxRowsPerFrame);
    if (MaxRowsPerFrame > MaxPendingRows)
        UE_LOG(LogTemp, Warning, TEXT("Crowd: -MaxRowsPerFrame=%d exceeds -MaxPendingRows=%d; clamping, since larger submissions are always rejected"),
            MaxRowsPerFrame, MaxPendingRows);
    MaxRowsPerFrame = FMath::Clamp(MaxRowsPerFrame, 1, MaxPendingRows);
    HalfSize = FMath::Sqrt(float(AgentCount)) * 75.f;

    if (MaxFPS > 0) IConsoleManager::Get().FindConsoleVariable(TEXT("t.MaxFPS"))->Set(MaxFPS);
    if (auto* VSync = IConsoleManager::Get().FindConsoleVariable(TEXT("r.VSync"))) VSync->Set(0);

    BuildScene();
    if (!Campus || Campus->Nodes.Num() < 2) { Fail(TEXT("Campus route data could not be loaded")); return; }
    HalfSize = Campus->Extent;
    const bool bSyncNavigation = ModeName.Equals(TEXT("Sync"), ESearchCase::IgnoreCase) ||
        ModeName.Equals(TEXT("Synchronous"), ESearchCase::IgnoreCase);
    if (bNavigation || bMigrations || bTurns)
    {
        MigrationDay = MakeUnique<FCampusMigrationDay>();
        FString MigrationError;
        if (!MigrationDay->Initialize(AgentCount, bNavigation, bTurns, Command, MigrationError, !bSyncNavigation, bFourDemo)) { Fail(MigrationError); return; }
        if (bFourDemo) MigrationTimeScale = 12.f;
        FParse::Value(Command, TEXT("MigrationTimeScale="), MigrationTimeScale);
        FParse::Value(Command, TEXT("MigrationFixedStep="), MigrationFixedStep);
        MigrationTimeScale = FMath::Clamp(MigrationTimeScale, .1f, 16.f);
        MigrationFixedStep = FMath::Clamp(MigrationFixedStep, 0.f, .1f);
        bExitAfterMigration = FParse::Param(Command, TEXT("ExitAfterMigrations"));
        if (MigrationFixedStep > 0 && !bNavigation && !bTurns)
        {
            // Replay every decision on the simulation clock, independent of machine speed.
            ModeName = TEXT("Sync");
        }
    }
    if (MigrationDay) ScheduledMigrationGenerations.Init(INDEX_NONE, AgentCount);
    const bool bUseModel = !MigrationDay || MigrationDay->UsesModel();
    const int32 ObservationSize = MigrationDay ? MigrationDay->GetObservationSize() : CrowdSim::ObservationSize;
    const int32 ActionSize = MigrationDay ? MigrationDay->GetActionSize() : CrowdSim::ActionSize;
    if (bUseModel && !LoadPolicy()) return;

    auto& Manager = EntityManager(GetWorld());
    FMassArchetypeSharedFragmentValues Shared;
    TArray<const UScriptStruct*> Types{FCrowdAgentFragment::StaticStruct()};
    if (bUseModel)
    {
        // One policy and one batching bucket shared by the whole crowd.
        FBoxSpace Observations, Actions;
        for (int32 I = 0; I < ObservationSize; ++I) Observations.Add(-1e6f, 1e6f);
        for (int32 I = 0; I < ActionSize; ++I) Actions.Add(-1e6f, 1e6f);
        const FInteractionDefinition Definition(TInstancedStruct<FSpace>::Make<FBoxSpace>(Observations),
            TInstancedStruct<FSpace>::Make<FBoxSpace>(Actions));
        auto* Provider = NewObject<UNNEBatchBackendProvider>(this);
        Provider->ModelData = ModelData;
        FNPCBackgroundSchedulerSettings Settings;
        Settings.WorkerCount = FMath::Clamp(Workers, 1, 64);
        Settings.MaxBatchSize = FMath::Max(1, MaxBatch);
        Settings.MaxPendingRows = MaxPendingRows;
        Settings.MaxPendingBytes = FMath::Max(Settings.MaxPendingBytes,
            int64(MaxPendingRows) * (ObservationSize + ActionSize) * int64(sizeof(float)) * 2);
        WorkerCount = Settings.WorkerCount;
        MaxBatchSize = Settings.MaxBatchSize;
        FString BucketError;
        Service = GetWorld()->GetSubsystem<UScholaMassInferenceSubsystem>();
        const int32 Bucket = Service ? Service->AcquireBucket(*Provider, Definition, Settings, BucketError) : INDEX_NONE;
        if (Bucket == INDEX_NONE) { Fail(TEXT("Background bucket: ") + BucketError); return; }
        {
            FScopeLock Lock(&Service->Mutex);
            Scheduler = Service->Bridge.GetScheduler(Bucket);
        }

        FScholaInferenceBucketFragment BucketFragment;
        BucketFragment.BucketId = Bucket;
        Shared.Add(Manager.GetOrCreateConstSharedFragment(BucketFragment));
        Shared.Sort();
        Types.Add(FScholaInferenceStateFragment::StaticStruct());
        Types.Add(FScholaInferenceScheduleFragment::StaticStruct());
    }
    const FMassArchetypeHandle Archetype = Manager.CreateArchetype(MakeArrayView(Types));
    Started = FPlatformTime::Seconds();
    {
        // Fill fragments before the creation context fires ScholaMass's schedule observer.
        auto Creation = Manager.BatchCreateEntities(Archetype, Shared, AgentCount, Entities);
        Transforms.SetNum(AgentCount);
        for (int32 I = 0; I < AgentCount; ++I)
        {
            auto& Agent = Manager.GetFragmentDataChecked<FCrowdAgentFragment>(Entities[I]);
            Agent.Instance = I;
            if (MigrationDay)
            {
                if (bUseModel)
                    Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entities[I]).PeriodSeconds = 0;
                Transforms[I] = FTransform(FRotator::ZeroRotator, FVector(0, 0, -1000), FVector::ZeroVector);
                continue;
            }
            Agent.Random = 0x9E3779B9u ^ uint32(I * 7919 + 1);
            Agent.CurrentNode = CrowdSim::NextRandom(Agent.Random) % Campus->Nodes.Num();
            const auto& Node = Campus->Nodes[Agent.CurrentNode];
            Agent.NextNode = Node.Links[CrowdSim::NextRandom(Agent.Random) % Node.Links.Num()];
            Agent.Goal = Campus->Nodes[Agent.NextNode].Position;
            Agent.Lane = (CrowdSim::NextRandom(Agent.Random) / float(MAX_uint32) - .5f) * 220.f;
            Agent.WalkSpeed = 115.f + CrowdSim::NextRandom(Agent.Random) % 70;
            Agent.Position = FMath::Lerp(Node.Position, Agent.Goal, CrowdSim::NextRandom(Agent.Random) / float(MAX_uint32));
            const FVector2f Along = (Agent.Goal - Node.Position).GetSafeNormal();
            Agent.Position += FVector2f(-Along.Y, Along.X) * Agent.Lane;
            Agent.Steering = (Agent.Goal - Agent.Position).GetSafeNormal();
            Agent.SyncNextDue = Started + PeriodSeconds * (CrowdSim::NextRandom(Agent.Random) / double(MAX_uint32));
            Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entities[I]).PeriodSeconds = PeriodSeconds;
            Transforms[I] = FTransform(FRotator::ZeroRotator, FVector(Agent.Position.X, Agent.Position.Y, 94), FVector(.6f, .6f, 1.8f));
        }
    }
    const bool bColorDestinations = bSplitDemo && FApp::CanEverRender() && !GUsingNullRHI;
    if (bColorDestinations)
    {
        FString HighlightError;
        const auto& Keys = MigrationDay->GetTurns()->GetBuildingKeys();
        if (!Campus->LoadDestinationHighlights(Keys, HighlightError)) { Fail(HighlightError); return; }
        auto* Material = LoadObject<UMaterialInterface>(nullptr, TEXT("/Game/Campus/M_DestinationStudents.M_DestinationStudents"));
        if (!Material) { Fail(TEXT("Missing M_DestinationStudents. Run Tools/create_destination_material.py in Unreal.")); return; }
        Crowd->NumCustomDataFloats = 3;
        Crowd->SetMaterial(0, Material);
        Crowd->SetCastShadow(false);
        ActiveDestinations.Init(false, Keys.Num());
        StudentColorKeys.Init(MAX_int32, AgentCount);
    }
    else Crowd->SetMaterial(0, LoadObject<UMaterialInterface>(nullptr, TEXT("/Game/Campus/M_Students.M_Students")));
    Crowd->AddInstances(Transforms, false, true, false);

    if (bUseModel)
    {
        Gather = NewObject<UCrowdGatherProcessor>(this);
        Apply = NewObject<UCrowdApplyProcessor>(this);
        Sync = NewObject<UCrowdSyncInferenceProcessor>(this);
        Move = NewObject<UCrowdMoveProcessor>(this);
        Gather->HalfSize = Sync->HalfSize = Move->HalfSize = HalfSize;
        Gather->MigrationDay = MigrationDay.Get();
        Apply->MigrationDay = Sync->MigrationDay = MigrationDay.Get();
        Gather->MaxRowsPerFrame = MaxRowsPerFrame;
        if (bTurns)
        {
            // Keep a completed decision usable until the next retry supersedes
            // its slot. The turn is still pending at the original junction.
            Gather->MaxActionAgeMs = Apply->MaxActionAgeMs = 0.f;
            Gather->PhaseLayout = ENPCPhaseLayout::PerEntityUniform;
        }
        Apply->Counters = Sync->Counters = &Counters;
        Sync->PeriodSeconds = PeriodSeconds;
        Sync->bUseSimulationClock = MigrationFixedStep > 0;
        Move->Transforms = &Transforms;
        Move->Campus = Campus;
        for (UMassProcessor* Processor : TArray<UMassProcessor*>{Gather, Apply, Sync, Move})
            Processor->CallInitialize(this, Manager.AsShared());

        // Without the API: one model instance, reshaped once to batch 1, called per agent.
        auto Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
        Sync->Model = Runtime.IsValid() ? Runtime->CreateModelCPU(ModelData) : nullptr;
        Sync->Instance = Sync->Model ? Sync->Model->CreateModelInstanceCPU() : nullptr;
        const UE::NNE::FTensorShape Shape = UE::NNE::FTensorShape::Make({1u, uint32(ObservationSize)});
        if (!Sync->Instance || Sync->Instance->SetInputTensorShapes(MakeArrayView(&Shape, 1)) != UE::NNE::EResultStatus::Ok)
        { Fail(TEXT("Could not create the synchronous ORT model instance")); return; }
    }

    SetMode(ModeName.Equals(TEXT("Sync"), ESearchCase::IgnoreCase) || ModeName.Equals(TEXT("Synchronous"), ESearchCase::IgnoreCase)
        ? ECrowdInferenceMode::Synchronous : ECrowdInferenceMode::Background);
    PreviousTick = WindowStart = FPlatformTime::Seconds();
    if (bTurns)
    {
        UE_LOG(LogTemp, Display, TEXT("Campus turn policy: %d students, %d workers, max batch %d"),
            AgentCount, WorkerCount, MaxBatchSize);
    }
    else if (bNavigation)
    {
        UE_LOG(LogTemp, Display, TEXT("Campus navigation: %d students"), AgentCount);
    }
    else if (MigrationDay)
    {
        UE_LOG(LogTemp, Display, TEXT("Campus day: %d students, %s policy, %.1f decisions/s per walking student"),
            AgentCount, MigrationDay->UsesModel() ? TEXT("learned") : TEXT("rules"), MigrationDay->UsesModel() ? 1. / PeriodSeconds : 0.);
    }
    else
    {
        UE_LOG(LogTemp, Display, TEXT("Crowd: %d agents at %.1f Hz (target %.0f decisions/s); %d workers, batch %d, %d rows/frame, %d pending rows"),
            AgentCount, 1.0 / PeriodSeconds, GetTargetDecisionsPerSecond(), WorkerCount, MaxBatchSize, MaxRowsPerFrame, MaxPendingRows);
    }
}

FCrowdLosses ACrowdComparisonHost::ReadLosses() const
{
    FCrowdLosses Losses;
    if (Scheduler)
    {
        const FNPCBackgroundStats Stats = Scheduler->GetStats();
        Losses.Rejected = Stats.BusyRejections;
        Losses.Invalid = Stats.InvalidSubmissions;
        Losses.Expired = Stats.ExpiredRows;
        Losses.Failed = Stats.FailedRows;
    }
    if (Service)
    {
        FScopeLock Lock(&Service->Mutex);
        Losses.Deferred = Service->Bridge.DeferredRows;
        Losses.Stale = Service->Bridge.DroppedRows;
    }
    return Losses;
}

FString ACrowdComparisonHost::DiagnoseShortfall(const FCrowdLosses& Window, double FrameMs) const
{
    if (Mode == ECrowdInferenceMode::Synchronous)
        return FString::Printf(TEXT("per-agent synchronous inference cannot keep up (%.1f ms/frame on the game thread)"), Display.InferenceMs);
    const auto Count = [](int64 Value) { return FText::AsNumber(Value).ToString(); };
    const double FramesPerSecond = 1000 / FMath::Max(FrameMs, 1.);
    // Several causes can overlap; list every one that occurred, most direct first.
    TArray<FString> Causes;
    if (Window.Invalid > 0)
        Causes.Add(FString::Printf(TEXT("%s submissions invalid (-MaxRowsPerFrame=%d must not exceed -MaxPendingRows=%d)"),
            *Count(Window.Invalid), MaxRowsPerFrame, MaxPendingRows));
    if (Window.Failed + Window.Expired > 0)
        Causes.Add(FString::Printf(TEXT("%s rows failed or expired"), *Count(Window.Failed + Window.Expired)));
    if (Window.Rejected > 0)
        Causes.Add(FString::Printf(TEXT("queue full: %s submissions rejected (limit %d rows)"), *Count(Window.Rejected), MaxPendingRows));
    if (Window.Deferred > Window.Rejected * MaxRowsPerFrame)
        Causes.Add(FString::Printf(TEXT("per-frame cap: %d rows x %.0f FPS = %s decisions/s max"),
            MaxRowsPerFrame, FramesPerSecond, *Count(FMath::RoundToInt64(MaxRowsPerFrame * FramesPerSecond))));
    if (Window.Stale > 0)
        Causes.Add(FString::Printf(TEXT("%s results older than %.0f ms discarded"), *Count(Window.Stale), Gather ? Gather->MaxActionAgeMs : 0.f));
    if (Display.ModelMsPerSecond > .8 * 1000 * WorkerCount)
        Causes.Add(FString::Printf(TEXT("workers %.0f%% busy"), Display.ModelMsPerSecond / (10. * WorkerCount)));
    if (Causes.IsEmpty())
        Causes.Add(FString::Printf(TEXT("frame time %.1f ms too long for %.0f Hz decisions"), FrameMs, 1.0 / PeriodSeconds));
    return FString::Join(Causes, TEXT("; "));
}

bool ACrowdComparisonHost::LoadPolicy()
{
    const bool bLearnedMigration = MigrationDay && MigrationDay->UsesModel();
    const bool bTurns = MigrationDay && MigrationDay->GetTurns();
    FString Path = FPaths::ProjectContentDir() / (bTurns ? TEXT("Models/turn_policy.onnx") : bLearnedMigration
        ? TEXT("Models/migration_policy.onnx") : TEXT("Models/crowd_policy.onnx"));
    TArray<uint8> Bytes;
    if (!FFileHelper::LoadFileToArray(Bytes, *Path)) { Fail(TEXT("Missing policy file ") + Path); return false; }
    ModelData = NewObject<UNNEModelData>(this);
    ModelData->Init(TEXT("onnx"), Bytes);
    return true;
}

void ACrowdComparisonHost::BuildScene()
{
    UWorld* World = GetWorld();
    TActorIterator<ACampusScene> It(World);
    if (It) Campus = *It;
    if (!Campus) Campus = World->SpawnActor<ACampusScene>();
    if (Campus->Nodes.IsEmpty()) Campus->LoadCampus();
    HalfSize = Campus->Extent;
    TActorIterator<ADirectionalLight> SunIt(World);
    auto* Sun = SunIt ? *SunIt : World->SpawnActor<ADirectionalLight>(FVector::ZeroVector, FRotator(-50, -35, 0));
    auto* SunLight = CastChecked<UDirectionalLightComponent>(Sun->GetLightComponent());
    SunLight->SetMobility(EComponentMobility::Movable);
    SunLight->SetIntensity(6);
    SunLight->SetAtmosphereSunLight(true);
    TActorIterator<ASkyAtmosphere> SkyIt(World);
    if (!SkyIt) World->SpawnActor<ASkyAtmosphere>();
    TActorIterator<ASkyLight> AmbientIt(World);
    auto* Ambient = AmbientIt ? *AmbientIt : World->SpawnActor<ASkyLight>();
    Ambient->GetLightComponent()->SetMobility(EComponentMobility::Movable);
    Ambient->GetLightComponent()->bRealTimeCapture = true;
    Ambient->GetLightComponent()->SetIntensity(1.5f);
    Ambient->GetLightComponent()->RecaptureSky();

    auto* Camera = World->SpawnActor<ACameraActor>(FVector(0, HalfSize * 1.55, HalfSize * 1.25), FRotator::ZeroRotator);
    Camera->SetActorRotation((FVector(0, -HalfSize * .1, 0) - Camera->GetActorLocation()).Rotation());
    CampusCamera = Camera;
    Camera->GetCameraComponent()->SetFieldOfView(55);
    Camera->GetCameraComponent()->bConstrainAspectRatio = false;
    if (auto* Player = World->GetFirstPlayerController()) Player->SetViewTarget(Camera);
}

void ACrowdComparisonHost::SetMode(ECrowdInferenceMode NewMode)
{
    Mode = NewMode;
    if (bSplitDemo) UE_LOG(LogTemp, Display, TEXT("Crowd mode: %s"),
        Mode == ECrowdInferenceMode::Background ? TEXT("ScholaMass background") : TEXT("Synchronous"));
    Counters = WindowCounters = CaptureCounters = {};
    if (Scheduler) WindowSchedulerStart = ModeSchedulerStart = CaptureSchedulerStart = Scheduler->GetStats();
    WindowLossStart = ModeLossStart = CaptureLossStart = ReadLosses();
    if (MigrationDay && MigrationDay->GetTurns()) WindowTurnRequests = MigrationDay->GetTurns()->Stats.Requested;
    WindowsInMode = 0;
    bShortfallLogged = false;
    Display.Shortfall.Reset();
    Display.TargetFraction = 1;
    if (Mode == ECrowdInferenceMode::Synchronous && Sync && !Entities.IsEmpty())
    {
        if (MigrationDay && Service) UpdateMigrationSchedules(false);
        // Agents keep their phase offsets; restart the synchronous schedule from now.
        auto& Manager = EntityManager(GetWorld());
        const double Now = FPlatformTime::Seconds();
        for (const FMassEntityHandle Entity : Entities)
        {
            auto& Agent = Manager.GetFragmentDataChecked<FCrowdAgentFragment>(Entity);
            if (MigrationDay)
                Agent.SyncNextDue = MigrationFixedStep > 0 ? MigrationDay->GetTime() : Now + PeriodSeconds * (Agent.Instance % 37) / 37.;
            else Agent.SyncNextDue = Now + FMath::Fmod(Agent.SyncNextDue - Started, PeriodSeconds);
        }
    }
}

void ACrowdComparisonHost::RunInference(float DeltaSeconds)
{
    auto& Manager = EntityManager(GetWorld());
    UE::Mass::FProcessingContext Context(Manager, DeltaSeconds);
    if (Mode == ECrowdInferenceMode::Synchronous)
    {
        UE::Mass::Executor::Run(*Sync, Context);
        if (Sync->bFailed) Fail(TEXT("Synchronous RunSync failed"));
    }
    else
    {
        // Apply results that finished since last frame, then submit newly due rows.
        UE::Mass::Executor::Run(*Apply, Context);
        if (MigrationDay && MigrationDay->GetTurns())
        {
            // Consume ready results first so a delayed frame does not retry
            // a decision that completed while the game thread was busy.
            MigrationDay->RetryOverdueTurns();
            UpdateMigrationSchedules(true);
        }
        UE::Mass::Executor::Run(*Gather, Context);
    }
}

void ACrowdComparisonHost::UpdateMigrationSchedules(bool bScheduleWalking)
{
    auto& Manager = EntityManager(GetWorld());
    FScopeLock Lock(&Service->Mutex);
    if (const auto* Turns = MigrationDay->GetTurns())
    {
        // Keep Mass slots only for students who are waiting for a junction result.
        for (int32 Index : ScheduledTurnStudents)
        {
            if (bScheduleWalking && Turns->WantsDecision(Index)) continue;
            const auto Entity = Entities[Index];
            Service->Bridge.ReleaseSlot(Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity));
            auto& Schedule = Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entity);
            Schedule.PeriodSeconds = 0;
            Schedule.bUrgent = false;
        }
        ScheduledTurnStudents.Reset();
        if (!bScheduleWalking) return;
        for (int32 Index : Turns->GetPendingStudents())
        {
            const auto Entity = Entities[Index];
            auto& Schedule = Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entity);
            auto& State = Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity);
            const int32 Generation = Turns->Generation(Index);
            if (Schedule.PeriodSeconds <= 0 || ScheduledMigrationGenerations[Index] != Generation)
            {
                Service->Bridge.ReleaseSlot(State);
                // On the periodic API, an urgent slot makes this one-shot request due.
                Schedule.PeriodSeconds = 10.f;
                Schedule.bUrgent = true;
                ScheduledMigrationGenerations[Index] = Generation;
            }
            else if (State.SubmittedSerial > 0)
            {
                // Suppress a second admission while this request is in flight.
                Schedule.NextDueTime = TNumericLimits<double>::Max();
            }
            ScheduledTurnStudents.Add(Index);
        }
        return;
    }
    for (int32 I = 0; I < Entities.Num(); ++I)
    {
        const auto Entity = Entities[I];
        auto& Schedule = Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entity);
        const bool bWalking = bScheduleWalking && MigrationDay->IsWalking(I);
        const int32 Generation = MigrationDay->GetGeneration(I);
        if ((!bWalking || ScheduledMigrationGenerations[I] != Generation) && Schedule.PeriodSeconds > 0)
        {
            // Invalidate outstanding rows when leaving the outdoor crowd.
            Service->Bridge.ReleaseSlot(Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity));
        }
        Schedule.PeriodSeconds = bWalking ? PeriodSeconds : 0;
        ScheduledMigrationGenerations[I] = Generation;
    }
}

void ACrowdComparisonHost::TickMigration(float DeltaSeconds, double Now, double FrameMs)
{
    if (bPaused || (MigrationDay->IsFinished() && ReportPath.IsEmpty())) return;

    const float TotalStep = MigrationFixedStep > 0 ? MigrationFixedStep : FMath::Min(.1f, DeltaSeconds) * MigrationTimeScale;
    const int32 Steps = FMath::Max(1, FMath::CeilToInt32(TotalStep / .1f));
    const float Step = TotalStep / Steps;
    const double PlanStart = FPlatformTime::Seconds();
    if (MigrationDay->UsesRoutePlanner()) MigrationDay->TickPlanner();
    const double MoveStart = FPlatformTime::Seconds();
    const int64 DecisionsBefore = MigrationDay->GetModelDecisions();
    const auto* Turns = MigrationDay->GetTurns();
    for (int32 I = 0; I < Steps && !MigrationDay->IsFinished(); ++I) MigrationDay->Step(Step, Transforms);
    const double InferenceStart = FPlatformTime::Seconds();

    if (MigrationDay->UsesModel())
    {
        if (Mode == ECrowdInferenceMode::Background && (!Turns || MigrationDay->IsFinished()))
            UpdateMigrationSchedules(!MigrationDay->IsFinished());
        if (!MigrationDay->IsFinished()) RunInference(TotalStep);
    }
    if (!Error.IsEmpty()) return;
    const double InferenceMs = ((MoveStart - PlanStart) + (FPlatformTime::Seconds() - InferenceStart)) * 1000;
    const double MovementMs = (InferenceStart - MoveStart) * 1000;
    if (MigrationDay->IsNavigation() && FrameMs > 0)
    {
        Display.FrameMs = Display.FrameMs > 0 ? FMath::Lerp(Display.FrameMs, FrameMs, .12) : FrameMs;
        Display.FramesPerSecond = 1000. / Display.FrameMs;
        if (MigrationDay->UsesLivePlanner()) Display.InferenceMs = FMath::Lerp(Display.InferenceMs, InferenceMs, .12);
    }
    CaptureMigrationPerformance(Now, FrameMs, InferenceMs, MovementMs);
    if (MigrationDay->HasTraceError()) { Fail(TEXT("Failed to write migration trace")); return; }

    if (FApp::CanEverRender() && !GUsingNullRHI)
    {
        if (MigrationDay->UsesModel())
        {
            Counters.Decisions += MigrationDay->GetModelDecisions() - DecisionsBefore;
            if (Turns)
            {
                WindowMigrationDemand += Turns->Stats.Requested - WindowTurnRequests;
                WindowTurnRequests = Turns->Stats.Requested;
            }
            else WindowMigrationDemand += MigrationDay->GetActive() * (MigrationFixedStep > 0 ? MigrationFixedStep : FrameMs / 1000.) / PeriodSeconds;
            UpdateDisplay(Now, FrameMs, InferenceMs, MovementMs);
        }
        if (!bNavigationMapView)
        {
            if (bSplitDemo) UpdateDestinationColors();
            KeepCrowdVisible();
            Crowd->BatchUpdateInstancesTransforms(0, Transforms, true, true, true);
        }
    }
    if (MigrationDay->IsFinished() && bExitAfterMigration) FPlatformMisc::RequestExitWithStatus(false, 0);
}

void ACrowdComparisonHost::CaptureMigrationPerformance(double Now, double FrameMs, double InferenceMs, double MovementMs)
{
    if (ReportPath.IsEmpty() || CaptureSeconds <= 0 || bReported) return;
    const auto* Turns = MigrationDay->GetTurns();
    if (!bCapturing)
    {
        if (Now - Started < WarmupSeconds) return;
        bCapturing = true;
        CaptureStart = Now;
        if (Turns) CaptureTurnStart = Turns->Stats;
        if (Sync) CaptureSyncCallsStart = Sync->ModelCalls;
        MigrationDecisionStart = MigrationDay->GetModelDecisions();
        if (MigrationDay->UsesLivePlanner()) CapturePlanningStart = MigrationDay->GetPlanningStats();
        if (Scheduler) CaptureSchedulerStart = Scheduler->GetStats();
        CaptureLossStart = ReadLosses();
        return;
    }
    CaptureFrameMs.Add(FrameMs);
    CaptureInferenceMs.Add(InferenceMs);
    CaptureMoveMs.Add(MovementMs);
    MigrationWalkingSeconds += MigrationDay->GetActive() * FrameMs / 1000.;
    if (MigrationDay->UsesModel() && !Turns)
        MigrationExpectedDecisions += MigrationDay->GetActive() * (MigrationFixedStep > 0 ? MigrationFixedStep : FrameMs / 1000.) / PeriodSeconds;
    const double Elapsed = Now - CaptureStart;
    if (Elapsed < CaptureSeconds) return;

    bReported = true;
    const int64 Decisions = MigrationDay->GetModelDecisions() - MigrationDecisionStart;
    const auto Loss = ReadLosses() - CaptureLossStart;
    auto Root = MakeShared<FJsonObject>();
    Root->SetStringField(TEXT("mode"), MigrationDay->IsNavigation() && !MigrationDay->UsesLivePlanner()
        ? TEXT("Precomputed") : !MigrationDay->UsesModel() && !MigrationDay->UsesLivePlanner() ? TEXT("Rules") :
        Mode == ECrowdInferenceMode::Background ? TEXT("Background") : TEXT("Sync"));
    Root->SetNumberField(TEXT("cohort_size"), Entities.Num());
    if (MigrationDay->IsNavigation() || Turns)
    {
        Root->SetNumberField(TEXT("completed_schedules"), MigrationDay->GetArrived());
        Root->SetNumberField(TEXT("completed_legs"), MigrationDay->GetLegsCompleted());
        Root->SetNumberField(TEXT("timed_out_students"), MigrationDay->GetTimedOut());
    }
    if (MigrationDay->UsesLivePlanner())
    {
        const auto End = MigrationDay->GetPlanningStats();
        const int64 Rows = End.ModelRows - CapturePlanningStart.ModelRows;
        const int64 Calls = End.ModelCalls - CapturePlanningStart.ModelCalls;
        const int64 Groups = End.InferenceGroups - CapturePlanningStart.InferenceGroups;
        Root->SetNumberField(TEXT("mean_game_thread_planning_ms"), Mean(CaptureInferenceMs));
        Root->SetNumberField(TEXT("p99_game_thread_planning_ms"), Quantile(CaptureInferenceMs, .99));
        Root->SetNumberField(TEXT("planned_routes"), End.Routes - CapturePlanningStart.Routes);
        Root->SetNumberField(TEXT("student_legs_planned"), End.StudentLegs - CapturePlanningStart.StudentLegs);
        Root->SetNumberField(TEXT("expanded_landmarks"), End.ExpandedNodes - CapturePlanningStart.ExpandedNodes);
        Root->SetNumberField(TEXT("fallback_routes"), End.FallbackRoutes - CapturePlanningStart.FallbackRoutes);
        Root->SetNumberField(TEXT("model_rows"), Rows);
        Root->SetNumberField(TEXT("model_calls"), Calls);
        Root->SetNumberField(TEXT("mean_rows_per_model_call"), Calls > 0 ? double(Rows) / Calls : 0.);
        Root->SetNumberField(TEXT("mean_inference_group_latency_ms"), Groups > 0
            ? (End.InferenceLatencyMs - CapturePlanningStart.InferenceLatencyMs) / Groups : 0.);
    }
    if (Turns)
    {
        const auto& Stats = Turns->Stats;
        const auto Backend = Scheduler->GetStats();
        const int64 Calls = Mode == ECrowdInferenceMode::Background
            ? Backend.BackendCalls - CaptureSchedulerStart.BackendCalls : Sync->ModelCalls - CaptureSyncCallsStart;
        const int64 Rows = Mode == ECrowdInferenceMode::Background
            ? Backend.BackendRows - CaptureSchedulerStart.BackendRows : Calls;
        Root->SetStringField(TEXT("policy"), TEXT("Junction turns"));
        Root->SetNumberField(TEXT("turn_requests"), Stats.Requested - CaptureTurnStart.Requested);
        Root->SetNumberField(TEXT("learned_turns_applied"), Stats.Applied - CaptureTurnStart.Applied);
        Root->SetNumberField(TEXT("resolved_turn_requests"), Stats.Resolved - CaptureTurnStart.Resolved);
        Root->SetNumberField(TEXT("retraced_edges"), Stats.RetracedEdges - CaptureTurnStart.RetracedEdges);
        Root->SetNumberField(TEXT("inference_retries"), Stats.Retries - CaptureTurnStart.Retries);
        Root->SetNumberField(TEXT("p99_turn_latency_ms"), Stats.P99LatencyMsSince(CaptureTurnStart));
        Root->SetNumberField(TEXT("model_rows"), Rows);
        Root->SetNumberField(TEXT("model_calls"), Calls);
        MigrationExpectedDecisions = Stats.Resolved - CaptureTurnStart.Resolved;
    }
    Root->SetNumberField(TEXT("mean_walking_students"), MigrationWalkingSeconds / Elapsed);
    Root->SetNumberField(TEXT("mean_game_thread_inference_ms"), Mean(CaptureInferenceMs));
    Root->SetNumberField(TEXT("p99_game_thread_inference_ms"), Quantile(CaptureInferenceMs, .99));
    Root->SetNumberField(TEXT("mean_movement_ms"), Mean(CaptureMoveMs));
    Root->SetNumberField(TEXT("p99_frame_ms"), Quantile(CaptureFrameMs, .99));
    Root->SetNumberField(TEXT("fps"), CaptureFrameMs.Num() / Elapsed);
    if (!MigrationDay->UsesLivePlanner())
    {
        Root->SetNumberField(TEXT("decisions_per_second"), Decisions / Elapsed);
        Root->SetNumberField(TEXT("fraction_of_active_target"), MigrationExpectedDecisions > 0 ? Decisions / MigrationExpectedDecisions : 0);
        Root->SetNumberField(TEXT("dropped_rows"), Loss.Lost());
        Root->SetNumberField(TEXT("rejected_requests"), Loss.Rejected);
    }
    if (Scheduler && Mode == ECrowdInferenceMode::Background)
    {
        const auto End = Scheduler->GetStats();
        Root->SetNumberField(TEXT("mean_rows_per_model_call"), RowsPerCall(End, CaptureSchedulerStart));
    }
    else if (MigrationDay->UsesModel()) Root->SetNumberField(TEXT("mean_rows_per_model_call"), Decisions > 0 ? 1 : 0);
    FString Json;
    FJsonSerializer::Serialize(Root, TJsonWriterFactory<>::Create(&Json));
    if (!FFileHelper::SaveStringToFile(Json, *ReportPath))
    {
        Fail(TEXT("Could not write migration performance report ") + ReportPath);
        return;
    }
    FPlatformMisc::RequestExitWithStatus(false, 0);
}

void ACrowdComparisonHost::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    if (Entities.IsEmpty() || !Error.IsEmpty()) return;
    if (auto* Player = GetWorld()->GetFirstPlayerController(); MigrationFixedStep <= 0 && Player &&
        (!MigrationDay || MigrationDay->UsesModel()) && Player->WasInputKeyJustPressed(EKeys::Tab))
        SetMode(Mode == ECrowdInferenceMode::Background ? ECrowdInferenceMode::Synchronous : ECrowdInferenceMode::Background);

    const double Now = FPlatformTime::Seconds();
    const double FrameMs = (Now - PreviousTick) * 1000;
    PreviousTick = Now;

    UpdateCamera(DeltaSeconds);
    if (MigrationDay)
    {
        TickMigration(DeltaSeconds, Now, FrameMs);
        return;
    }
    const FCrowdFrameCounters Before = Counters;
    RunInference(DeltaSeconds);
    const double InferenceMs = (FPlatformTime::Seconds() - Now) * 1000;

    const double MoveStart = FPlatformTime::Seconds();
    UE::Mass::FProcessingContext MoveContext(EntityManager(GetWorld()), DeltaSeconds);
    if (!bPaused) UE::Mass::Executor::Run(*Move, MoveContext);
    const double MoveMs = (FPlatformTime::Seconds() - MoveStart) * 1000;
    KeepCrowdVisible();
    Crowd->BatchUpdateInstancesTransforms(0, Transforms, true, true, true);
    UpdateDisplay(Now, FrameMs, InferenceMs, MoveMs);
    if (!bScreenshotTaken && Now-Started >= FMath::Min(WarmupSeconds * .5, 5.0))
    {
        FString Screenshot;
        if (FParse::Value(FCommandLine::Get(), TEXT("Screenshot="), Screenshot))
            FScreenshotRequest::RequestScreenshot(Screenshot, !FParse::Param(FCommandLine::Get(), TEXT("CleanScreenshot")), false);
        bScreenshotTaken = true;
    }
    if (ReportPath.IsEmpty() || CaptureSeconds <= 0 || bReported) return;
    if (!bCapturing && Now - Started >= WarmupSeconds)
    {
        bCapturing = true;
        CaptureStart = Now;
        CaptureCounters = {};
        if (Scheduler) CaptureSchedulerStart = Scheduler->GetStats();
        CaptureLossStart = ReadLosses();
    }
    else if (bCapturing)
    {
        CaptureFrameMs.Add(FrameMs);
        CaptureInferenceMs.Add(InferenceMs);
        CaptureMoveMs.Add(MoveMs);
        CaptureCounters.Decisions += Counters.Decisions - Before.Decisions;
        CaptureCounters.LatencySumMs += Counters.LatencySumMs - Before.LatencySumMs;
        if (Now - CaptureStart >= CaptureSeconds) WriteReport();
    }
}

void ACrowdComparisonHost::UpdateDisplay(double Now, double FrameMs, double InferenceMs, double MoveMs)
{
    ++WindowFrames;
    WindowFrameMs += FrameMs;
    WindowInferenceMs += InferenceMs;
    WindowMoveMs += MoveMs;
    RecentFrameMs.Add(FrameMs);
    if (RecentFrameMs.Num() > 600) RecentFrameMs.RemoveAt(0, RecentFrameMs.Num() - 600, EAllowShrinking::No);
    const double Elapsed = Now - WindowStart;
    if (Elapsed < 1) return;
    const int64 Decisions = Counters.Decisions - WindowCounters.Decisions;
    Display.FramesPerSecond = WindowFrames / Elapsed;
    Display.FrameMs = WindowFrameMs / WindowFrames;
    Display.FrameP99Ms = Quantile(RecentFrameMs, .99);
    Display.InferenceMs = WindowInferenceMs / WindowFrames;
    Display.MoveMs = WindowMoveMs / WindowFrames;
    Display.DecisionsPerSecond = Decisions / Elapsed;
    Display.LatencyMs = Decisions > 0 ? (Counters.LatencySumMs - WindowCounters.LatencySumMs) / Decisions : 0;
    if (Scheduler && Mode == ECrowdInferenceMode::Background)
    {
        const FNPCBackgroundStats Stats = Scheduler->GetStats();
        Display.RowsPerModelCall = RowsPerCall(Stats, WindowSchedulerStart);
        Display.ModelMsPerSecond = ServiceMs(Stats, WindowSchedulerStart) / Elapsed;
        WindowSchedulerStart = Stats;
    }
    else
    {
        Display.RowsPerModelCall = 1;
        Display.ModelMsPerSecond = WindowInferenceMs / Elapsed;
    }
    const FCrowdLosses Losses = ReadLosses();
    const FCrowdLosses Window = Losses - WindowLossStart;
    WindowLossStart = Losses;
    const FCrowdLosses SinceMode = Losses - ModeLossStart;
    Display.DroppedRows = Mode == ECrowdInferenceMode::Background ? SinceMode.Lost() + SinceMode.Rejected + SinceMode.Invalid : 0;
    Display.LossesPerSecond = {int64(Window.Deferred / Elapsed), int64(Window.Rejected / Elapsed), int64(Window.Invalid / Elapsed),
        int64(Window.Expired / Elapsed), int64(Window.Failed / Elapsed), int64(Window.Stale / Elapsed)};
    const double Expected = MigrationDay ? WindowMigrationDemand : GetTargetDecisionsPerSecond() * Elapsed;
    Display.TargetFraction = Expected > 0 ? Decisions / Expected : 1;
    WindowMigrationDemand = 0;
    // The first windows after start or a mode switch include pipeline fill.
    if (++WindowsInMode > 2 && (!MigrationDay || !MigrationDay->GetTurns()) && Display.TargetFraction < TargetTolerance)
    {
        Display.Shortfall = DiagnoseShortfall(Window, Display.FrameMs);
        if (!bShortfallLogged)
            UE_LOG(LogTemp, Warning, TEXT("Crowd: delivering %.0f%% of the target %.0f decisions/s; limited by %s"),
                Display.TargetFraction * 100, Expected / Elapsed, *Display.Shortfall);
        bShortfallLogged = true;
    }
    else if (WindowsInMode > 2)
    {
        if (bShortfallLogged) UE_LOG(LogTemp, Display, TEXT("Crowd: target decision rate met again"));
        Display.Shortfall.Reset();
        bShortfallLogged = false;
    }
    WindowCounters = Counters;
    WindowStart = Now;
    WindowFrames = 0;
    WindowFrameMs = WindowInferenceMs = WindowMoveMs = 0;
}

double ACrowdComparisonHost::GetCaptureProgress(FString& OutPhase) const
{
    if (ReportPath.IsEmpty() || CaptureSeconds <= 0) return -1;
    const double Now = FPlatformTime::Seconds();
    if (!bCapturing) { OutPhase = TEXT("Warming up"); return FMath::Clamp((Now - Started) / FMath::Max(WarmupSeconds, .01), 0., 1.); }
    OutPhase = TEXT("Measuring");
    return FMath::Clamp((Now - CaptureStart) / CaptureSeconds, 0., 1.);
}

void ACrowdComparisonHost::WriteReport()
{
    bReported = true;
    const double Elapsed = FPlatformTime::Seconds() - CaptureStart;
    const int64 Decisions = CaptureCounters.Decisions;
    auto Root = MakeShared<FJsonObject>();
    Root->SetStringField(TEXT("mode"), Mode == ECrowdInferenceMode::Background ? TEXT("background") : TEXT("synchronous"));
    Root->SetNumberField(TEXT("agents"), Entities.Num());
    Root->SetStringField(TEXT("scene"), TEXT("50H_NORTH campus"));
    Root->SetNumberField(TEXT("cad_buildings"), Campus->BuildingCount);
    Root->SetNumberField(TEXT("route_nodes"), Campus->Nodes.Num());
    Root->SetNumberField(TEXT("decision_hz_per_agent"), 1.0 / PeriodSeconds);
    Root->SetNumberField(TEXT("target_decisions_per_second"), GetTargetDecisionsPerSecond());
    Root->SetNumberField(TEXT("max_fps"), IConsoleManager::Get().FindConsoleVariable(TEXT("t.MaxFPS"))->GetFloat());
    Root->SetNumberField(TEXT("capture_seconds"), Elapsed);
    Root->SetNumberField(TEXT("frames"), CaptureFrameMs.Num());
    Root->SetNumberField(TEXT("fps"), CaptureFrameMs.Num() / Elapsed);
    Root->SetNumberField(TEXT("frame_ms_mean"), Mean(CaptureFrameMs));
    Root->SetNumberField(TEXT("frame_ms_p50"), Quantile(CaptureFrameMs, .5));
    Root->SetNumberField(TEXT("frame_ms_p95"), Quantile(CaptureFrameMs, .95));
    Root->SetNumberField(TEXT("frame_ms_p99"), Quantile(CaptureFrameMs, .99));
    Root->SetNumberField(TEXT("inference_game_thread_ms_mean"), Mean(CaptureInferenceMs));
    Root->SetNumberField(TEXT("inference_game_thread_ms_p99"), Quantile(CaptureInferenceMs, .99));
    Root->SetNumberField(TEXT("decisions_per_second"), Decisions / Elapsed);
    const double TargetFraction = Decisions / (GetTargetDecisionsPerSecond() * Elapsed);
    const bool bTargetMet = TargetFraction >= TargetTolerance;
    Root->SetNumberField(TEXT("target_fraction"), TargetFraction);
    Root->SetBoolField(TEXT("target_met"), bTargetMet);
    Root->SetNumberField(TEXT("movement_ms_mean"), Mean(CaptureMoveMs));
    Root->SetNumberField(TEXT("workers"), WorkerCount);
    Root->SetNumberField(TEXT("max_batch"), MaxBatchSize);
    Root->SetNumberField(TEXT("max_rows_per_frame"), MaxRowsPerFrame);
    Root->SetNumberField(TEXT("max_pending_rows"), MaxPendingRows);
    const FCrowdLosses Losses = ReadLosses() - CaptureLossStart;
    if (Mode == ECrowdInferenceMode::Background)
    {
        Root->SetNumberField(TEXT("deferred_rows"), double(Losses.Deferred));
        Root->SetNumberField(TEXT("rejected_submissions"), double(Losses.Rejected + Losses.Invalid));
        Root->SetNumberField(TEXT("expired_rows"), double(Losses.Expired));
        Root->SetNumberField(TEXT("failed_rows"), double(Losses.Failed));
        Root->SetNumberField(TEXT("stale_results_discarded"), double(Losses.Stale));
    }
    const FString Shortfall = bTargetMet ? FString() : DiagnoseShortfall(Losses, Mean(CaptureFrameMs));
    if (!bTargetMet) Root->SetStringField(TEXT("shortfall"), Shortfall);
    Root->SetNumberField(TEXT("action_latency_ms_mean"), Decisions > 0 ? CaptureCounters.LatencySumMs / Decisions : 0);
    if (Scheduler && Mode == ECrowdInferenceMode::Background)
    {
        const FNPCBackgroundStats Stats = Scheduler->GetStats();
        Root->SetNumberField(TEXT("rows_per_model_call"), RowsPerCall(Stats, CaptureSchedulerStart));
        Root->SetNumberField(TEXT("model_calls"), double(Stats.BackendCalls - CaptureSchedulerStart.BackendCalls));
        Root->SetNumberField(TEXT("model_ms_per_second"), ServiceMs(Stats, CaptureSchedulerStart) / Elapsed);
        Root->SetNumberField(TEXT("dropped_rows"), double(Dropped(Stats, CaptureSchedulerStart) + Losses.Stale));
        Root->SetNumberField(TEXT("submitted_requests"), double(Stats.SubmittedRequests - CaptureSchedulerStart.SubmittedRequests));
        Root->SetNumberField(TEXT("busy_rejections"), double(Stats.BusyRejections - CaptureSchedulerStart.BusyRejections));
        Root->SetNumberField(TEXT("invalid_submissions"), double(Stats.InvalidSubmissions - CaptureSchedulerStart.InvalidSubmissions));
    }
    else
    {
        Root->SetNumberField(TEXT("rows_per_model_call"), 1);
        Root->SetNumberField(TEXT("model_calls"), double(Decisions));
        Root->SetNumberField(TEXT("model_ms_per_second"), Mean(CaptureInferenceMs) * CaptureInferenceMs.Num() / Elapsed);
        Root->SetNumberField(TEXT("dropped_rows"), 0);
    }
    Root->SetStringField(TEXT("cpu"), FPlatformMisc::GetCPUBrand().TrimStartAndEnd());
    Root->SetStringField(TEXT("gpu"), GRHIAdapterName);
    TArray<TSharedPtr<FJsonValue>> Frames;
    for (double Value : CaptureFrameMs) Frames.Add(MakeShared<FJsonValueNumber>(FMath::RoundToDouble(Value * 1000) / 1000));
    Root->SetArrayField(TEXT("frame_ms"), Frames);
    FString Json;
    FJsonSerializer::Serialize(Root, TJsonWriterFactory<>::Create(&Json));
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(ReportPath), true);
    const bool bSaved = FFileHelper::SaveStringToFile(Json, *ReportPath);
    UE_LOG(LogTemp, Display, TEXT("Crowd comparison report %s: %s"), bSaved ? TEXT("written") : TEXT("FAILED"), *ReportPath);
    if (!bTargetMet)
        UE_LOG(LogTemp, Error, TEXT("Crowd: target decision rate missed: %.0f of %.0f decisions/s (%.0f%%); limited by %s"),
            Decisions / Elapsed, GetTargetDecisionsPerSecond(), TargetFraction * 100, *Shortfall);
    // Exit 3: report written, but the crowd did not receive the decisions it asked for.
    FPlatformMisc::RequestExitWithStatus(false, !bSaved ? 1 : bTargetMet ? 0 : 3);
}

void ACrowdComparisonHost::Fail(const FString& Message)
{
    Error = Message;
    UE_LOG(LogTemp, Error, TEXT("Crowd comparison: %s"), *Message);
    if (!ReportPath.IsEmpty() || FParse::Param(FCommandLine::Get(), TEXT("ExitAfterMigrations")))
        FPlatformMisc::RequestExitWithStatus(false, 1);
}

void ACrowdComparisonHost::EndPlay(EEndPlayReason::Type Reason)
{
    if (auto* Subsystem = GetWorld()->GetSubsystem<UMassEntitySubsystem>())
    {
        auto& Manager = Subsystem->GetMutableEntityManager();
        TArray<FMassEntityHandle> Valid;
        for (const FMassEntityHandle Entity : Entities)
            if (Manager.IsEntityValid(Entity)) Valid.Add(Entity);
        if (!Valid.IsEmpty()) Manager.BatchDestroyEntities(Valid);
    }
    Entities.Empty();
    if (Sync) { Sync->Instance.Reset(); Sync->Model.Reset(); }
    Super::EndPlay(Reason);
}

void ACrowdComparisonHost::UpdateDestinationColors()
{
    if (StudentColorKeys.IsEmpty()) return;
    const auto& Day = *MigrationDay->GetTurns();
    ActiveDestinations.Init(false, ActiveDestinations.Num());
    for (int32 I = 0; I < StudentColorKeys.Num(); ++I)
    {
        if (!Day.IsStudentVisible(I) && !Day.IsFirstClassOnly()) continue;
        const int32 Key = Day.GetStopColorKey(Day.GetTargetStop(I));
        if (Key >= 0) ActiveDestinations[Key] = true;
        if (StudentColorKeys[I] == Key) continue;
        const auto Color = FCampusTurnDay::DestinationColor(Key);
        const float RGB[] = {Color.R, Color.G, Color.B};
        Crowd->SetCustomData(I, MakeArrayView(RGB));
        StudentColorKeys[I] = Key;
    }
    Campus->SetActiveDestinations(ActiveDestinations);
}

void ACrowdComparisonHost::KeepCrowdVisible()
{
    // A 60 cm student is under a pixel wide from a few hundred metres and
    // disappears under temporal AA. Distant students grow so each stays at
    // least MinPixels wide; nearby ones keep their real 60 x 180 cm size.
    constexpr float Width = 60, Height = 180, MinPixels = 3;
    const auto* Player = GetWorld()->GetFirstPlayerController();
    if (!Player || !CampusCamera) return;
    int32 ViewWidth = 0, ViewHeight = 0;
    Player->GetViewportSize(ViewWidth, ViewHeight);
    if (ViewWidth <= 0) return;
    if (bSplitDemo)
    {
        const float Size = FMath::Max(1.f, (bFourDemo ? 3.f : 1.5f) * CampusCamera->GetCameraComponent()->OrthoWidth / (ViewWidth * .5f * Width));
        const auto& Day = *MigrationDay->GetTurns();
        for (int32 I = 0; I < Transforms.Num(); ++I)
        {
            if (!Day.IsStudentVisible(I)) continue;
            const auto P = Day.GetStudentPosition(I);
            Transforms[I].SetLocation(FVector(100 * P.X, 100 * P.Y, 4.f + Height * Size / 2.f));
            Transforms[I].SetScale3D(FVector(Width, Width, Height) * (Size / 100.f));
        }
        return;
    }
    // The campus camera is always the view target; read it after this frame's move.
    const FVector Eye = CampusCamera->GetActorLocation();
    const float HalfFov = FMath::DegreesToRadians(CampusCamera->GetCameraComponent()->FieldOfView) / 2;
    const float ScalePerCm = MinPixels * 2 * FMath::Tan(HalfFov) / ViewWidth / Width;
    ParallelFor(Transforms.Num(), [&](int32 I)
    {
        FTransform& Transform = Transforms[I];
        if (MigrationDay && !MigrationDay->IsWalking(I)) return;
        FVector Location = Transform.GetLocation();
        Location.Z = Height / 2;
        const float Scale = FMath::Max(1.f, float(FVector::Dist(Eye, Location)) * ScalePerCm);
        Location.Z = 4 + Height / 2 * Scale;
        Transform.SetLocation(Location);
        Transform.SetScale3D(FVector(Width, Width, Height) * (Scale / 100));
    }, Transforms.Num() < 4096 ? EParallelForFlags::ForceSingleThread : EParallelForFlags::None);
}

void ACrowdComparisonHost::UpdateCamera(float DeltaSeconds)
{
    auto* Player = GetWorld()->GetFirstPlayerController();
    if (!Player || !CampusCamera) return;
    if (Player->WasInputKeyJustPressed(EKeys::SpaceBar)) bPaused = !bPaused;
    if (bSplitDemo)
    {
        const bool bSelectFour = Player->WasInputKeyJustPressed(EKeys::F1);
        const bool bSelectScale = Player->WasInputKeyJustPressed(EKeys::F2);
        if (!bSwitchingDemo && ((bSelectFour && !bFourDemo) || (bSelectScale && bFourDemo)))
        {
            bSwitchingDemo = true;
            const auto Map = UGameplayStatics::GetCurrentLevelName(this, true);
            UGameplayStatics::OpenLevel(this, FName(*Map), true, bSelectFour ? TEXT("CampusDemo=Four") : TEXT("CampusDemo=Scale"));
            return;
        }
        if (Campus->MapSize.IsNearlyZero()) return;
        const auto Rotation = FCampusTurnDay::OverviewRotation();
        auto* Camera = CampusCamera->GetCameraComponent();
        if (PreviousCameraPreset == INDEX_NONE)
        {
            const auto Center = Campus->MapOrigin + Campus->MapSize * .5f;
            const FVector Target(Center.X * 100, Center.Y * 100, 0);
            CampusCamera->SetActorLocationAndRotation(Target - Rotation.Vector() * 210000.0, Rotation);
            Camera->SetProjectionMode(ECameraProjectionMode::Orthographic);
            Camera->bAutoCalculateOrthoPlanes = false;
            Camera->SetOrthoNearClipPlane(0.f);
            Camera->SetOrthoFarClipPlane(425000.f);
            PreviousCameraPreset = 0;
        }
        int32 Width = 0, Height = 0;
        Player->GetViewportSize(Width, Height);
        if (Height <= 0) return;
        const float Aspect = .5f * Width / Height;
        const auto Right = CampusCamera->GetActorRightVector().GetAbs();
        const auto Up = CampusCamera->GetActorUpVector().GetAbs();
        const auto Size = Campus->MapSize;
        const float Horizontal = Right.X * Size.X + Right.Y * Size.Y, Vertical = Up.X * Size.X + Up.Y * Size.Y;
        Camera->SetOrthoWidth(100.f * FMath::Max(Horizontal / .94f, (Vertical + 95.f) * Aspect / .86f));
        return;
    }
    if (Player->WasInputKeyJustPressed(EKeys::One) || Player->WasInputKeyJustPressed(EKeys::R))
    {
        CameraPreset = 0;
        PreviousCameraPreset = INDEX_NONE;
    }
    if (Player->WasInputKeyJustPressed(EKeys::Two)) CameraPreset = 1;
    if (Player->WasInputKeyJustPressed(EKeys::Three)) CameraPreset = 2;
    if (Player->WasInputKeyJustPressed(EKeys::Four))
    {
        CameraPreset = 3;
        PreviousCameraPreset = -1; // Replay the same camera path for either inference mode.
    }
    if (Player->WasInputKeyJustPressed(EKeys::N)) bShowRoutes = !bShowRoutes;
    if (bShowRoutes && (!MigrationDay || !MigrationDay->IsNavigation()))
    {
        for (int32 I=0; I<Campus->Nodes.Num(); ++I)
        {
            const auto& Node = Campus->Nodes[I];
            for (int32 J : Node.Links) if (J>I)
            {
                const auto& End = Campus->Nodes[J];
                DrawDebugLine(GetWorld(), FVector(Node.Position.X,Node.Position.Y,30), FVector(End.Position.X,End.Position.Y,30), FColor(50,230,205), false, -1, 0, 5);
            }
        }
    }
    auto* Camera = CampusCamera->GetCameraComponent();
    const bool bMapView = CameraPreset == 0 && NavigationGateView < 0 && MigrationDay && MigrationDay->IsNavigation();
    if (PreviousCameraPreset != CameraPreset)
    {
        FVector Center = FVector::ZeroVector;
        FVector StreetLook = FVector::ZeroVector;
        float Closest = MAX_flt;
        for (const auto& Node : Campus->Nodes)
        {
            const float Distance = (Node.Position - FVector2f(-15000,18000)).SizeSquared();
            if (Distance < Closest)
            {
                Closest = Distance;
                Center = FVector(Node.Position.X, Node.Position.Y, 80);
                const auto& Neighbour = Campus->Nodes[Node.Links[0]];
                StreetLook = FVector(Neighbour.Position.X, Neighbour.Position.Y, 150);
            }
        }
        // University College is the landmark checked against the recovered CAD regions.
        if (CameraPreset == 1) Center = FVector(-1000, 22500, 1000);
        CameraPanStarted = FPlatformTime::Seconds();
        // Scripts/render-video.sh waits for this line before it starts capturing.
        if (CameraPreset == 3) UE_LOG(LogTemp, Display, TEXT("Camera flight started"));
        const FVector Locations[] = {FVector(0, HalfSize * 1.55, HalfSize * 1.25), Center + FVector(0, 12000, 9000), Center + FVector(0,0,120), Center};
        FVector Location = Locations[CameraPreset];
        FVector Target = CameraPreset == 0 ? FVector::ZeroVector : CameraPreset == 2 ? StreetLook : Center;
        if (CameraPreset == 0 && NavigationGateView >= 0 && MigrationDay && MigrationDay->IsNavigation())
        {
            const auto Stop = MigrationDay->GetNavigationGate(NavigationGateView);
            const FVector Gate(100 * Stop.X, 100 * Stop.Y, 0);
            const FVector Inward = (-Gate).GetSafeNormal2D();
            Location = Gate - Inward * 1800 + FVector(0, 0, 5500);
            Target = Gate + Inward * 1200;
        }
        Camera->SetProjectionMode(bMapView ? ECameraProjectionMode::Orthographic : ECameraProjectionMode::Perspective);
        Camera->bOverrideAspectRatioAxisConstraint = bMapView;
        if (bNavigationMapView != bMapView)
        {
            bNavigationMapView = bMapView;
            Campus->SetActorHiddenInGame(bMapView);
            Crowd->SetVisibility(!bMapView);
        }
        if (bMapView)
        {
            int32 Width = 0, Height = 0;
            Player->GetViewportSize(Width, Height);
            const float Aspect = Width > 0 && Height > 0 ? float(Width) / Height : 16.f / 9.f;
            Camera->SetAspectRatio(Aspect);
            Camera->SetAspectRatioAxisConstraint(AspectRatio_MaintainXFOV);
            const auto& Grid = MigrationDay->GetNavigationGrid();
            const float MapWidth = Grid.Width * Grid.Cell;
            const float MapHeight = Grid.Height * Grid.Cell;
            Camera->SetOrthoWidth(110.f * FMath::Max(MapWidth, MapHeight * Aspect));
            Camera->bAutoCalculateOrthoPlanes = false;
            Camera->SetOrthoNearClipPlane(0.f);
            Camera->SetOrthoFarClipPlane(4 * HalfSize);
            CampusCamera->SetActorLocation(FVector(100 * (Grid.Origin.X + .5f * MapWidth),
                100 * (Grid.Origin.Y + .5f * MapHeight), 2 * HalfSize));
            CampusCamera->SetActorRotation(FRotator(-90, -90, 0));
        }
        else
        {
            CampusCamera->SetActorLocation(Location);
            CampusCamera->SetActorRotation((Target - Location).Rotation());
        }
        PreviousCameraPreset = CameraPreset;
    }
    if (bMapView)
    {
        const int32 Horizontal = int32(Player->IsInputKeyDown(EKeys::D)) - int32(Player->IsInputKeyDown(EKeys::A));
        const int32 Vertical = int32(Player->IsInputKeyDown(EKeys::W)) - int32(Player->IsInputKeyDown(EKeys::S));
        const FVector Pan = CampusCamera->GetActorRightVector() * Horizontal + CampusCamera->GetActorUpVector() * Vertical;
        CampusCamera->AddActorWorldOffset(Pan.GetClampedToMaxSize(1) * Camera->OrthoWidth * DeltaSeconds * .55f);
        const int32 Zoom = int32(Player->IsInputKeyDown(EKeys::Q)) - int32(Player->IsInputKeyDown(EKeys::E));
        if (Zoom)
            Camera->SetOrthoWidth(FMath::Clamp(Camera->OrthoWidth * FMath::Exp(Zoom * 1.5f * DeltaSeconds),
                5000.f, 5.f * HalfSize));
        return;
    }
    if (CameraPreset == 3)
    {
        // Wall time gives both inference modes the same flight.
        FVector Location, Target;
        const double KeyTime = FlightKeys[UE_ARRAY_COUNT(FlightKeys) - 1].Time;
        EvaluateFlight((FPlatformTime::Seconds() - CameraPanStarted) * KeyTime / FlightSeconds, Location, Target);
        CampusCamera->SetActorLocation(Location);
        CampusCamera->SetActorRotation((Target - Location).Rotation());
        return;
    }
    FVector Motion = FVector::ZeroVector;
    if (Player->IsInputKeyDown(EKeys::W)) Motion += CampusCamera->GetActorForwardVector();
    if (Player->IsInputKeyDown(EKeys::S)) Motion -= CampusCamera->GetActorForwardVector();
    if (Player->IsInputKeyDown(EKeys::D)) Motion += CampusCamera->GetActorRightVector();
    if (Player->IsInputKeyDown(EKeys::A)) Motion -= CampusCamera->GetActorRightVector();
    if (Player->IsInputKeyDown(EKeys::E)) Motion.Z += 1;
    if (Player->IsInputKeyDown(EKeys::Q)) Motion.Z -= 1;
    CampusCamera->AddActorWorldOffset(Motion * DeltaSeconds * (Player->IsInputKeyDown(EKeys::LeftShift) ? 18000 : 4000));
    float MouseX=0, MouseY=0;
    Player->GetInputMouseDelta(MouseX, MouseY);
    if (Player->IsInputKeyDown(EKeys::RightMouseButton))
    {
        FRotator Rotation = CampusCamera->GetActorRotation();
        Rotation.Yaw += MouseX * .15f;
        Rotation.Pitch = FMath::Clamp(Rotation.Pitch + MouseY * .15f, -85.f, 15.f);
        CampusCamera->SetActorRotation(Rotation);
    }
}
