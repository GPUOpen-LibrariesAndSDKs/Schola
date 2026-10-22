// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "TagArena.h"

#include "Camera/CameraActor.h"
#include "Camera/CameraComponent.h"
#include "Components/SphereComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/PlayerInput.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformTime.h"
#include "InputCoreTypes.h"
#include "InputKeyEventArgs.h"
#include "Materials/MaterialInterface.h"
#include "Misc/CommandLine.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "TimerManager.h"
#include "UObject/ConstructorHelpers.h"

namespace
{
constexpr float RunnerSpeed = 340.0f;
constexpr float TaggerSpeed = 500.0f;
constexpr float RoundLimit = 12.0f;
const FVector RunnerStart(-500, 0, 45);
const FVector TaggerStart(500, 0, 45);

FVector2D DirectionTo(const FVector& From, const FVector& To, float Speed, float Dt)
{
    const FVector2D Delta(To.X - From.X, To.Y - From.Y);
    // Avoid overshooting the waypoint, including at low frame rates.
    return Delta.GetClampedToMaxSize(FMath::Max(0.001f, Speed * Dt)) / FMath::Max(0.001f, Speed * Dt);
}
}

ATagArenaPawn::ATagArenaPawn()
{
    Body = CreateDefaultSubobject<USphereComponent>(TEXT("Body"));
    RootComponent = Body;
    Body->InitSphereRadius(36);
    Body->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
    Body->SetCollisionObjectType(ECC_Pawn);
    Body->SetCollisionResponseToAllChannels(ECR_Ignore);
    Body->SetCollisionResponseToChannel(ECC_WorldStatic, ECR_Block);
    Body->SetCollisionResponseToChannel(ECC_WorldDynamic, ECR_Block);
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Sphere(TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Cylinder(TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
    Visual = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Visual"));
    Visual->SetupAttachment(Body);
    Visual->SetStaticMesh(Sphere.Object);
    Visual->SetRelativeScale3D(FVector(0.65f));
    Visual->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    Ring = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("GroundRing"));
    Ring->SetupAttachment(Body);
    Ring->SetStaticMesh(Cylinder.Object);
    Ring->SetRelativeLocation(FVector(0, 0, -38));
    Ring->SetRelativeScale3D(FVector(0.94f, 0.94f, 0.035f));
    Ring->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    AutoPossessAI = EAutoPossessAI::Disabled;
}

void ATagArenaPawn::SetRunnerAppearance(bool bRunner)
{
    const TCHAR* Path = bRunner ? TEXT("/Game/TagArena/Materials/M_Runner.M_Runner") : TEXT("/Game/TagArena/Materials/M_Tagger.M_Tagger");
    if (auto* Material = LoadObject<UMaterialInterface>(nullptr, Path))
    {
        Visual->SetMaterial(0, Material);
        Ring->SetMaterial(0, Material);
    }
}

void ATagArenaPawn::MovePlanar(const FVector2D& Direction, float Speed, float DeltaSeconds)
{
    const FVector2D Input = Direction.GetClampedToMaxSize(1.0f);
    const FVector Delta(Input.X * Speed * DeltaSeconds, Input.Y * Speed * DeltaSeconds, 0);
    FHitResult Hit;
    AddActorWorldOffset(Delta, true, &Hit);
    if (Hit.bBlockingHit)
    {
        const FVector Slide = FVector::VectorPlaneProject(Delta * (1.0f - Hit.Time), Hit.Normal);
        AddActorWorldOffset(Slide, true);
    }
}

ATagArenaGameMode::ATagArenaGameMode()
{
    PrimaryActorTick.bCanEverTick = true;
    DefaultPawnClass = nullptr;
    HUDClass = ATagArenaHUD::StaticClass();
}

void ATagArenaGameMode::StartPlay()
{
    Super::StartPlay();
    FActorSpawnParameters Spawn;
    Spawn.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    Runner = GetWorld()->SpawnActor<ATagArenaPawn>(RunnerStart, FRotator::ZeroRotator, Spawn);
    Tagger = GetWorld()->SpawnActor<ATagArenaPawn>(TaggerStart, FRotator::ZeroRotator, Spawn);
    check(Runner && Tagger);
    Runner->SetRunnerAppearance(true);
    Tagger->SetRunnerAppearance(false);
    if (auto* Player = GetWorld()->GetFirstPlayerController())
    {
        Player->Possess(Runner);
        Player->bAutoManageActiveCameraTarget = false;
        Player->SetInputMode(FInputModeGameOnly());
        Player->bShowMouseCursor = false;
        for (TActorIterator<ACameraActor> Camera(GetWorld()); Camera; ++Camera)
        {
            if (Camera->ActorHasTag(TEXT("TagArenaCamera")))
            {
                Camera->GetCameraComponent()->SetOrthoWidth(3200);
                Camera->GetCameraComponent()->PostProcessSettings.bOverride_MotionBlurAmount = true;
                Camera->GetCameraComponent()->PostProcessSettings.MotionBlurAmount = 0;
                Player->SetViewTarget(*Camera);
                break;
            }
        }
    }
    ResultsPath = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("TagArena"), TEXT("Rounds.csv"));
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(ResultsPath), true);
    if (!FPaths::FileExists(ResultsPath))
    {
        FFileHelper::SaveStringToFile(TEXT("round,mode,runner_side,tagger_side,result,seconds\n"), *ResultsPath);
    }
    bSmokeTest = FParse::Param(FCommandLine::Get(), TEXT("TagArenaSmokeTest"));
    bLearningTest = FParse::Param(FCommandLine::Get(), TEXT("TagLearningTest"));
    LearningPath = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("TagLearning"), TEXT("Telemetry.csv"));
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(LearningPath), true);
    FFileHelper::SaveStringToFile(TEXT("wall_seconds,round,mode,phase,speed,experiences,updates,policy_version,p_left,critic_mse,batch_ms,queue,reward,inference_ms\n"), *LearningPath);
    if (!bSmokeTest) FFileHelper::SaveStringToFile(TEXT("round,observation,action,p_left,chosen_probability,tagger_route,runner_route,reward,next_observation,terminal,policy_version,submitted\n"), *(FPaths::ProjectSavedDir()/TEXT("TagLearning/Experiences.csv")));
    if (!bSmokeTest)
    {
        int32 Seed = 2026;
        FParse::Value(FCommandLine::Get(), TEXT("TagLearningSeed="), Seed);
        NativeLearner = MakeUnique<tag_learning::Learner>(Seed);
    }
    if (bLearningTest)
    {
        RunnerMode = ETagRunnerMode::LeftBias;
        LearningTestStarted = FPlatformTime::Seconds();
    }
    StartRound();
    AddEvent(TEXT("Categorical actor started. LEFT/RIGHT comes directly from the actor, from round one."));
    if (FParse::Param(FCommandLine::Get(), TEXT("TagStartPaused")))
    {
        bPaused = true;
        AddEvent(TEXT("Paused for manual play. Press Space, then use WASD to run."));
    }
    if (bSmokeTest)
    {
        GetWorldTimerManager().SetTimerForNextTick(this, &ATagArenaGameMode::RunSmokeTest);
    }
}

void ATagArenaGameMode::EndPlay(const EEndPlayReason::Type Reason)
{
    if (NativeLearner) { NativeLearner->Stop(); NativeLearner.Reset(); }
    Super::EndPlay(Reason);
}

void ATagArenaGameMode::AddEvent(const FString& Message)
{
    LiveEvents.Insert(Message, 0);
    if (LiveEvents.Num() > 7) { LiveEvents.SetNum(7); }
    UE_LOG(LogTemp, Display, TEXT("TagLearning: %s"), *Message);
}

float ATagArenaGameMode::GetRecentCatchRate() const
{
    if (RecentRewards.IsEmpty()) return 0;
    int32 Count = 0;
    for (float Reward : RecentRewards) Count += Reward > 0 ? 1 : 0;
    return float(Count) / RecentRewards.Num();
}

void ATagArenaGameMode::UpdateLearningTelemetry()
{
    if (!NativeLearner) return;
    LearningStats = NativeLearner->ReadStats();
    LearningStatus = !bUseLearner ? TEXT("FIXED 50/50 / model preserved") : bFreezeLearning ? TEXT("FROZEN / using learned weights") :
        !LearningStats.healthy ? TEXT("ERROR / invalid training value") : TEXT("LIVE / categorical actor-critic");
    const double Now = FPlatformTime::Seconds();
    if (Now >= NextHistoryTime)
    {
        ProbabilityHistory.Add(LearningStats.expectedLeft);
        if (ProbabilityHistory.Num() > 120) ProbabilityHistory.RemoveAt(0);
        NextHistoryTime = Now + 0.25;
    }
    if (!bLearningTest && Now >= NextTelemetryTime)
    {
        const FString Row = FString::Printf(TEXT("%.3f,%d,%s,%d,%.0f,%u,%u,%u,%.5f,%.5f,%.5f,%u,%.1f,%.5f\n"),
            GetWorld()->GetRealTimeSeconds(), RoundNumber, *GetModeLabel(), int32(Phase), SimulationSpeed,
            LearningStats.experiences, LearningStats.updates, LearningStats.appliedVersion, LearningStats.expectedLeft,
            LearningStats.criticMse, LearningStats.batchMs, LearningStats.queued, LearningStats.lastReward, LearningStats.inferenceMs);
        FFileHelper::SaveStringToFile(Row, *LearningPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append);
        NextTelemetryTime = Now + 0.5;
    }
}

FString ATagArenaGameMode::SideLabel(int32 Side)
{
    return Side < 0 ? TEXT("LEFT") : Side > 0 ? TEXT("RIGHT") : TEXT("--");
}

FString ATagArenaGameMode::GetModeLabel() const
{
    switch (RunnerMode)
    {
    case ETagRunnerMode::LeftBias: return TEXT("AUTO / 80% LEFT");
    case ETagRunnerMode::RightBias: return TEXT("AUTO / 80% RIGHT");
    case ETagRunnerMode::Balanced: return TEXT("AUTO / 50 : 50");
    default: return TEXT("YOU / WASD");
    }
}

void ATagArenaGameMode::StartRound()
{
    ++RoundNumber;
    Phase = ETagRoundPhase::Countdown;
    PhaseSeconds = RoundSeconds = 0;
    RunnerSide = 0;
    // The learner receives only [1]. Choose and sample BEFORE this round's runner choice.
    bRoundUsesLearner = bUseLearner && NativeLearner.IsValid();
    CurrentDecision = bRoundUsesLearner ? NativeLearner->Choose() : tag_learning::Decision{};
    if (bRoundUsesLearner)
    {
        // No second coin flip, warm-up substitution, epsilon override, or probability remapping.
        TaggerSide = CurrentDecision.route;
        if (!CurrentDecision.valid) { bPaused = true; AddEvent(TEXT("Actor returned an invalid action. Game paused; no fallback action used.")); }
    }
    else
    {
        // Only the explicitly selected comparison baseline bypasses the actor.
        TaggerSide = TaggerRandom.FRand() < 0.5f ? -1 : 1;
        CurrentDecision.route = TaggerSide;
        CurrentDecision.chosenProbability = 0.5f;
        CurrentDecision.valid = true;
    }
    const float LeftProbability = RunnerMode == ETagRunnerMode::LeftBias ? 0.8f : RunnerMode == ETagRunnerMode::RightBias ? 0.2f : 0.5f;
    ScriptedSide = RunnerRandom.FRand() < LeftProbability ? -1 : 1;
    bTaggerRevealed = false;
    ResultText.Reset();
    Runner->SetActorLocation(RunnerStart, false, nullptr, ETeleportType::TeleportPhysics);
    Tagger->SetActorLocation(TaggerStart, false, nullptr, ETeleportType::TeleportPhysics);
    Tagger->SetActorHiddenInGame(false);
}

void ATagArenaGameMode::ResetSession(bool bResetLearning)
{
    Catches = Escapes = Timeouts = LeftRoutes = RightRoutes = 0;
    RoundNumber = 0;
    TaggerRandom.Initialize(2026);
    RunnerRandom.Initialize(90210);
    bPaused = false;
    RecentRewards.Reset();
    if (bResetLearning && NativeLearner)
    {
        NativeLearner->Stop();
        NativeLearner = MakeUnique<tag_learning::Learner>(2026);
        bFreezeLearning = false;
        NativeLearner->SetTrainingEnabled(bUseLearner);
        SubmittedExperiences = SkippedExperiences = 0;
        ProbabilityHistory.Reset();
        LastExperienceText = TEXT("No scored experience yet.");
        AddEvent(TEXT("RESET: fresh actor/critic weights and scores. No saved model loaded."));
    }
    StartRound();
}

void ATagArenaGameMode::ChangeMode(ETagRunnerMode Mode)
{
    RunnerMode = Mode;
    // Keep the learner completely unchanged when runner behaviour changes.
    ResetSession(false);
    AddEvent(TEXT("Runner mode changed. Actor, critic and optimizer preserved."));
}

void ATagArenaGameMode::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    DisplayFrameMs = DisplayFrameMs == 0 ? 1000 * DeltaSeconds : FMath::Lerp(DisplayFrameMs, 1000 * DeltaSeconds, 0.05f);
    if (!Runner || !Tagger || bSmokeTest) { return; }
    UpdateLearningTelemetry();
    if (bLearningTest) { AdvanceLearningTest(); return; }
    if (auto* Player = GetWorld()->GetFirstPlayerController())
    {
        if (Player->WasInputKeyJustPressed(EKeys::One)) { ChangeMode(ETagRunnerMode::Human); }
        if (Player->WasInputKeyJustPressed(EKeys::Two)) { ChangeMode(ETagRunnerMode::LeftBias); }
        if (Player->WasInputKeyJustPressed(EKeys::Three)) { ChangeMode(ETagRunnerMode::RightBias); }
        if (Player->WasInputKeyJustPressed(EKeys::Four)) { ChangeMode(ETagRunnerMode::Balanced); }
        if (Player->WasInputKeyJustPressed(EKeys::R)) { ++SkippedExperiences; AddEvent(TEXT("Round restarted: incomplete experience discarded.")); StartRound(); }
        if (Player->WasInputKeyJustPressed(EKeys::T)) { ResetSession(); }
        if (Player->WasInputKeyJustPressed(EKeys::SpaceBar)) { bPaused = !bPaused; }
        if (Player->WasInputKeyJustPressed(EKeys::L))
        {
            bFreezeLearning = !bFreezeLearning;
            NativeLearner->SetTrainingEnabled(bUseLearner && !bFreezeLearning);
            AddEvent(bFreezeLearning ? TEXT("Training frozen. NPC keeps using its current learned policy.") : TEXT("Training resumed. Weights were preserved."));
        }
        if (Player->WasInputKeyJustPressed(EKeys::B))
        {
            bUseLearner = !bUseLearner;
            NativeLearner->SetTrainingEnabled(bUseLearner && !bFreezeLearning);
            RecentRewards.Reset();
            AddEvent(bUseLearner ? TEXT("Native policy selected. Learned weights preserved.") : TEXT("Fixed 50/50 baseline selected. Training suspended."));
            StartRound();
        }
        if (Player->WasInputKeyJustPressed(EKeys::F))
        {
            SimulationSpeed = SimulationSpeed == 1 ? 4 : SimulationSpeed == 4 ? 12 : 1;
            AddEvent(FString::Printf(TEXT("Simulation speed %.0fx. All experiences still come from actual rounds."), SimulationSpeed));
        }
    }
    // Substeps prevent missing a catch when the renderer stalls; time is never silently discarded.
    float Remaining = DeltaSeconds * SimulationSpeed;
    while (Remaining > 0)
    {
        const float Step = FMath::Min(Remaining, 1.0f / 60.0f);
        AdvanceArena(Step);
        Remaining -= Step;
    }
}

void ATagArenaGameMode::AdvanceArena(float Dt)
{
    if (bLearningTest) bTestActorOwnsAllActions &= bRoundUsesLearner && CurrentDecision.valid && TaggerSide == CurrentDecision.route;
    if (bPaused) { return; }
    PhaseSeconds += Dt;
    if (Phase == ETagRoundPhase::Finished)
    {
        if (PhaseSeconds >= 2.0f) { StartRound(); }
        return;
    }
    if (Phase == ETagRoundPhase::Countdown)
    {
        if (PhaseSeconds >= 1.25f) { Phase = ETagRoundPhase::Active; PhaseSeconds = 0; }
        return;
    }
    RoundSeconds += Dt;
    const FVector Before = Runner->GetActorLocation();
    FVector2D Input = FVector2D::ZeroVector;
    if (RunnerMode == ETagRunnerMode::Human)
    {
        if (!bSmokeTest)
        {
            if (auto* Player = GetWorld()->GetFirstPlayerController())
            {
                Input.X = float(Player->IsInputKeyDown(EKeys::W)) - float(Player->IsInputKeyDown(EKeys::S));
                Input.Y = float(Player->IsInputKeyDown(EKeys::D)) - float(Player->IsInputKeyDown(EKeys::A));
            }
        }
    }
    else
    {
        const FVector Target = Before.X < -335
            ? FVector(-330, ScriptedSide * 500, 45) : FVector(575, ScriptedSide * 500, 45);
        Input = DirectionTo(Before, Target, RunnerSpeed, Dt);
    }
    Runner->MovePlanar(Input, RunnerSpeed, Dt);
    const FVector Position = Runner->GetActorLocation();
    if (RunnerSide != 0 && Position.X < -300)
    {
        // A runner cannot reveal the tagger's commitment, retreat, then switch routes for a scored escape.
        FinishRound(TEXT("Aborted"));
        return;
    }
    if (RunnerSide == 0 && Position.X > -250 && FMath::Abs(Position.Y) > 430)
    {
        RunnerSide = Position.Y < 0 ? -1 : 1;
        bTaggerRevealed = true;
    }
    if (bTaggerRevealed)
    {
        const FVector Intercept(300, TaggerSide * 500, 45);
        Tagger->MovePlanar(DirectionTo(Tagger->GetActorLocation(), Intercept, TaggerSpeed, Dt), TaggerSpeed, Dt);
    }
    if (FVector::DistSquared2D(Position, Tagger->GetActorLocation()) <= FMath::Square(72.0f))
    {
        FinishRound(TEXT("Caught"));
    }
    else if (Position.X >= 540)
    {
        FinishRound(TEXT("Escaped"));
    }
    else if (RoundSeconds >= RoundLimit)
    {
        FinishRound(TEXT("Timeout"));
    }
}

void ATagArenaGameMode::FinishRound(const FString& Result)
{
    Phase = ETagRoundPhase::Finished;
    PhaseSeconds = 0;
    if (Result == TEXT("Caught")) { ++Catches; ResultText = TEXT("TAGGED"); }
    else if (Result == TEXT("Escaped")) { ++Escapes; ResultText = TEXT("YOU ESCAPED"); }
    else if (Result == TEXT("Timeout")) { ++Timeouts; ResultText = TEXT("TIME IS UP"); }
    else { ResultText = TEXT("ROUTE RESTART"); }
    if (Result != TEXT("Aborted"))
    {
        if (RunnerSide < 0) { ++LeftRoutes; }
        if (RunnerSide > 0) { ++RightRoutes; }
    }
    LastOutcome = {RoundNumber, RunnerSide, TaggerSide, RoundSeconds, Result, RunnerMode};
    const bool bScored = Result == TEXT("Caught") || Result == TEXT("Escaped");
    if (bScored)
    {
        const float Reward = Result == TEXT("Caught") ? 1.0f : -1.0f;
        RecentRewards.Add(Reward);
        if (RecentRewards.Num() > 64) RecentRewards.RemoveAt(0);
        bool bSubmitted = false;
        if (bRoundUsesLearner && bUseLearner && !bFreezeLearning)
        {
            bSubmitted = NativeLearner->Submit({CurrentDecision, Reward, TaggerSide, uint32(RoundNumber)});
            if (bSubmitted) ++SubmittedExperiences;
        }
        LastExperienceText = FString::Printf(TEXT("o=[1] a=%s r=%+.0f next=[1] done=1"), *SideLabel(CurrentDecision.route), Reward);
        if (!bSmokeTest)
        {
            const FString ExperienceRow = FString::Printf(TEXT("%d,1,%d,%.6f,%.6f,%d,%d,%.0f,1,1,%u,%d\n"),
                RoundNumber, CurrentDecision.route, CurrentDecision.leftProbability, CurrentDecision.chosenProbability, TaggerSide, RunnerSide, Reward,
                CurrentDecision.version, bSubmitted ? 1 : 0);
            FFileHelper::SaveStringToFile(ExperienceRow, *(FPaths::ProjectSavedDir()/TEXT("TagLearning/Experiences.csv")),
                FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append);
        }
        AddEvent(FString::Printf(TEXT("R%d %s | %s | v%u | %s"), RoundNumber, *SideLabel(TaggerSide), *Result, CurrentDecision.version, bSubmitted ? TEXT("experience queued") : TEXT("training off")));
    }
    else
    {
        ++SkippedExperiences;
        AddEvent(Result == TEXT("Timeout") ? TEXT("Timeout: no training sample (no scored catch/escape).") : TEXT("Aborted: incomplete experience discarded."));
    }
    const FString Row = FString::Printf(TEXT("%d,%s,%d,%d,%s,%.3f\n"), RoundNumber, *GetModeLabel(), RunnerSide, TaggerSide, *Result, RoundSeconds);
    if (!bSmokeTest)
    {
        FFileHelper::SaveStringToFile(Row, *ResultsPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append);
    }
    UE_LOG(LogTemp, Display, TEXT("TagArena: %s"), *Row.TrimEnd());
    OnRoundFinished.Broadcast(LastOutcome);
}

void ATagArenaGameMode::RunSmokeTest()
{
    bool bPassed = true;
    int32 Checks = 0;
    auto Check = [&](bool Condition, const TCHAR* Label)
    {
        ++Checks;
        bPassed &= Condition;
        UE_LOG(LogTemp, Display, TEXT("TagArena smoke [%s]: %s"), Condition ? TEXT("PASS") : TEXT("FAIL"), Label);
    };
    Check(Runner && Tagger, TEXT("Runner and tagger spawned"));
    auto* Player = GetWorld()->GetFirstPlayerController();
    auto* Camera = Player ? Cast<ACameraActor>(Player->GetViewTarget()) : nullptr;
    Check(Camera && Camera->GetActorForwardVector().Z < -0.99f, TEXT("Active camera looks down at the arena"));
    Check(Player && Player->GetPawn() == Runner, TEXT("Local player possesses the runner"));
    // Exercise the same held-key state used by a real player, rather than bypassing input with scripted movement.
    if (Player)
    {
        const FKey Keys[] = {EKeys::W, EKeys::A, EKeys::S, EKeys::D};
        const FVector Directions[] = {FVector(1, 0, 0), FVector(0, -1, 0), FVector(-1, 0, 0), FVector(0, 1, 0)};
        for (int32 Index = 0; Index < 4; ++Index)
        {
            RunnerMode = ETagRunnerMode::Human;
            StartRound();
            Phase = ETagRoundPhase::Active;
            const auto Device = FInputDeviceId::CreateFromInternalId(0);
            Player->InputKey(FInputKeyEventArgs(nullptr, Device, Keys[Index], IE_Pressed, uint64(0)));
            Player->PlayerInput->ProcessInputStack(TArray<UInputComponent*>(), 1.0f / 60.0f, false);
            bSmokeTest = false;
            AdvanceArena(1.0f / 30.0f);
            bSmokeTest = true;
            Player->InputKey(FInputKeyEventArgs(nullptr, Device, Keys[Index], IE_Released, uint64(0)));
            Player->PlayerInput->ProcessInputStack(TArray<UInputComponent*>(), 1.0f / 60.0f, false);
            const FVector Delta = Runner->GetActorLocation() - RunnerStart;
            Check(FVector::DotProduct(Delta, Directions[Index]) > 5.0f, *FString::Printf(TEXT("Held %s moves the runner in the expected direction"), *Keys[Index].ToString()));
        }
    }
    Runner->SetActorLocation(RunnerStart);
    Runner->MovePlanar(FVector2D(1, 0), RunnerSpeed, 2);
    Check(Runner->GetActorLocation().X < -250 && Runner->GetActorLocation().X > -300, TEXT("Central cover blocks movement"));
    Runner->SetActorLocation(RunnerStart);
    Runner->MovePlanar(FVector2D(-1, 0), RunnerSpeed, 10);
    Check(Runner->GetActorLocation().X > -620, TEXT("Arena boundary blocks movement"));
    RunnerMode = ETagRunnerMode::Balanced;
    for (int32 RunnerChoice : {-1, 1})
    {
        for (int32 TaggerChoice : {-1, 1})
        {
            StartRound();
            ScriptedSide = RunnerChoice;
            TaggerSide = TaggerChoice;
            for (int32 Frame = 0; Frame < 1200 && Phase != ETagRoundPhase::Finished; ++Frame)
            {
                AdvanceArena(1.0f / 60.0f);
            }
            Check(Phase == ETagRoundPhase::Finished, TEXT("Scripted round finishes"));
            Check(RunnerSide == RunnerChoice, TEXT("Route telemetry matches traversal"));
            Check(LastOutcome.Result == (RunnerChoice == TaggerChoice ? TEXT("Caught") : TEXT("Escaped")), TEXT("Same-route catches; opposite-route escapes"));
        }
    }
    RunnerMode = ETagRunnerMode::Human;
    StartRound();
    Check(Runner->GetActorLocation().Equals(RunnerStart), TEXT("Round reset restores runner position"));
    Phase = ETagRoundPhase::Active;
    Runner->SetActorLocation(FVector(-150, -500, 45));
    AdvanceArena(1.0f / 60.0f);
    const int32 BeforeAbort = CompletedRounds();
    Runner->SetActorLocation(FVector(-350, -500, 45));
    AdvanceArena(1.0f / 60.0f);
    Check(LastOutcome.Result == TEXT("Aborted") && CompletedRounds() == BeforeAbort, TEXT("Retreat after commitment voids the round without scoring"));
    StartRound();
    const float BeforePause = PhaseSeconds;
    bPaused = true;
    AdvanceArena(1);
    Check(PhaseSeconds == BeforePause, TEXT("Pause freezes the arena clock"));
    bPaused = false;
    for (int32 Frame = 0; Frame < 1000 && Phase != ETagRoundPhase::Finished; ++Frame) { AdvanceArena(1.0f / 60.0f); }
    Check(LastOutcome.Result == TEXT("Timeout") && RunnerSide == 0, TEXT("Idle round times out without inventing a route"));
    ResetSession();
    Check(CompletedRounds() == 0 && RoundNumber == 1, TEXT("Session reset clears scores"));
    const FString Report = FString::Printf(TEXT("{\"passed\":%s,\"checks\":%d}\n"), bPassed ? TEXT("true") : TEXT("false"), Checks);
    const FString Path = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("Automation/TagArenaSmoke.json"));
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(Path), true);
    FFileHelper::SaveStringToFile(Report, *Path);
    FPlatformMisc::RequestExitWithStatus(true, bPassed ? 0 : 1);
}

void ATagArenaGameMode::AdvanceLearningTest()
{
    auto Check = [&](bool Condition, const TCHAR* Label)
    {
        ++LearningTestChecks;
        bLearningTestPassed &= Condition;
        UE_LOG(LogTemp, Display, TEXT("TagLearning integration [%s]: %s"), Condition ? TEXT("PASS") : TEXT("FAIL"), Label);
    };
    if (FPlatformTime::Seconds() - LearningTestStarted > 180 || !LearningStats.healthy)
    {
        Check(false, TEXT("Learner remains healthy and finishes within the test deadline"));
        NativeLearner->Stop();
        FPlatformMisc::RequestExitWithStatus(false, 1);
        return;
    }
    if (Phase == ETagRoundPhase::Finished)
    {
        if (LearningStats.experiences < uint32(SubmittedExperiences) || LearningStats.training || LearningStats.queued) return;
        ++LearningTestRounds;
        if (LearningTestPhase == 0 && LearningTestRounds == 1)
            Check(LearningStats.updates == 1 && CurrentDecision.valid, TEXT("Actor controls and learns from the first round without random warm-up"));
        if (LearningTestRounds == 256)
        {
            const FString PhaseMessage = FString::Printf(TEXT("Phase %d: P(left)=%.3f; recent catch rate=%.3f; experiences=%u"),
                LearningTestPhase, LearningStats.expectedLeft, GetRecentCatchRate(), LearningStats.experiences);
            UE_LOG(LogTemp, Display, TEXT("%s"), *PhaseMessage);
            if (LearningTestPhase == 0)
            {
                LearningTestFirstProbability = LearningStats.expectedLeft;
                Check(LearningStats.expectedLeft > 0.6f, TEXT("Actual arena outcomes train a preference for left against the left-biased runner"));
                const uint32 Before = LearningStats.updates;
                ChangeMode(ETagRunnerMode::RightBias);
                UpdateLearningTelemetry();
                Check(LearningStats.updates == Before && LearningStats.experiences == 256, TEXT("Changing runner bias preserves the learner and its experience"));
                LearningTestPhase = 1;
                LearningTestRounds = 0;
                return;
            }
            Check(LearningStats.expectedLeft < 0.4f, TEXT("Same model adapts to the reversed runner bias without a weight reset"));
            Check(LearningStats.experiences == 512 && LearningStats.updates == 512, TEXT("One direct actor-critic update per scored round, including round one"));
            Check(LearningStats.updates > 0 && LearningStats.dropped == 0 && LearningStats.rejected == 0, TEXT("Real training updates, no dropped or invalid experience"));
            Check(bTestActorOwnsAllActions, TEXT("Every tagger route equals the actor's returned categorical action"));
            Check(!FModuleManager::Get().IsModuleLoaded(TEXT("Schola")), TEXT("The arena and learner run without loading Schola"));
            NativeLearner->SetTrainingEnabled(false);
            const uint32 FrozenUpdates = LearningStats.updates;
            const float FrozenProbability = LearningStats.expectedLeft;
            for (int32 i = 0; i < 20; ++i) NativeLearner->Choose();
            const auto Frozen = NativeLearner->ReadStats();
            Check(Frozen.updates == FrozenUpdates && Frozen.expectedLeft == FrozenProbability, TEXT("Frozen policy keeps its learned weights while inference continues"));
            NativeLearner->Stop();
            const FString Report = FString::Printf(TEXT("{\"passed\":%s,\"checks\":%d,\"experiences\":%u,\"updates\":%u,\"left_phase_p_left\":%.6f,\"right_phase_p_left\":%.6f,\"right_recent_catch_rate\":%.6f,\"schola_api_used\":false,\"actor_only\":%s,\"algorithm\":\"categorical_actor_critic\"}\n"),
                bLearningTestPassed ? TEXT("true") : TEXT("false"), LearningTestChecks, LearningStats.experiences, LearningStats.updates,
                LearningTestFirstProbability, LearningStats.expectedLeft, GetRecentCatchRate(), bTestActorOwnsAllActions ? TEXT("true") : TEXT("false"));
            FString Path = FPaths::ProjectSavedDir() / TEXT("TagLearning/ArenaLearningTest.json");
            FParse::Value(FCommandLine::Get(), TEXT("TagLearningReport="), Path);
            FFileHelper::SaveStringToFile(Report, *Path);
            FPlatformMisc::RequestExitWithStatus(false, bLearningTestPassed ? 0 : 1);
            return;
        }
        StartRound();
    }
    // Fast test mode uses the actual world/pawns/collision with fixed substeps.
    // It never invents samples, uses a reward formula outside FinishRound, or trains on a future route.
    for (int32 Step = 0; Step < 720 && Phase != ETagRoundPhase::Finished; ++Step) AdvanceArena(1.0f / 60.0f);
}
