// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "GameFramework/Actor.h"
#if __has_include("Mass/EntityHandle.h")
#include "Mass/EntityHandle.h"
#else
#include "MassEntityHandle.h" // UE 5.7 and earlier
#endif
#include "MassEntityTypes.h"
#include "Inference/BackgroundInferenceTypes.h"
#include "CrowdTypes.h"
#include "CampusMigrationDay.h"
#include "CrowdComparisonHost.generated.h"

class ACampusScene;
class ACameraActor;
class UInstancedStaticMeshComponent;
class UNNEModelData;
class UCrowdGatherProcessor;
class UCrowdApplyProcessor;
class UCrowdSyncInferenceProcessor;
class UCrowdMoveProcessor;
class FNPCBackgroundScheduler;
class UScholaMassInferenceSubsystem;

/**
 * Rows the background path did not turn into decisions, by cause. Cumulative;
 * subtract two snapshots for a window.
 */
struct FCrowdLosses
{
    /** Due rows held back by the per-frame cap or a rejected (queue full) submission; retried next frame. */
    int64 Deferred = 0;
    /** Submissions the scheduler rejected because its pending-row or byte limit was full. */
    int64 Rejected = 0;
    /** Submissions the scheduler can never accept (e.g. more rows than its pending-row limit). */
    int64 Invalid = 0;
    /** Rows that expired or failed inside the scheduler. */
    int64 Expired = 0, Failed = 0;
    /** Results discarded on apply as stale (older than the action age limit) or superseded. */
    int64 Stale = 0;

    FCrowdLosses operator-(const FCrowdLosses& O) const
    { return {Deferred - O.Deferred, Rejected - O.Rejected, Invalid - O.Invalid, Expired - O.Expired, Failed - O.Failed, Stale - O.Stale}; }
    int64 Lost() const { return Expired + Failed + Stale; }
};

/** Values shown on the HUD, refreshed once per second. */
struct FCrowdDisplayStats
{
    double FramesPerSecond = 0;
    double FrameMs = 0;
    double FrameP99Ms = 0;
    double InferenceMs = 0;
    double DecisionsPerSecond = 0;
    double LatencyMs = 0;
    double RowsPerModelCall = 0;
    /** Total model time per second: on the game thread (sync) or summed across workers (background). */
    double ModelMsPerSecond = 0;
    int64 DroppedRows = 0;
    /** Delivered decisions as a fraction of the target rate. */
    double TargetFraction = 1;
    /** Movement/separation update on the game thread (parallel across chunks). */
    double MoveMs = 0;
    /** Unserved rows per second in the last window. */
    FCrowdLosses LossesPerSecond;
    /** Empty when the target rate is met; otherwise the most likely limiting factor. */
    FString Shortfall;
};

/**
 * Spawns a Mass crowd driven by one ONNX steering policy and runs it either
 * with Schola's background NPC API or with per-agent synchronous inference.
 * Everything except the inference path (scene, agents, observations, movement,
 * decision rate) is identical between the two modes.
 *
 * Command line: -Mode=Background|Sync -Agents=20000 -DecisionHz=10 -MaxFPS=F
 * -Workers=N -MaxBatch=N; capture with -CaptureSeconds=S -WarmupSeconds=S -Report=path.json
 */
UCLASS()
class ACrowdComparisonHost : public AActor
{
    GENERATED_BODY()
public:
    ACrowdComparisonHost();
    virtual void Tick(float DeltaSeconds) override;

    void SetMode(ECrowdInferenceMode NewMode);
    ECrowdInferenceMode GetMode() const { return Mode; }
    bool IsNavigationMapView() const { return bNavigationMapView; }
    bool ShowsRoutes() const { return bShowRoutes; }
    bool IsSplitDemo() const { return bSplitDemo && Error.IsEmpty(); }
    bool IsFourDemo() const { return bFourDemo; }
    const ACampusScene* GetCampus() const { return Campus; }
    const TBitArray<>& GetActiveDestinations() const { return ActiveDestinations; }
    int32 GetAgentCount() const { return Entities.Num(); }
    double GetTargetDecisionsPerSecond() const { return Entities.Num() / PeriodSeconds; }
    const FCampusMigrationDay* GetMigrationDay() const { return MigrationDay.Get(); }
    const FCrowdDisplayStats& GetDisplayStats() const { return Display; }
    const FString& GetError() const { return Error; }
    double GetCaptureProgress(FString& OutPhase) const;

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(EEndPlayReason::Type Reason) override;

private:
    UPROPERTY() TObjectPtr<ACampusScene> Campus;
    UPROPERTY() TObjectPtr<ACameraActor> CampusCamera;
    int32 CameraPreset = 0;
    int32 NavigationGateView = INDEX_NONE;
    int32 PreviousCameraPreset = -1;
    double CameraPanStarted = 0;
    /** Length of the preset-4 camera flight; -FlightSeconds=S. */
    double FlightSeconds = 10;
    bool bNavigationMapView = false;
    bool bShowRoutes = false;
    bool bScreenshotTaken = false;
    bool bPaused = false;
    bool bSplitDemo = false, bFourDemo = false, bSwitchingDemo = false;
    TBitArray<> ActiveDestinations;
    TArray<int32> StudentColorKeys;
    void UpdateDestinationColors();
    void UpdateCamera(float DeltaSeconds);
    void KeepCrowdVisible();
    bool LoadPolicy();
    void BuildScene();
    void UpdateMigrationSchedules(bool bScheduleWalking);
    void TickMigration(float DeltaSeconds, double Now, double FrameMs);
    void CaptureMigrationPerformance(double Now, double FrameMs, double InferenceMs, double MovementMs);
    void RunInference(float DeltaSeconds);
    void UpdateDisplay(double Now, double FrameMs, double InferenceMs, double MoveMs);
    void WriteReport();
    void Fail(const FString& Message);

    UPROPERTY(VisibleAnywhere) TObjectPtr<UInstancedStaticMeshComponent> Crowd;
    UPROPERTY() TObjectPtr<UNNEModelData> ModelData;
    UPROPERTY() TObjectPtr<UCrowdGatherProcessor> Gather;
    UPROPERTY() TObjectPtr<UCrowdApplyProcessor> Apply;
    UPROPERTY() TObjectPtr<UCrowdSyncInferenceProcessor> Sync;
    UPROPERTY() TObjectPtr<UCrowdMoveProcessor> Move;
    UPROPERTY() TObjectPtr<UScholaMassInferenceSubsystem> Service;

    FCrowdLosses ReadLosses() const;
    FString DiagnoseShortfall(const FCrowdLosses& Window, double FrameMs) const;
    int32 MaxRowsPerFrame = 0, MaxPendingRows = 0, WorkerCount = 0, MaxBatchSize = 0;
    /** Shortfall is judged only after this many full windows in a mode (pipeline fill). */
    int32 WindowsInMode = 0;
    bool bShortfallLogged = false;

    TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Scheduler;
    TArray<FMassEntityHandle> Entities;
    TArray<FTransform> Transforms;
    TUniquePtr<FCampusMigrationDay> MigrationDay;
    float MigrationTimeScale = 1, MigrationFixedStep = 0;
    TArray<int32> ScheduledMigrationGenerations, ScheduledTurnStudents;
    bool bExitAfterMigration = false;
    ECrowdInferenceMode Mode = ECrowdInferenceMode::Background;
    FCrowdFrameCounters Counters;
    FCrowdDisplayStats Display;
    FString Error;
    float HalfSize = 0;
    double PeriodSeconds = .5;

    // Rolling one-second display window.
    double WindowStart = 0, PreviousTick = 0;
    int32 WindowFrames = 0;
    double WindowMigrationDemand = 0;
    double WindowFrameMs = 0, WindowInferenceMs = 0, WindowMoveMs = 0;
    FCrowdLosses WindowLossStart, ModeLossStart;
    FCrowdFrameCounters WindowCounters;
    TArray<double> RecentFrameMs;
    FNPCBackgroundStats WindowSchedulerStart, ModeSchedulerStart;

    // Optional measured capture for the comparison report.
    FString ReportPath;
    double WarmupSeconds = 10, CaptureSeconds = 0, Started = 0, CaptureStart = 0;
    bool bCapturing = false, bReported = false;
    TArray<double> CaptureFrameMs, CaptureInferenceMs, CaptureMoveMs;
    FCrowdLosses CaptureLossStart;
    FCrowdFrameCounters CaptureCounters;
    FCampusPlanningStats CapturePlanningStart;
    int64 MigrationDecisionStart = 0, CaptureSyncCallsStart = 0, WindowTurnRequests = 0;
    FCampusTurnStats CaptureTurnStart;
    double MigrationWalkingSeconds = 0, MigrationExpectedDecisions = 0;
    FNPCBackgroundStats CaptureSchedulerStart;
};
