// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// Local playable scaffold for native online-learning experiments.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "GameFramework/HUD.h"
#include "GameFramework/Pawn.h"
#include "TagLearner.h"
#include "TagArena.generated.h"

class USphereComponent;
class UStaticMeshComponent;

UCLASS()
class CSC301HOST_API ATagArenaPawn : public APawn
{
    GENERATED_BODY()
public:
    ATagArenaPawn();
    void SetRunnerAppearance(bool bRunner);
    void MovePlanar(const FVector2D& Direction, float Speed, float DeltaSeconds);
private:
    UPROPERTY() TObjectPtr<USphereComponent> Body;
    UPROPERTY() TObjectPtr<UStaticMeshComponent> Visual;
    UPROPERTY() TObjectPtr<UStaticMeshComponent> Ring;
};

enum class ETagRunnerMode : uint8 { Human, LeftBias, RightBias, Balanced };
enum class ETagRoundPhase : uint8 { Countdown, Active, Finished };

/** Measured outcome only. A history entry is not an optimizer update or learning claim. */
struct FTagArenaOutcome
{
    int32 Round = 0;
    int32 RunnerSide = 0;
    int32 TaggerSide = 0;
    float Duration = 0;
    FString Result;
    ETagRunnerMode Mode = ETagRunnerMode::Human;
};

DECLARE_MULTICAST_DELEGATE_OneParam(FTagArenaRoundFinished, const FTagArenaOutcome&);

/** Native categorical actor directly selects the tagger route and learns from rewards. */
UCLASS()
class CSC301HOST_API ATagArenaGameMode : public AGameModeBase
{
    GENERATED_BODY()
public:
    ATagArenaGameMode();
    void StartPlay() override;
    void Tick(float DeltaSeconds) override;
    void EndPlay(const EEndPlayReason::Type Reason) override;

    ETagRunnerMode RunnerMode = ETagRunnerMode::Human;
    ETagRoundPhase Phase = ETagRoundPhase::Countdown;
    int32 RoundNumber = 0;
    int32 Catches = 0;
    int32 Escapes = 0;
    int32 Timeouts = 0;
    int32 LeftRoutes = 0;
    int32 RightRoutes = 0;
    int32 RunnerSide = 0;
    int32 TaggerSide = 0;
    float RoundSeconds = 0;
    float PhaseSeconds = 0;
    bool bPaused = false;
    bool bTaggerRevealed = false;
    FString ResultText;
    FTagArenaOutcome LastOutcome;
    FTagArenaRoundFinished OnRoundFinished;
    tag_learning::Decision CurrentDecision;
    tag_learning::Stats LearningStats;
    TArray<FString> LiveEvents;
    TArray<float> ProbabilityHistory;
    FString LastExperienceText = TEXT("No scored experience yet.");
    FString LearningStatus = TEXT("Starting native learner...");
    bool bUseLearner = true;
    bool bFreezeLearning = false;
    float SimulationSpeed = 1;
    float DisplayFrameMs = 0;
    int32 SubmittedExperiences = 0;
    int32 SkippedExperiences = 0;
    float GetRecentCatchRate() const;
    int32 GetRecentCount() const { return RecentRewards.Num(); }

    ATagArenaPawn* GetRunner() const { return Runner; }
    ATagArenaPawn* GetTagger() const { return Tagger; }
    FString GetModeLabel() const;
    static FString SideLabel(int32 Side);
    int32 CompletedRounds() const { return Catches + Escapes + Timeouts; }

private:
    UPROPERTY() TObjectPtr<ATagArenaPawn> Runner;
    UPROPERTY() TObjectPtr<ATagArenaPawn> Tagger;
    FRandomStream TaggerRandom{2026};
    FRandomStream RunnerRandom{90210};
    int32 ScriptedSide = 0;
    bool bSmokeTest = false;
    FString ResultsPath;
    FString LearningPath;
    TUniquePtr<tag_learning::Learner> NativeLearner;
    TArray<float> RecentRewards;
    bool bRoundUsesLearner = false;
    double NextHistoryTime = 0;
    double NextTelemetryTime = 0;
    bool bLearningTest = false;
    int32 LearningTestPhase = 0;
    int32 LearningTestRounds = 0;
    int32 LearningTestChecks = 0;
    bool bLearningTestPassed = true;
    bool bTestActorOwnsAllActions = true;
    float LearningTestFirstProbability = 0.5f;
    double LearningTestStarted = 0;

    void StartRound();
    void AdvanceArena(float DeltaSeconds);
    void FinishRound(const FString& Result);
    void ChangeMode(ETagRunnerMode Mode);
    void ResetSession(bool bResetLearning = true);
    void RunSmokeTest();
    void AddEvent(const FString& Message);
    void UpdateLearningTelemetry();
    void AdvanceLearningTest();
};

UCLASS()
class CSC301HOST_API ATagArenaHUD : public AHUD
{
    GENERATED_BODY()
public:
    void DrawHUD() override;
};
