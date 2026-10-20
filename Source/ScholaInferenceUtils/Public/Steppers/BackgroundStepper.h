// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "Agent/AgentInterface.h"
#include "Policies/NPCInferencePolicy.h"
#include "BackgroundStepper.generated.h"

UINTERFACE(meta=(CannotImplementInterfaceInBlueprint))
class SCHOLAINFERENCEUTILS_API UNPCBackgroundRowAgent : public UInterface { GENERATED_BODY() };
class SCHOLAINFERENCEUTILS_API INPCBackgroundRowAgent
{
    GENERATED_BODY()
public:
    virtual void WriteObservationRow(TArrayView<float> Row) = 0;
    virtual void ApplyActionRow(TConstArrayView<float> Row) = 0;
};

USTRUCT()
struct FNPCBackgroundAgentSlot
{
    GENERATED_BODY()
    UPROPERTY() TScriptInterface<IAgent> Agent;
    UPROPERTY() TInstancedStruct<FPoint> Observation;
    UPROPERTY() TInstancedStruct<FPoint> Action;
    TArray<float> HeldRow;
    float Period = 1.f;
    double NextDue = 0;
    uint32 Serial = 1, SubmittedSerial = 0, AppliedSerial = 0;
    int32 Phase = 0;
    bool bHasAction = false, bAppliedThisStep = false, bUrgent = false, bRemoving = false;
};
USTRUCT()
struct FNPCBackgroundStepperRequest
{
    GENERATED_BODY()
    FNPCBackgroundRequest Handle;
    TArray<int32> Slots;
    TArray<uint32> SlotSerials;
    UPROPERTY() TArray<TInstancedStruct<FPoint>> PassthroughPoints;
    uint32 Serial = 0;
    double ObsTime = 0;
};

/** Game-thread-only deferred Actor delivery. Step never waits for inference. */
UCLASS(BlueprintType)
class SCHOLAINFERENCEUTILS_API UBackgroundStepper : public UObject
{
    GENERATED_BODY()
public:
    UFUNCTION(BlueprintCallable, Category="Schola|Stepper")
    bool Init(UNPCInferencePolicy* InPolicy, const TArray<TScriptInterface<IAgent>>& Agents, const TArray<float>& DecisionHz);
    /** Add/init run outside Step; callback removals/reset defer storage cleanup until Step exits. */
    UFUNCTION(BlueprintCallable, Category="Schola|Stepper") int32 AddAgent(TScriptInterface<IAgent> Agent, float DecisionHz);
    UFUNCTION(BlueprintCallable, Category="Schola|Stepper") void RemoveAgent(int32 Slot);
    UFUNCTION(BlueprintCallable, Category="Schola|Stepper") void Step(double Now);
    UFUNCTION(BlueprintCallable, Category="Schola|Stepper") void RequestDecisionNow(int32 Slot);
    UFUNCTION(BlueprintCallable, Category="Schola|Stepper") void Reset();
    /** Game-thread diagnostic: current scheduled deadline, including Busy deferral. Zero for an unused slot. */
    double GetNextDecisionTime(int32 Slot) const
    {
        check(IsInGameThread());
        return Slots.IsValidIndex(Slot) && Slots[Slot].Agent.GetObject() && !Slots[Slot].bRemoving ? Slots[Slot].NextDue : 0.0;
    }
    virtual void BeginDestroy() override;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schola|Stepper") bool bAutoTick = false;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schola|Stepper") float MaxActionAgeMs = 100.f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schola|Stepper") bool bReapplyHeldAction = false;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schola|Stepper") ENPCPhaseLayout PhaseLayout = ENPCPhaseLayout::ModelAware;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schola|Stepper", meta=(ClampMin="1")) int32 MaxRowsPerFrame = 4096;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Schola|Stepper") int64 DeferredRows = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Schola|Stepper") int64 SubmittedRows = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Schola|Stepper") int64 AppliedRows = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Schola|Stepper") int64 DroppedRows = 0;
private:
    void ApplyHeld(int32 Slot);
    UPROPERTY() TObjectPtr<UNPCInferencePolicy> Policy;
    UPROPERTY() TArray<FNPCBackgroundAgentSlot> Slots;
    UPROPERTY() TArray<FNPCBackgroundStepperRequest> Pending;
    UPROPERTY() TArray<FNPCBackgroundStepperRequest> FreeRequests;
    TArray<int64> Keys;
    TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Scheduler;
    TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe> Codec;
    TSharedPtr<FNPCCohortScheduler> Cohorts;
    TArray<int32> Due;
    FNPCBackgroundResult Result;
    TArray<int32> DeferredRemovals;
    uint32 Generation = 0;
    double CurrentTime = 0;
    bool bStarted = false, bStepping = false, bResetRequested = false;
};
