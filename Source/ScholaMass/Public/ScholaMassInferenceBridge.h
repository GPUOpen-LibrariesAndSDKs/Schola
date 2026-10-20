// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "ScholaInferenceFragments.h"
#include "Misc/Optional.h"
#include "Inference/BackgroundInferenceScheduler.h"
#include "Inference/CohortScheduler.h"
#include "Mass/EntityHandle.h"

struct FMassEntityManager;

/** Standalone-manager-testable, world-local transport and dense held-action table.
 * Call from ordered Mass processors; never accesses Actors or UObjects.
 */
class SCHOLAMASS_API FScholaMassInferenceBridge
{
public:
    ~FScholaMassInferenceBridge();
    void AddBucket(int32 Id, uint32 Generation, TSharedRef<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Scheduler,
                   TSharedRef<FNPCCohortScheduler> Cohorts);
    bool InitializeSlot(FMassEntityHandle Entity, int32 BucketId, FScholaInferenceScheduleFragment& Schedule,
                        FScholaInferenceStateFragment& State, double Now, TOptional<ENPCPhaseLayout> Layout = {});
    void ReleaseSlot(FScholaInferenceStateFragment& State);
    ENPCBackgroundStatus Submit(int32 BucketId, FMassEntityManager& Manager, TConstArrayView<FMassEntityHandle> Entities,
        TFunctionRef<void(TArrayView<float>)> Fill, double Now, float MaxAgeMs);
    void Drain(FMassEntityManager& Manager, double Now, float MaxAgeMs, TArray<FMassEntityHandle>& Ready);
    /** Ready entities remain available until the matching apply processor acknowledges them. */
    /** True only after a matching apply processor acknowledged the newest row. */
    bool IsActionDelivered(const FScholaInferenceStateFragment& State) const;
    void MarkDelivered(const FScholaInferenceStateFragment& State);
    void DiscardReady(const FScholaInferenceStateFragment& State);
    TConstArrayView<float> GetHeldAction(const FScholaInferenceStateFragment& State) const;
    TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> GetScheduler(int32 Id) const;
    double GetHeldObservationTime(const FScholaInferenceStateFragment& State) const;
    void Shutdown();
    int64 DeferredRows = 0, DroppedRows = 0, AppliedRows = 0;
private:
    struct FSlot
    {
        FMassEntityHandle Entity;
        uint32 Serial = 1, DeliveredSerial = 0;
        int32 BucketId = INDEX_NONE, Phase = 0, ReadyIndex = INDEX_NONE;
        float Period = 1;
        ENPCPhaseLayout Layout = ENPCPhaseLayout::ModelAware;
        TArray<float> Action;
        double ActionObsTime = 0;
    };
    struct FRequest
    {
        FNPCBackgroundRequest Handle;
        TArray<FMassEntityHandle> Entities;
        TArray<uint32> SlotSerials;
        uint32 Serial = 0;
        double ObsTime = 0;
    };
    struct FBucket
    {
        TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Scheduler;
        TSharedPtr<FNPCCohortScheduler> Cohorts;
        uint32 Generation = 0;
        TArray<FRequest> Pending;
    };
    TMap<int32, FBucket> Buckets;
    TArray<FSlot> Slots;
    TArray<int32> FreeSlots, ReadySlots;
    void ClearReady(FSlot& Slot);
    FNPCBackgroundResult Result;
};
