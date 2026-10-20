// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "Subsystems/WorldSubsystem.h"
#include "Inference/NPCBackgroundBackendProvider.h"
#include "Inference/BackgroundInferenceScheduler.h"
#include "Inference/CohortScheduler.h"
#include "NPCBackgroundInferenceSubsystem.generated.h"

class UBackgroundStepper;

struct FNPCBackgroundBucketHandle
{
    int32 Id = INDEX_NONE;
    uint32 Generation = 0;
    bool IsSet() const { return Id != INDEX_NONE; }
};

/** Per-world registry. Registry mutations run on the game thread; schedulers are thread-safe. */
UCLASS()
class SCHOLAINFERENCEUTILS_API UNPCBackgroundInferenceSubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()
public:
    FNPCBackgroundBucketHandle AcquireBucket(UNPCBackgroundBackendProvider& Provider,
        const FInteractionDefinition& Definition, const FNPCBackgroundSchedulerSettings& Settings, FString& OutError);
    void ReleaseBucket(FNPCBackgroundBucketHandle Handle);
    FNPCBackgroundScheduler* GetScheduler(FNPCBackgroundBucketHandle Handle) const;
    TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> ShareScheduler(FNPCBackgroundBucketHandle Handle) const;
    const FNPCRowCodec* GetCodec(FNPCBackgroundBucketHandle Handle) const;
    TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe> ShareCodec(FNPCBackgroundBucketHandle Handle) const;
    TSharedPtr<FNPCCohortScheduler> ShareCohorts(FNPCBackgroundBucketHandle Handle) const;
    uint32 GetGeneration(FNPCBackgroundBucketHandle Handle) const;
    UFUNCTION(BlueprintPure, Category="Schola|Inference") TArray<FNPCBackgroundStats> GetAllBucketStats() const;
    virtual void Deinitialize() override;
    void RegisterStepper(UBackgroundStepper* Stepper);
    void UnregisterStepper(UBackgroundStepper* Stepper);
    virtual void Tick(float DeltaTime) override;
    virtual bool IsTickable() const override { return !RegisteredSteppers.IsEmpty(); }
    virtual TStatId GetStatId() const override;
private:
    struct FBucket
    {
        TSharedPtr<FNPCCohortScheduler> Cohorts;
        FString Key;
        uint32 Generation = 0;
        int32 References = 0;
        TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Scheduler;
        TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe> Codec;
    };
    const FBucket* Find(FNPCBackgroundBucketHandle Handle) const;
    TArray<TWeakObjectPtr<UBackgroundStepper>> RegisteredSteppers;
    TMap<int32, FBucket> Buckets;
    int32 NextId = 0;
    uint32 NextGeneration = 1;
};
