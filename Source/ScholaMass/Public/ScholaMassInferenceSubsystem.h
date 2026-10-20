// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "MassSubsystemBase.h"
#include "Mass/ExternalSubsystemTraits.h"
#include "Inference/NPCBackgroundInferenceSubsystem.h"
#include "ScholaMassInferenceBridge.h"
#include "ScholaMassInferenceSubsystem.generated.h"

UCLASS()
class SCHOLAMASS_API UScholaMassInferenceSubsystem : public UMassSubsystemBase
{
    GENERATED_BODY()
public:
    int32 AcquireBucket(UNPCBackgroundBackendProvider& Provider, const FInteractionDefinition& Definition,
        const FNPCBackgroundSchedulerSettings& Settings, FString& Error);
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    /** Access Bridge only while holding Mutex. Fill callbacks may run in parallel and must not enter Bridge. */
    FCriticalSection Mutex;
    FScholaMassInferenceBridge Bridge;
private:
    UPROPERTY() TObjectPtr<UNPCBackgroundInferenceSubsystem> Registry;
    TMap<int32, FNPCBackgroundBucketHandle> Handles;
};
template<> struct TMassExternalSubsystemTraits<UScholaMassInferenceSubsystem>
{ enum { GameThreadOnly = false, ThreadSafeWrite = false }; };
