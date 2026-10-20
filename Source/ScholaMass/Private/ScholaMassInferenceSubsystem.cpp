// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "ScholaMassInferenceSubsystem.h"
#include "Engine/World.h"
#include "Misc/ScopeLock.h"
void UScholaMassInferenceSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    Registry = Collection.InitializeDependency<UNPCBackgroundInferenceSubsystem>();
    OverrideSubsystemTraits<UScholaMassInferenceSubsystem>(Collection);
}
int32 UScholaMassInferenceSubsystem::AcquireBucket(UNPCBackgroundBackendProvider& Provider,
    const FInteractionDefinition& Definition, const FNPCBackgroundSchedulerSettings& Settings, FString& Error)
{
    check(IsInGameThread());
    // A Mass adapter requires real row semantics, not the point-index prototype transport.
    if (Provider.IsA<UNPCPassthroughBackendProvider>()) { Error = TEXT("Mass requires a model row provider"); return INDEX_NONE; }
    FScopeLock Lock(&Mutex);
    const auto H = Registry->AcquireBucket(Provider, Definition, Settings, Error);
    if (!H.IsSet()) return INDEX_NONE;
    if (Handles.Contains(H.Id)) { Registry->ReleaseBucket(H); return H.Id; }
    Handles.Add(H.Id, H);
    Bridge.AddBucket(H.Id, H.Generation, Registry->ShareScheduler(H).ToSharedRef(), Registry->ShareCohorts(H).ToSharedRef());
    return H.Id;
}
void UScholaMassInferenceSubsystem::Deinitialize()
{
    FScopeLock Lock(&Mutex);
    Bridge.Shutdown();
    if (Registry) for (const auto& Pair : Handles) Registry->ReleaseBucket(Pair.Value);
    Handles.Empty(); Registry = nullptr;
    Super::Deinitialize();
}
