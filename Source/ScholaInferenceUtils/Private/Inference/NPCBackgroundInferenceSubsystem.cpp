// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Inference/NPCBackgroundInferenceSubsystem.h"
#include "Steppers/BackgroundStepper.h"

const UNPCBackgroundInferenceSubsystem::FBucket* UNPCBackgroundInferenceSubsystem::Find(FNPCBackgroundBucketHandle H) const
{
    const auto* B = Buckets.Find(H.Id);
    return B && B->Generation == H.Generation ? B : nullptr;
}

FNPCBackgroundBucketHandle UNPCBackgroundInferenceSubsystem::AcquireBucket(UNPCBackgroundBackendProvider& Provider,
    const FInteractionDefinition& Definition, const FNPCBackgroundSchedulerSettings& Settings, FString& OutError)
{
    check(IsInGameThread());
    // Full schema text includes bounds as well as names/types/shapes. Never share incompatible codecs.
    FString Schema, Config;
    FInteractionDefinition::StaticStruct()->ExportText(Schema, &Definition, nullptr, nullptr, PPF_None, nullptr);
    FNPCBackgroundSchedulerSettings::StaticStruct()->ExportText(Config, &Settings, nullptr, nullptr, PPF_None, nullptr);
    const FString Key = Provider.GetBucketKey() + TEXT("\n") + Schema + TEXT("\n") + Config;
    for (auto& Pair : Buckets)
        if (Pair.Value.Key == Key)
        { ++Pair.Value.References; OutError.Reset(); return {Pair.Key, Pair.Value.Generation}; }
    FNPCRowLayout Layout;
    auto Backend = Provider.CreateBackend(Definition, Layout, OutError);
    if (!Backend) return {};
    FBucket B;
    if (!Provider.IsA<UNPCPassthroughBackendProvider>())
    {
        auto Codec = FNPCRowCodec::Create(Definition, Layout, OutError);
        if (!Codec) return {};
        B.Codec = MakeShared<FNPCRowCodec, ESPMode::ThreadSafe>(MoveTemp(Codec.GetValue()));
    }
    B.Scheduler = FNPCBackgroundScheduler::Create(Settings, Backend.ToSharedRef(), &OutError);
    if (!B.Scheduler) return {};
    B.Cohorts = MakeShared<FNPCCohortScheduler>(NextId);
    B.Key = Key; B.References = 1; B.Generation = NextGeneration++;
    const FNPCBackgroundBucketHandle H{NextId++, B.Generation};
    Buckets.Add(H.Id, MoveTemp(B));
    return H;
}
void UNPCBackgroundInferenceSubsystem::ReleaseBucket(FNPCBackgroundBucketHandle H)
{
    check(IsInGameThread());
    if (!Find(H)) return;
    auto& B = Buckets.FindChecked(H.Id);
    if (--B.References == 0) { B.Scheduler->Shutdown(); Buckets.Remove(H.Id); }
}
FNPCBackgroundScheduler* UNPCBackgroundInferenceSubsystem::GetScheduler(FNPCBackgroundBucketHandle H) const
{ const auto* B = Find(H); return B ? B->Scheduler.Get() : nullptr; }
TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> UNPCBackgroundInferenceSubsystem::ShareScheduler(FNPCBackgroundBucketHandle H) const
{ const auto* B = Find(H); return B ? B->Scheduler : nullptr; }
const FNPCRowCodec* UNPCBackgroundInferenceSubsystem::GetCodec(FNPCBackgroundBucketHandle H) const
{ const auto* B = Find(H); return B ? B->Codec.Get() : nullptr; }
TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe> UNPCBackgroundInferenceSubsystem::ShareCodec(FNPCBackgroundBucketHandle H) const
{ const auto* B = Find(H); return B ? B->Codec : nullptr; }
TSharedPtr<FNPCCohortScheduler> UNPCBackgroundInferenceSubsystem::ShareCohorts(FNPCBackgroundBucketHandle H) const
{ const auto* B = Find(H); return B ? B->Cohorts : nullptr; }
uint32 UNPCBackgroundInferenceSubsystem::GetGeneration(FNPCBackgroundBucketHandle H) const
{ const auto* B = Find(H); return B ? B->Generation : 0; }
TArray<FNPCBackgroundStats> UNPCBackgroundInferenceSubsystem::GetAllBucketStats() const
{
    TArray<FNPCBackgroundStats> Result;
    for (const auto& Pair : Buckets) Result.Add(Pair.Value.Scheduler->GetStats());
    return Result;
}
void UNPCBackgroundInferenceSubsystem::Deinitialize()
{
    for (auto& Pair : Buckets) Pair.Value.Scheduler->Shutdown();
    Buckets.Empty(); RegisteredSteppers.Empty();
    Super::Deinitialize();
}

void UNPCBackgroundInferenceSubsystem::RegisterStepper(UBackgroundStepper* Stepper)
{ check(IsInGameThread()); RegisteredSteppers.AddUnique(Stepper); }
void UNPCBackgroundInferenceSubsystem::UnregisterStepper(UBackgroundStepper* Stepper)
{ check(IsInGameThread()); RegisteredSteppers.Remove(Stepper); }
void UNPCBackgroundInferenceSubsystem::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    RegisteredSteppers.RemoveAll([](const auto& Stepper) { return !Stepper.IsValid(); });
    // Step can unregister itself when its binding is invalidated.
    for (int32 I = RegisteredSteppers.Num()-1; I >= 0; --I)
        if (RegisteredSteppers.IsValidIndex(I))
            if (auto* Stepper = RegisteredSteppers[I].Get(); Stepper && Stepper->bAutoTick) Stepper->Step(FPlatformTime::Seconds());
}
TStatId UNPCBackgroundInferenceSubsystem::GetStatId() const
{ RETURN_QUICK_DECLARE_CYCLE_STAT(UNPCBackgroundInferenceSubsystem, STATGROUP_Tickables); }
