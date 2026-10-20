// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "MassProcessor.h"
#include "MassObserverProcessor.h"
#include "MassSignalProcessorBase.h"
#include "MassEntityQuery.h"
#include "Inference/CohortScheduler.h"
#include "ScholaMassInferenceProcessors.generated.h"

UCLASS(Abstract)
class SCHOLAMASS_API UScholaMassGatherProcessor : public UMassProcessor
{
    GENERATED_BODY()
public:
    UScholaMassGatherProcessor();
    UPROPERTY(EditAnywhere, Config, Category="Schola|Inference") ENPCPhaseLayout PhaseLayout = ENPCPhaseLayout::ModelAware;
    UPROPERTY(EditAnywhere, Config, Category="Schola|Inference") int32 ParallelGatherMinEntities = 2048;
    UPROPERTY(EditAnywhere, Config, Category="Schola|Inference") int32 MaxRowsPerFrame = 4096;
    UPROPERTY(EditAnywhere, Config, Category="Schola|Inference") float MaxActionAgeMs = 100.f;
protected:
    virtual void ConfigureQueries(const TSharedRef<FMassEntityManager>& Manager) override;
    virtual void ConfigureObservationRequirements(FMassEntityQuery&) {}
    virtual void WriteObservations(FMassExecutionContext& Context, TConstArrayView<int32> DueIndices,
        TArrayView<float> Rows) PURE_VIRTUAL(UScholaMassGatherProcessor::WriteObservations, );
    virtual void Execute(FMassEntityManager& Manager, FMassExecutionContext& Context) override;
    FMassEntityQuery Query;
};
UCLASS(Abstract)
class SCHOLAMASS_API UScholaMassApplyProcessor : public UMassProcessor
{
    GENERATED_BODY()
public:
    UScholaMassApplyProcessor();
    UPROPERTY(EditAnywhere, Config, Category="Schola|Inference") float MaxActionAgeMs = 100.f;
    UPROPERTY(EditAnywhere, Config, Category="Schola|Inference") FName ActionReadySignal = NAME_None;
protected:
    virtual void ConfigureQueries(const TSharedRef<FMassEntityManager>& Manager) override;
    virtual void ConfigureActionRequirements(FMassEntityQuery&) {}
    virtual void ApplyActions(FMassExecutionContext& Context, TConstArrayView<int32> EntityIndices,
        TConstArrayView<float> Rows) PURE_VIRTUAL(UScholaMassApplyProcessor::ApplyActions, );
    virtual void Execute(FMassEntityManager& Manager, FMassExecutionContext& Context) override;
    FMassEntityQuery Query;
};
UCLASS()
class SCHOLAMASS_API UScholaMassScheduleObserver : public UMassObserverProcessor
{
    GENERATED_BODY()
public:
    UScholaMassScheduleObserver();
protected:
    virtual void ConfigureQueries(const TSharedRef<FMassEntityManager>& Manager) override;
    virtual void Execute(FMassEntityManager& Manager, FMassExecutionContext& Context) override;
    FMassEntityQuery Query;
};

/** Explicit wake-up hook; no automatic significance heuristic is enabled. */
UCLASS()
class SCHOLAMASS_API UScholaMassDecideNowProcessor : public UMassSignalProcessorBase
{
    GENERATED_BODY()
public:
    UScholaMassDecideNowProcessor();
protected:
    virtual void InitializeInternal(UObject& Owner, const TSharedRef<FMassEntityManager>& Manager) override;
    virtual void ConfigureQueries(const TSharedRef<FMassEntityManager>& Manager) override;
    virtual void SignalEntities(FMassEntityManager& Manager, FMassExecutionContext& Context, FMassSignalNameLookup& Signals) override;
};
