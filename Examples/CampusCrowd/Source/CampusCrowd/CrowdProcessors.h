// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "MassProcessor.h"
#include "MassEntityQuery.h"
#include "ScholaMassInferenceProcessors.h"
#include "CrowdTypes.h"
#include "CrowdProcessors.generated.h"

class ACampusScene;
class FCampusMigrationDay;

namespace UE::NNE { class IModelCPU; class IModelInstanceCPU; }

/** With the API: writes observation rows for agents the ScholaMass scheduler marks due. */
UCLASS()
class UCrowdGatherProcessor : public UScholaMassGatherProcessor
{
    GENERATED_BODY()
public:
    UCrowdGatherProcessor();
    const FCampusMigrationDay* MigrationDay = nullptr;
    float HalfSize = 0;
protected:
    virtual void ConfigureObservationRequirements(FMassEntityQuery& Query) override;
    virtual void WriteObservations(FMassExecutionContext& Context, TConstArrayView<int32> DueIndices, TArrayView<float> Rows) override;
};

/** With the API: applies completed background actions on the game thread. */
UCLASS()
class UCrowdApplyProcessor : public UScholaMassApplyProcessor
{
    GENERATED_BODY()
public:
    UCrowdApplyProcessor();
    FCampusMigrationDay* MigrationDay = nullptr;
    FCrowdFrameCounters* Counters = nullptr;
protected:
    virtual void ConfigureActionRequirements(FMassEntityQuery& Query) override;
    virtual void ApplyActions(FMassExecutionContext& Context, TConstArrayView<int32> EntityIndices, TConstArrayView<float> Rows) override;
};

/**
 * Without the API: the straightforward approach. Each due agent builds its
 * observation and runs one batch-1 inference on the game thread.
 */
UCLASS()
class UCrowdSyncInferenceProcessor : public UMassProcessor
{
    GENERATED_BODY()
public:
    UCrowdSyncInferenceProcessor();
    FCampusMigrationDay* MigrationDay = nullptr;
    TSharedPtr<UE::NNE::IModelCPU> Model;
    TSharedPtr<UE::NNE::IModelInstanceCPU> Instance;
    FCrowdFrameCounters* Counters = nullptr;
    float HalfSize = 0, PeriodSeconds = 0.5f;
    int64 ModelCalls = 0;
    bool bFailed = false, bUseSimulationClock = false;
protected:
    virtual void ConfigureQueries(const TSharedRef<FMassEntityManager>& Manager) override;
    virtual void Execute(FMassEntityManager& Manager, FMassExecutionContext& Context) override;
    FMassEntityQuery Query;
};

/** Shared by both modes: steers toward the latest action and writes instance transforms. */
UCLASS()
class UCrowdMoveProcessor : public UMassProcessor
{
    GENERATED_BODY()
public:
    UCrowdMoveProcessor();
    const ACampusScene* Campus = nullptr;
    TArray<FTransform>* Transforms = nullptr;
    float HalfSize = 0;
protected:
    virtual void ConfigureQueries(const TSharedRef<FMassEntityManager>& Manager) override;
    virtual void Execute(FMassEntityManager& Manager, FMassExecutionContext& Context) override;
    FMassEntityQuery Query;
    // Separation grid, rebuilt each frame by counting sort and reused to avoid per-frame allocation.
    TArray<FVector2f> Positions;
    TArray<int32> CellOf, CellStart, CellFill, Sorted;
};
