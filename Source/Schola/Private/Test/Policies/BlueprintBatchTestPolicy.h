// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "Policies/BlueprintPolicy.h"
#include "BlueprintBatchTestPolicy.generated.h"

/** Simulates Blueprint Think dispatch without creating a UI or Blueprint asset. */
UCLASS()
class UBlueprintBatchTestPolicy : public UBlueprintPolicy
{
    GENERATED_BODY()
public:
    bool bBatchOverride = false;
    bool bWrongCount = false;
    bool bBusy = false;
    int32 ThinkCalls = 0;
    int32 BatchCalls = 0;

    void ProcessEvent(UFunction* Function, void* Parameters) override
    {
        if (Function->GetFName() == TEXT("Think"))
        {
            struct FThinkParameters
            {
                FInstancedStruct InObservations;
                FInstancedStruct OutAction;
            };
            auto* Args = static_cast<FThinkParameters*>(Parameters);
            Args->OutAction = Args->InObservations;
            ++ThinkCalls;
            return;
        }
        Super::ProcessEvent(Function, Parameters);
    }

    bool BatchedThink_Implementation(const TArray<FInstancedStruct>& Observations,
        TArray<FInstancedStruct>& Actions) override
    {
        if (!bBatchOverride) { return Super::BatchedThink_Implementation(Observations, Actions); }
        ++BatchCalls;
        Actions = Observations;
        if (bWrongCount) { Actions.Reset(); }
        return true;
    }

    bool IsInferenceBusy_Implementation() const override { return bBusy; }
};
