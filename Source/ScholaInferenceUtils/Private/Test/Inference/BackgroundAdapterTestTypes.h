// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "Inference/NPCBackgroundBackendProvider.h"
#include "Inference/DeterministicBatchBackend.h"
#include "Steppers/BackgroundStepper.h"
#include "../Steppers/GatedStepperTestTypes.h"
#include "BackgroundAdapterTestTypes.generated.h"

UCLASS()
class UBackgroundAdapterTestProvider : public UNPCBackgroundBackendProvider
{
    GENERATED_BODY()
public:
    TSharedPtr<FDeterministicBatchBackend> Backend;
    TSharedPtr<IBatchInferenceBackend> GenericBackend;
    virtual FString GetBucketKey() const override { return TEXT("Schola.AdapterTest"); }
    virtual TSharedPtr<IBatchInferenceBackend> CreateBackend(const FInteractionDefinition&,
        FNPCRowLayout& Layout, FString& Error) override
    {
        Error.Reset(); Layout = {};
        Layout.ObsSize = Layout.ActSize = 1;
        Layout.Inputs = Layout.Outputs = {{TEXT("value"), ENPCRowDType::Float, 0, 1}};
        Layout.UpdateHash(); return GenericBackend ? GenericBackend : Backend;
    }
};

UCLASS()
class UBackgroundCallbackRowAgent : public USequenceTestAgent, public INPCBackgroundRowAgent
{
    GENERATED_BODY()
public:
    virtual void WriteObservationRow(TArrayView<float> Row) override
    {
        if (OnObserve) OnObserve();
        Row[0] = float(++ObservationCounter);
    }
    virtual void ApplyActionRow(TConstArrayView<float> Row) override
    {
        if (OnAct) OnAct();
        Applied.Add(Row[0]);
    }
};
