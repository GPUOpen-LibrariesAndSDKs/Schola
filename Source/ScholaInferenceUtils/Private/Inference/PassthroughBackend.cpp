// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Inference/NPCBackgroundBackendProvider.h"
namespace
{
class FPassthroughContext : public IBatchInferenceContext
{
    virtual bool RunBatch(const float* Observations, int32 Rows, float* Actions) override
    { FMemory::Memcpy(Actions, Observations, Rows * sizeof(float)); return true; }
};
class FPassthroughBackend : public IBatchInferenceBackend
{
    virtual int32 GetObservationSize() const override { return 1; }
    virtual int32 GetActionSize() const override { return 1; }
    virtual FString GetName() const override { return TEXT("Passthrough"); }
    virtual TUniquePtr<IBatchInferenceContext> CreateContext() override { return MakeUnique<FPassthroughContext>(); }
};
}
TSharedPtr<IBatchInferenceBackend> UNPCPassthroughBackendProvider::CreateBackend(const FInteractionDefinition&,
    FNPCRowLayout& OutLayout, FString& OutError)
{
    OutError.Reset();
    OutLayout = {};
    OutLayout.ObsSize = OutLayout.ActSize = 1;
    OutLayout.Inputs = OutLayout.Outputs = {{TEXT("index"), ENPCRowDType::Float, 0, 1}};
    OutLayout.UpdateHash();
    return MakeShared<FPassthroughBackend>();
}
