// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "Inference/BatchInferenceBackend.h"
#include "Inference/NPCRowCodec.h"
#include "NPCBackgroundBackendProvider.generated.h"

UCLASS(Abstract, EditInlineNew, BlueprintType)
class SCHOLAINFERENCEUTILS_API UNPCBackgroundBackendProvider : public UObject
{
    GENERATED_BODY()
public:
    /** Called on the game thread, before workers are started. */
    virtual TSharedPtr<IBatchInferenceBackend> CreateBackend(const FInteractionDefinition& Definition,
        FNPCRowLayout& OutLayout, FString& OutError) PURE_VIRTUAL(UNPCBackgroundBackendProvider::CreateBackend, return nullptr;);
    virtual FString GetBucketKey() const { return GetPathName(); }
};

/** Index transport for prototypes. The policy keeps the source points until consumption. */
UCLASS(EditInlineNew, BlueprintType)
class SCHOLAINFERENCEUTILS_API UNPCPassthroughBackendProvider : public UNPCBackgroundBackendProvider
{
    GENERATED_BODY()
public:
    virtual TSharedPtr<IBatchInferenceBackend> CreateBackend(const FInteractionDefinition&,
        FNPCRowLayout& OutLayout, FString& OutError) override;
    virtual FString GetBucketKey() const override { return TEXT("Schola.Passthrough.v1"); }
};
