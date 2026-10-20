// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "Inference/NPCBackgroundBackendProvider.h"
#include "NNEModelData.h"
#include "NNEBatchBackendProvider.generated.h"

UCLASS(EditInlineNew, BlueprintType)
class SCHOLANNE_API UNNEBatchBackendProvider : public UNPCBackgroundBackendProvider
{
    GENERATED_BODY()
public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schola|Inference") TObjectPtr<UNNEModelData> ModelData;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schola|Inference") FString RuntimeName = TEXT("NNERuntimeORTCpu");
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schola|Inference", meta=(ClampMin="0")) int32 MaxBatchSize = 0;
    /** Enable if input reshaping costs more than 5% of service time at batch 32. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schola|Inference") bool bPadToPowerOfTwo = false;
    virtual TSharedPtr<IBatchInferenceBackend> CreateBackend(const FInteractionDefinition& Definition,
        FNPCRowLayout& OutLayout, FString& OutError) override;
    virtual FString GetBucketKey() const override;
};
