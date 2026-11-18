// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "NNEModelData.h"
#include "Policies/SharedPolicyInterface.h"
#include "Inference/BatchInferenceBackend.h"
#include "Inference/NPCRowCodec.h"
#include "NNESharedPolicy.generated.h"

/**
 * @brief ISharedPolicy backed by one Unreal NNE model that computes every head ("Architecture A").
 *
 * The model must have one float input [batch, ObservationWidth] and one float output
 * [batch, ActionWidth] per head. An output named "<Head><HeadOutputSuffix>" serves head <Head>; with
 * the default suffix the benchmark's A_all_heads model exposes the heads "expert_0" and "expert_1".
 *
 * BatchedThinkWithHeads makes a single model call for the whole batch, then copies each NPC's row
 * out of that NPC's head output. Row order is the order of the arguments, so no regrouping or
 * un-shuffling is needed. Reshaping, staging and output-shape checks are delegated to
 * UNNEBatchBackendProvider, and observation encoding to FNPCRowCodec.
 *
 * Call Init() once on the game thread, then use the policy from one thread at a time.
 */
UCLASS(BlueprintType, EditInlineNew, DefaultToInstanced)
class SCHOLANNE_API UNNESharedPolicy : public UObject, public ISharedPolicy
{
	GENERATED_BODY()

public:
	/** Imported ONNX model that exposes all heads. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|SharedPolicy")
	TObjectPtr<UNNEModelData> ModelData;

	/** NNE runtime. The batch backend supports NNERuntimeORTCpu and NNERuntimeBasicCpu. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|SharedPolicy")
	FString RuntimeName = TEXT("NNERuntimeORTCpu");

	/** Largest NPC batch per call. 0 means the model's own limit (unbounded for a dynamic batch). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|SharedPolicy", meta = (ClampMin = "0"))
	int32 MaxBatchSize = 0;

	/** Output tensor name suffix that identifies a head. Outputs without it are not heads. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|SharedPolicy")
	FString HeadOutputSuffix = TEXT("_actions");

	/**
	 * @brief Loads the model, checks its tensor contract, and discovers the heads.
	 *
	 * Fails, leaving the policy not ready, if the runtime or model is missing, if the model does not
	 * have exactly one float [batch, width] input, if any output is not a float [batch, width] tensor,
	 * or if no output names a head.
	 *
	 * @return True if the policy is ready for BatchedThinkWithHeads.
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|SharedPolicy")
	bool Init();

	/** True after a successful Init(). */
	UFUNCTION(BlueprintPure, Category = "Schola|SharedPolicy")
	bool IsReady() const { return Context.IsValid(); }

	/** Floats per observation row, or 0 before Init(). */
	int32 GetObservationWidth() const { return ObservationWidth; }

	/** Number of model calls made by BatchedThinkWithHeads, including calls that failed. */
	int64 GetModelCallCount() const { return ModelCallCount; }

	/**
	 * @brief Runs one NNE call for all rows and returns each NPC's own head output in input order.
	 *
	 * Every row is validated before the model runs: the batch must be non-empty and within
	 * MaxBatchSize, Observations and Heads must have equal length, each head must be registered, and
	 * each observation must be a finite FBoxPoint of ObservationWidth values. The NNE output shapes
	 * are validated after the run. Any failure returns false, logs the reason, and leaves OutActions
	 * unchanged.
	 *
	 * @param[in] Observations One FBoxPoint per NPC.
	 * @param[in] Heads One registered head name per NPC.
	 * @param[out] OutActions One FBoxPoint per NPC, sized to that NPC's head. Written only on success.
	 */
	virtual bool BatchedThinkWithHeads(const TArray<TInstancedStruct<FPoint>>& Observations,
		const TArray<FName>& Heads, TArray<TInstancedStruct<FPoint>>& OutActions) override;

	/** Heads discovered by Init(). Empty before a successful Init(). */
	virtual void GetRegisteredHeads(TSet<FName>& OutHeads) const override;

private:
	/** Where one head's values sit inside a row of the model's concatenated outputs. */
	struct FHeadOutput
	{
		int32 Offset = 0;
		int32 Width = 0;
	};

	TSharedPtr<IBatchInferenceBackend> Backend;
	TUniquePtr<IBatchInferenceContext> Context;
	TOptional<FNPCRowCodec> Codec;
	TMap<FName, FHeadOutput> HeadOutputs;
	int32 ObservationWidth = 0;
	int32 ActionRowWidth = 0;
	int64 ModelCallCount = 0;
};
