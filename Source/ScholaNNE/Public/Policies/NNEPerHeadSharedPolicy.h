// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "NNEModelData.h"
#include "Policies/SharedPolicyInterface.h"
#include "Inference/BatchInferenceBackend.h"
#include "Inference/NPCRowCodec.h"
#include "NNEPerHeadSharedPolicy.generated.h"

/**
 * @brief ISharedPolicy backed by one Unreal NNE model per head ("Architecture C").
 *
 * Each entry in HeadModels maps a head name to a complete model (backbone plus that head's layers)
 * with one float input [batch, ObservationWidth] and one float output [batch, ActionWidth]. Every
 * head must read the same ObservationWidth because the stepper sends one observation format for all
 * NPCs; action widths may differ between heads.
 *
 * Unlike UNNESharedPolicy, which runs every head on every row in one call, this policy runs each
 * row only through its own head's model, at the cost of one model call per head present in a batch
 * and one stored copy of the backbone per head.
 *
 * Call Init() once on the game thread, then use the policy from one thread at a time.
 */
UCLASS(BlueprintType, EditInlineNew, DefaultToInstanced)
class SCHOLANNE_API UNNEPerHeadSharedPolicy : public UObject, public ISharedPolicy
{
	GENERATED_BODY()

public:
	/** Imported ONNX model for each head, keyed by head name. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|SharedPolicy")
	TMap<FName, TObjectPtr<UNNEModelData>> HeadModels;

	/** NNE runtime. The batch backend supports NNERuntimeORTCpu and NNERuntimeBasicCpu. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|SharedPolicy")
	FString RuntimeName = TEXT("NNERuntimeORTCpu");

	/** Largest number of rows per model call. 0 means each model's own limit (unbounded for a dynamic batch). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|SharedPolicy", meta = (ClampMin = "0"))
	int32 MaxBatchSize = 0;

	/**
	 * @brief Loads every head's model and checks its tensor contract.
	 *
	 * Fails, leaving the policy not ready, if HeadModels is empty, a head has no name or no model, a
	 * model is not exactly one float [batch, width] input and one float [batch, width] output, or two
	 * heads read different observation widths.
	 *
	 * @return True if the policy is ready for BatchedThinkWithHeads.
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|SharedPolicy")
	bool Init();

	/** True after a successful Init(). */
	UFUNCTION(BlueprintPure, Category = "Schola|SharedPolicy")
	bool IsReady() const { return !Heads.IsEmpty(); }

	/** Floats per observation row, shared by every head, or 0 before Init(). */
	int32 GetObservationWidth() const { return ObservationWidth; }

	/** Number of model calls made by BatchedThinkWithHeads, including calls that failed. */
	int64 GetModelCallCount() const { return ModelCallCount; }

	/**
	 * @brief Groups rows by head, runs each present head's model once on its rows, and returns actions in input order.
	 *
	 * Every row is validated and encoded before any model runs: the batch must be non-empty,
	 * Observations and InHeads must have equal length, each head must be registered and its row count
	 * within MaxBatchSize, and each observation must be a finite FBoxPoint of ObservationWidth values.
	 * Any failure returns false, logs the reason, and leaves OutActions unchanged.
	 *
	 * @param[in] Observations One FBoxPoint per NPC.
	 * @param[in] InHeads One registered head name per NPC.
	 * @param[out] OutActions One FBoxPoint per NPC, sized to that NPC's head. Written only on success.
	 */
	virtual bool BatchedThinkWithHeads(const TArray<TInstancedStruct<FPoint>>& Observations,
		const TArray<FName>& InHeads, TArray<TInstancedStruct<FPoint>>& OutActions) override;

	/** Heads loaded by Init(). Empty before a successful Init(). */
	virtual void GetRegisteredHeads(TSet<FName>& OutHeads) const override;

private:
	/** One head's loaded model and its execution state. */
	struct FHeadModel
	{
		TSharedPtr<IBatchInferenceBackend> Backend;
		TUniquePtr<IBatchInferenceContext> Context;
		TOptional<FNPCRowCodec> Codec;
		int32 ActionWidth = 0;
	};

	TMap<FName, FHeadModel> Heads;
	int32 ObservationWidth = 0;
	int64 ModelCallCount = 0;
};
