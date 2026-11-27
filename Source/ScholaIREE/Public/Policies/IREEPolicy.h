// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "IREEModelData.h"
#include "Common/InteractionDefinition.h"
#include "Common/InstancedStructUtils.h"
#include "Common/BlueprintErrorUtils.h"
#include "IREEUtils/IREEBuffer.h"
#include "Policies/PolicyInterface.h"
#include "Policies/IREEPolicyState.h"
#include <atomic>
#include "IREEPolicy.generated.h"

class FIREERuntimeSession;

/**
 * @brief Policy implementation that runs Schola-exported IREE ``.vmfb`` models.
 *
 * Executes inference through the IREE C runtime (not Unreal NNE).
 */
UCLASS(Blueprintable, BlueprintType, EditInlineNew, DefaultToInstanced)
class SCHOLAIREE_API UIREEPolicy : public UObject, public IPolicy
{
	GENERATED_BODY()

public:
	UIREEPolicy(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());
	UIREEPolicy(FVTableHelper& Helper);
	virtual ~UIREEPolicy() override;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Definition")
	FInteractionDefinition PolicyDefinition;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ExposeOnSpawn = true), Category = "Policy Properties")
	TObjectPtr<UIREEModelData> ModelData;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ExposeOnSpawn = true), Category = "Policy Properties")
	FString DeviceURI = TEXT("local-sync");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ExposeOnSpawn = true, ClampMin = 1), Category = "Policy Properties")
	int MaxStateSequenceLength = 1;

	UPROPERTY(VisibleAnywhere, Category = "Policy Properties")
	bool bNetworkLoaded = false;

	UFUNCTION(BlueprintCallable, Category = "Schola|Policy")
	bool Think(const FInstancedStruct& InObservations, const TScriptInterface<IPolicyState>& InOutState, FInstancedStruct& OutAction)
	{
		if (!InObservations.GetScriptStruct())
		{
			RaiseInvalidInstancedStructError(TEXT("IREEPolicy::Think"));
			return false;
		}
		if (!InObservations.GetScriptStruct()->IsChildOf(FPoint::StaticStruct()))
		{
			RaiseInstancedStructTypeMismatchError(InObservations, TEXT("FPoint"), TEXT("IREEPolicy::Think"));
			return false;
		}
		if (OutAction.GetScriptStruct() && !OutAction.GetScriptStruct()->IsChildOf(FPoint::StaticStruct()))
		{
			RaiseInstancedStructTypeMismatchError(OutAction, TEXT("FPoint"), TEXT("IREEPolicy::Think"));
			return false;
		}
		return Think(ToTypedInstancedStruct<FPoint>(InObservations), InOutState, ToTypedInstancedStruct<FPoint>(OutAction));
	}

	virtual bool CreateInitialState(TScriptInterface<IPolicyState>& OutState) const override;

	UFUNCTION(BlueprintCallable, Category = "Schola|Policy")
	TScriptInterface<IPolicyState> CreateInitialState() const
	{
		TScriptInterface<IPolicyState> State;
		CreateInitialState(State);
		return State;
	}

	virtual bool Think(
		const TInstancedStruct<FPoint>& InObservations,
		const TScriptInterface<IPolicyState>& InOutState,
		TInstancedStruct<FPoint>& OutActions) override;

	UFUNCTION(BlueprintCallable, Category = "Schola|Policy")
	bool Init(const FInteractionDefinition& InPolicyDefinition) override;

	/** Buffer storing action data for neural network output */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Data")
	TInstancedStruct<FIREEPointBuffer> ActionBuffer;

	/** Buffer storing observation data for neural network input */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Data")
	TInstancedStruct<FIREEPointBuffer> ObservationBuffer;

	/** Layout of each state tensor pair in the model, read once in Init and used to allocate new states */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Data")
	TArray<FIREEStateLayout> StateLayouts;

	UFUNCTION(BlueprintCallable, Category = "Policy Properties")
	bool IsInferenceBusy() const override;

protected:
	bool InitStateBuffersAndBindings(const FScholaIREEModelMetadata& Metadata);
	static UIREEPolicyState* AsIREEPolicyState(const TScriptInterface<IPolicyState>& InState);
	bool IsStateCompatible(const UIREEPolicyState& InState) const;
	bool BindState(UIREEPolicyState& InOutState);
	void InitNonStateBuffers(const FInteractionDefinition& InDefinition);
	bool InitNonStateBindings(const FInteractionDefinition& InDefinition);
	bool SetupBuffersAndBindings(const FInteractionDefinition& PolicyDefinition, const FScholaIREEModelMetadata& Metadata);
	bool ResetRuntimeSession();

private:
	TUniquePtr<FIREERuntimeSession> RuntimeSession;
	TArray<FScholaIREETensorInfo> InputTensorDescs;
	TArray<FScholaIREETensorInfo> OutputTensorDescs;
	TArray<FIREETensorBinding> InputBindings;
	TArray<FIREETensorBinding> OutputBindings;
	TArray<int32> StateInputBindingIndices;
	TArray<int32> StateOutputBindingIndices;
	TWeakObjectPtr<UIREEPolicyState> BoundState;
	std::atomic<bool> bInferenceInFlight { false };
};