// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Policies/IREEPolicy.h"
#include "IREE/IREERuntimeSession.h"
#include "IREEUtils/IREEBindingCreator.h"
#include "Misc/ScopeExit.h"
#include "IREEUtils/IREEPointToBufferConverter.h"
#include "IREEUtils/IREEBufferAllocator.h"
#include "IREEUtils/IREEPointCreator.h"
#include "UObject/Package.h"
#include "LogScholaIREE.h"

UIREEPolicy::UIREEPolicy(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
}

UIREEPolicy::UIREEPolicy(FVTableHelper& Helper)
	: Super(Helper)
{
}

UIREEPolicy::~UIREEPolicy() = default;

bool UIREEPolicy::ResetRuntimeSession()
{
	RuntimeSession = MakeUnique<FIREERuntimeSession>();
	if (!RuntimeSession->Initialize(DeviceURI, ModelData->VmfbBytes))
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::ResetRuntimeSession: failed to initialize IREE session");
		RuntimeSession.Reset();
		return false;
	}
	return true;
}

bool UIREEPolicy::InitStateBuffersAndBindings(const FScholaIREEModelMetadata& Metadata)
{
	StateLayouts.Reset();
	StateInputBindingIndices.Reset();
	StateOutputBindingIndices.Reset();
	BoundState.Reset();

	for (int32 Index = 0; Index < Metadata.States.Num(); ++Index)
	{
		StateInputBindingIndices.Add(InputTensorDescs.Num());
		StateOutputBindingIndices.Add(OutputTensorDescs.Num());

		const FScholaIREEStateInfo& State = Metadata.States[Index];
		InputTensorDescs.Add({State.GetInName(), State.Dtype, State.Shape});

		StateLayouts.Emplace(State.Shape, State.StateMetadata.MaxSeqLen);
		OutputTensorDescs.Add({State.GetOutName(), State.Dtype, State.Shape});
	}

	return true;
}

UIREEPolicyState* UIREEPolicy::AsIREEPolicyState(const TScriptInterface<IPolicyState>& InState)
{
	UIREEPolicyState* State = Cast<UIREEPolicyState>(InState.GetObject());
	if (!State)
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::AsIREEPolicyState: expected UIREEPolicyState, got {0}",
			InState.GetObject() ? InState.GetObject()->GetClass()->GetName() : FString(TEXT("None")));
	}
	return State;
}

bool UIREEPolicy::IsStateCompatible(const UIREEPolicyState& InState) const
{
	if (InState.Buffers.Num() != StateLayouts.Num())
	{
		return false;
	}
	for (int32 Index = 0; Index < InState.Buffers.Num(); ++Index)
	{
		const FIREEStateBuffer& Buffer = InState.Buffers[Index];
		const FIREEStateLayout& Layout = StateLayouts[Index];
		if (Buffer.Shape != Layout.Shape || Buffer.StateDimSize != Layout.StateDimSize
			|| Buffer.StateBuffer.Num() != Layout.GetBufferNumElements())
		{
			return false;
		}
	}
	return true;
}

bool UIREEPolicy::BindState(UIREEPolicyState& InOutState)
{
	if (!IsStateCompatible(InOutState))
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::BindState: incompatible state object");
		return false;
	}
	if (BoundState.Get() == &InOutState)
	{
		return true;
	}
	for (int32 Index = 0; Index < StateLayouts.Num(); ++Index)
	{
		InputBindings[StateInputBindingIndices[Index]] = InOutState.Buffers[Index].MakeInputBinding();
		OutputBindings[StateOutputBindingIndices[Index]] = InOutState.Buffers[Index].MakeOutputBinding();
	}
	BoundState = &InOutState;
	return true;
}

bool UIREEPolicy::CreateInitialState(TScriptInterface<IPolicyState>& OutState) const
{
	OutState = nullptr;
	if (!bNetworkLoaded)
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::CreateInitialState: network not loaded");
		return false;
	}
	UIREEPolicyState* State = NewObject<UIREEPolicyState>(GetTransientPackage());
	State->Buffers.Reserve(StateLayouts.Num());
	for (const FIREEStateLayout& Layout : StateLayouts)
	{
		State->Buffers.Emplace(Layout);
	}
	OutState = State;
	return true;
}

void UIREEPolicy::InitNonStateBuffers(const FInteractionDefinition& InDefinition)
{
	FIREEBufferAllocator::AllocateBuffer(InDefinition.ObsSpaceDefn, ObservationBuffer);
	FIREEBufferAllocator::AllocateBuffer(InDefinition.ActionSpaceDefn, ActionBuffer);
}

bool UIREEPolicy::InitNonStateBindings(const FInteractionDefinition& InDefinition)
{
	if (FIREEBindingCreator::CreateBindings(InDefinition.ObsSpaceDefn, ObservationBuffer, InputTensorDescs, InputBindings))
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::InitNonStateBindings: failed observation bindings");
		return false;
	}
	if (FIREEBindingCreator::CreateBindings(InDefinition.ActionSpaceDefn, ActionBuffer, OutputTensorDescs, OutputBindings))
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::InitNonStateBindings: failed action bindings");
		return false;
	}
	return true;
}

bool UIREEPolicy::SetupBuffersAndBindings(const FInteractionDefinition& InDefinition, const FScholaIREEModelMetadata& Metadata)
{
	// Copy rather than point into ModelData, so a model swapped after Init cannot invalidate the bound layout.
	// InitStateBuffersAndBindings then appends the state tensors, giving the entry function's full call
	// signature: observations then state_in_<name>, actions then state_out_<name>.
	InputTensorDescs = Metadata.Inputs;
	OutputTensorDescs = Metadata.Outputs;

	InitNonStateBuffers(InDefinition);

	if (!InitStateBuffersAndBindings(Metadata))
	{
		return false;
	}

	if (InputTensorDescs.Num() == 0 || OutputTensorDescs.Num() == 0)
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::SetupBuffersAndBindings: model metadata has no inputs or outputs");
		return false;
	}

	// Sized after the state tensors are appended so BindState can index into them.
	InputBindings.Init(FIREETensorBinding(), InputTensorDescs.Num());
	OutputBindings.Init(FIREETensorBinding(), OutputTensorDescs.Num());

	if (!InitNonStateBindings(InDefinition))
	{
		return false;
	}

	return true;
}

bool UIREEPolicy::Think(
	const TInstancedStruct<FPoint>& InObservations,
	const TScriptInterface<IPolicyState>& InOutState,
	TInstancedStruct<FPoint>& OutAction)
{
	bool Expected = false;
	if (!bInferenceInFlight.compare_exchange_weak(Expected, true, std::memory_order_acq_rel))
	{
		UE_LOGFMT(LogScholaIREE, Verbose, "UIREEPolicy::Think: inference already in flight");
		return false;
	}

	ON_SCOPE_EXIT { bInferenceInFlight.store(false, std::memory_order_release); };

	if (!bNetworkLoaded || !RuntimeSession.IsValid())
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::Think: network not loaded");
		return false;
	}

	if (!OutAction.IsValid())
	{
		if (!ActionBuffer.IsValid() || !PolicyDefinition.ActionSpaceDefn.IsValid())
		{
			UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::Think: action buffer invalid after init");
			return false;
		}
		FIREEPointCreator::CreatePoint(ActionBuffer, OutAction, PolicyDefinition.ActionSpaceDefn);
	}

	UIREEPolicyState* PolicyState = AsIREEPolicyState(InOutState);
	if (!PolicyState || !BindState(*PolicyState))
	{
		return false;
	}

	FIREEPointToBufferConverter::ConvertPointToBuffer(InObservations, ObservationBuffer, PolicyDefinition.ObsSpaceDefn);

	if (!RuntimeSession->Invoke(
			ModelData->Metadata.EntryFunction,
			InputBindings,
			InputTensorDescs,
			OutputBindings,
			OutputTensorDescs))
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::Think: IREE invoke failed");
		return false;
	}

	FIREEPointCreator::CreatePoint(ActionBuffer, OutAction, PolicyDefinition.ActionSpaceDefn);

	if (!PolicyState->NotifyStateUpdate())
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::Think: NotifyStateUpdate failed");
		return false;
	}

	return true;
}

bool UIREEPolicy::Init(const FInteractionDefinition& InPolicyDefinition)
{
	PolicyDefinition = InPolicyDefinition;

	if (!ModelData || ModelData->VmfbBytes.Num() == 0)
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::Init: invalid model data");
		bNetworkLoaded = false;
		return false;
	}

	if (!ModelData->bMetadataParsed && !ModelData->ParseMetadata())
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::Init: metadata not parsed");
		bNetworkLoaded = false;
		return false;
	}

	if (!ResetRuntimeSession())
	{
		bNetworkLoaded = false;
		return false;
	}

	if (!SetupBuffersAndBindings(InPolicyDefinition, ModelData->Metadata))
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEPolicy::Init: failed to setup buffers and bindings");
		bNetworkLoaded = false;
		return false;
	}

	bNetworkLoaded = true;
	return true;
}

bool UIREEPolicy::IsInferenceBusy() const
{
	return bInferenceInFlight.load(std::memory_order_relaxed);
}