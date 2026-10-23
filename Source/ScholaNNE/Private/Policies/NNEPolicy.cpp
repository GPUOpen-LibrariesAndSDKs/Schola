// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Policies/NNEPolicy.h"
#include "Misc/ScopeExit.h"
#include "UObject/Package.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNERuntimeRunSync.h"
#include "NNEStatus.h"
#include "NNEUtils/NNEPointToBufferConverter.h"
#include "NNEUtils/NNEBufferAllocator.h"
#include "NNEUtils/NNEPointCreator.h"
#include "NNEUtils/NNEBindingCreator.h"
#include "LogScholaNNE.h"

TArray<FString> UNNEPolicy::GetRuntimeNames() const
{
	// we don't support RDG yet so skip it here
	// TArray < FString> ValidRuntimes = UE::NNE::GetAllRuntimeNames < INNERuntimeCPU>();
	// ValidRuntimes.Append(UE::NNE::GetAllRuntimeNames<INNERuntimeGPU>());
	return UE::NNE::GetAllRuntimeNames();
}

IRuntimeInterface* UNNEPolicy::GetRuntime(const FString& SelectedRuntimeName) const
{
	TWeakInterfacePtr<INNERuntimeCPU> CPUPtr = UE::NNE::GetRuntime<INNERuntimeCPU>(SelectedRuntimeName);
	if (CPUPtr.IsValid())
	{
		return new FCPURuntimeWrapper(CPUPtr);
	}

	TWeakInterfacePtr<INNERuntimeGPU> GPUPtr = UE::NNE::GetRuntime<INNERuntimeGPU>(SelectedRuntimeName);
	if (GPUPtr.IsValid())
	{
		return new FGPURuntimeWrapper(GPUPtr);
	}
	// Should probably never happen but
	return nullptr;
}

bool UNNEPolicy::Think(
	const TInstancedStruct<FPoint>&		  InObservations,
	const TScriptInterface<IPolicyState>& InOutState,
	TInstancedStruct<FPoint>&			  OutAction)
{
	bool Expected = false;
	if (!bInferenceInFlight.compare_exchange_weak(Expected, true, std::memory_order_acq_rel))
	{
		UE_LOGFMT(LogScholaNNE, Verbose, "NNEPolicy::Think(): Inference already in flight");
		return false;
	}

	// Scope guard to reset inference flag
	ON_SCOPE_EXIT
	{
		bInferenceInFlight.store(false, std::memory_order_release);
	};

	// If the network hasn't been initialized, bail immediately
	if (!bNetworkLoaded)
	{
		UE_LOGFMT(LogScholaNNE, Error, "NNEPolicy::Think(): Network not loaded, returning false");
		return false;
	}

	// Initialize the OutAction container lazily, but only after initialization
	if (!OutAction.IsValid())
	{
		if (!ActionBuffer.IsValid() || !PolicyDefinition.ActionSpaceDefn.IsValid())
		{
			UE_LOGFMT(LogScholaNNE, Error, "NNEPolicy::Think(): ActionBuffer or ActionSpace is invalid after init");
			return false;
		}
		// Initialize the OutAction with the ActionSpace Definition
		FNNEPointCreator::CreatePoint(ActionBuffer, OutAction, PolicyDefinition.ActionSpaceDefn);
	}

	UNNEPolicyState* NNEState = AsNNEPolicyState(InOutState);
	if (!NNEState || !BindState(*NNEState))
	{
		return false;
	}

	// A chunked model only runs when its chunk needs a new prediction
	const bool bRunModel = !IsActionChunked() || NNEState->ActionChunker.NeedsNewChunk();
	if (bRunModel)
	{
		FNNEPointToBufferConverter::ConvertPointToBuffer(InObservations, ObservationBuffer, PolicyDefinition.ObsSpaceDefn);
		if ((int)ModelInstance->RunSync(InputBindings, OutputBindings) != 0)
		{
			UE_LOGFMT(LogScholaNNE, Error, "NNEPolicy::Think(): Model inference failed");
			return false;
		}
	}

	if (IsActionChunked())
	{
		const TConstArrayView<float> NewChunk = bRunModel ? TConstArrayView<float>(ActionChunkBuffer) : TConstArrayView<float>();
		if (!NNEState->ActionChunker.Consume(NewChunk, ChunkedAction))
		{
			UE_LOGFMT(LogScholaNNE, Error, "NNEPolicy::Think(): Failed to take this step's action from the action chunk");
			return false;
		}
	}

	// Materialize all inference outputs before committing the caller-owned recurrent state.
	FNNEPointCreator::CreatePoint(ActionBuffer, OutAction, PolicyDefinition.ActionSpaceDefn);

	// The model wrote its new state into each buffer's pending slot; this moves it into the sequence.
	if (bRunModel && !NNEState->NotifyStateUpdate())
	{
		UE_LOGFMT(LogScholaNNE, Error, "NNEPolicy::Think(): NotifyStateUpdate failed");
		return false;
	}
	return true;
}

bool UNNEPolicy::InitInputTensorShapes(TSharedPtr<IModelInstanceRunSync> InModelInstance)
{
	TArray<UE::NNE::FTensorShape> TempShapeArray;

	for (const UE::NNE::FTensorDesc& TensorDesc : InModelInstance->GetInputTensorDescs())
	{
		TempShapeArray.Add(UE::NNE::FTensorShape::MakeFromSymbolic(TensorDesc.GetShape()));
	}

	return UE::NNE::IModelInstanceRunSync::ESetInputTensorShapesStatus::Ok == InModelInstance->SetInputTensorShapes(TempShapeArray);
}

bool UNNEPolicy::InitStateBuffersAndBindings(TSharedPtr<IModelInstanceRunSync> InModelInstance)
{
	this->StateLayouts.Reset();
	this->StateInputBindingIndices.Reset();
	this->StateOutputBindingIndices.Reset();
	this->BoundState.Reset();

	// Find all the State Tensors from the Tensor Descriptions and record the layout of each one.
	// The state bindings point into the caller's state object, so they are set per Think in BindState.
	TConstArrayView<UE::NNE::FTensorDesc> InputTensorDescs = InModelInstance->GetInputTensorDescs();
	for (int i = 0; i < InputTensorDescs.Num(); i++)
	{
		if (InputTensorDescs[i].GetName().StartsWith(TEXT("state_in")))
		{
			this->StateLayouts.Emplace(InputTensorDescs[i].GetShape().GetData(), this->MaxStateSequenceLength);
			this->StateInputBindingIndices.Add(i);
		}
	}
	// Go through the output tensors and find the state tensors, pairing them with the input state tensors in order
	TConstArrayView<UE::NNE::FTensorDesc> OutputTensorDescs = InModelInstance->GetOutputTensorDescs();
	for (int i = 0; i < OutputTensorDescs.Num(); i++)
	{
		if (OutputTensorDescs[i].GetName().StartsWith(TEXT("state_out")))
		{
			this->StateOutputBindingIndices.Add(i);
		}
	}

	if (this->StateInputBindingIndices.Num() != this->StateOutputBindingIndices.Num())
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::InitStateBuffersAndBindings(): Model has {0} state_in tensors but {1} state_out tensors, expected one of each per state",
			this->StateInputBindingIndices.Num(), this->StateOutputBindingIndices.Num());
		return false;
	}
	return true;
}

UNNEPolicyState* UNNEPolicy::AsNNEPolicyState(const TScriptInterface<IPolicyState>& InState)
{
	UNNEPolicyState* NNEState = Cast<UNNEPolicyState>(InState.GetObject());
	if (!NNEState)
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::AsNNEPolicyState(): Expected a UNNEPolicyState, got {0}",
			InState.GetObject() ? InState.GetObject()->GetClass()->GetName() : FString(TEXT("None")));
	}
	return NNEState;
}

bool UNNEPolicy::IsStateCompatible(const UNNEPolicyState& InState) const
{
	if (InState.Buffers.Num() != this->StateLayouts.Num())
	{
		return false;
	}
	const int32 ChunkActionDim = this->ChunkedAction.Num();
	const int32 ChunkSize = IsActionChunked() ? this->ActionChunkSize : 0;
	if (InState.ActionChunker.ChunkSize != ChunkSize || InState.ActionChunker.ActionDim != ChunkActionDim)
	{
		return false;
	}
	for (int i = 0; i < InState.Buffers.Num(); i++)
	{
		const FNNEStateBuffer& Buffer = InState.Buffers[i];
		const FNNEStateLayout& Layout = this->StateLayouts[i];
		if (Buffer.Shape != Layout.Shape
			|| Buffer.StateDimSize != Layout.StateDimSize
			|| Buffer.StateBuffer.Num() != Layout.GetBufferNumElements())
		{
			return false;
		}
	}
	return true;
}

bool UNNEPolicy::BindState(UNNEPolicyState& InOutState)
{
	if (!IsStateCompatible(InOutState))
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::BindState(): {0} does not match this model's state tensors or action chunk, it must be created by CreateInitialState for this model",
			InOutState.GetName());
		return false;
	}

	// Bindings are left in place after Think, so reusing the same state skips rebinding
	if (this->BoundState.Get() == &InOutState)
	{
		return true;
	}

	for (int i = 0; i < this->StateLayouts.Num(); i++)
	{
		this->InputBindings[this->StateInputBindingIndices[i]] = InOutState.Buffers[i].MakeInputBinding();
		this->OutputBindings[this->StateOutputBindingIndices[i]] = InOutState.Buffers[i].MakeOutputBinding();
	}
	this->BoundState = &InOutState;
	return true;
}

bool UNNEPolicy::CreateInitialState(TScriptInterface<IPolicyState>& OutState) const
{
	OutState = nullptr;
	if (!bNetworkLoaded)
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::CreateInitialState(): Network not loaded, call Init first");
		return false;
	}
	UNNEPolicyState* State = NewObject<UNNEPolicyState>(GetTransientPackage());
	if (IsActionChunked())
	{
		State->ActionChunker.Init(this->ActionChunkSize, this->ChunkedAction.Num(), this->ActionChunking);
	}
	State->Buffers.Reserve(this->StateLayouts.Num());
	for (const FNNEStateLayout& Layout : this->StateLayouts)
	{
		State->Buffers.Emplace(Layout);
	}
	OutState = State;
	return true;
}

void UNNEPolicy::InitNonStateBuffers(const FInteractionDefinition& InDefinition)
{
	FNNEBufferAllocator::AllocateBuffer(InDefinition.ObsSpaceDefn, this->ObservationBuffer);
	FNNEBufferAllocator::AllocateBuffer(InDefinition.ActionSpaceDefn, this->ActionBuffer);
}

bool UNNEPolicy::InitNonStateBindings(const FInteractionDefinition& InDefinition)
{
	// Observation Bindings
	if (FNNEBindingCreator::CreateBindings(InDefinition.ObsSpaceDefn, this->ObservationBuffer, ModelInstance->GetInputTensorDescs(), this->InputBindings))
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::InitNonStateBindings(): Failed to create bindings for Observation Space");
		return false;
	}

	// Action Bindings
	if (FNNEBindingCreator::CreateBindings(InDefinition.ActionSpaceDefn, this->ActionBuffer, ModelInstance->GetOutputTensorDescs(), this->OutputBindings))
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::InitNonStateBindings(): Failed to create bindings for Action Space");
		return false;
	}
	return true;
}

int32 UNNEPolicy::GetActionChunkSize(TConstArrayView<int32> OutputShape, int32 ActionDim)
{
	if (ActionDim <= 0)
	{
		return 1;
	}
	int64 NumElements = 1;
	for (int32 i = 0; i < OutputShape.Num(); i++)
	{
		if (OutputShape[i] < 0)
		{
			if (i != 0)
			{
				return 1;
			}
			continue;
		}
		NumElements *= OutputShape[i];
	}
	if (NumElements <= ActionDim)
	{
		return 1;
	}
	return NumElements % ActionDim == 0 ? static_cast<int32>(NumElements / ActionDim) : 0;
}

bool UNNEPolicy::InitActionChunking(TSharedPtr<IModelInstanceRunSync> InModelInstance)
{
	this->ActionChunkSize = 1;
	this->ActionChunkBuffer.Reset();
	this->ChunkedAction = TArrayView<float>();

	TConstArrayView<UE::NNE::FTensorDesc> OutputTensorDescs = InModelInstance->GetOutputTensorDescs();
	for (int32 i = 0; i < OutputTensorDescs.Num(); i++)
	{
		// State outputs are only bound per Think in BindState, so here they have no data
		const UE::NNE::FTensorDesc& TensorDesc = OutputTensorDescs[i];
		const UE::NNE::FTensorBindingCPU& Binding = this->OutputBindings[i];
		if (!Binding.Data || TensorDesc.GetDataType() != ENNETensorDataType::Float)
		{
			continue;
		}

		const int32 ActionDim = static_cast<int32>(Binding.SizeInBytes / sizeof(float));
		const int32 ChunkSize = GetActionChunkSize(TensorDesc.GetShape().GetData(), ActionDim);
		if (ChunkSize == 0)
		{
			UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::InitActionChunking(): Output {0} is larger than its action of size {1} but is not a whole number of actions",
				TensorDesc.GetName(), ActionDim);
			return false;
		}
		if (ChunkSize > 1)
		{
			if (IsActionChunked())
			{
				UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::InitActionChunking(): Output {0} is a second action chunk, only models with one chunked action output are supported",
					TensorDesc.GetName());
				return false;
			}

			this->ActionChunkSize = ChunkSize;
			this->ChunkedAction = TArrayView<float>(static_cast<float*>(Binding.Data), ActionDim);
			this->ActionChunkBuffer.SetNumZeroed(ChunkSize * ActionDim);
			this->OutputBindings[i] = { (void*)this->ActionChunkBuffer.GetData(), this->ActionChunkBuffer.Num() * sizeof(float) };
			UE_LOGFMT(LogScholaNNE, Log, "UNNEPolicy::InitActionChunking(): Output {0} predicts {1} actions per inference", TensorDesc.GetName(), ChunkSize);
		}
	}
	return true;
}

bool UNNEPolicy::AllocateBindingArrays(TSharedPtr<IModelInstanceRunSync> InModelInstance)
{
	InputBindings.Init({ nullptr, 0 }, InModelInstance->GetInputTensorDescs().Num());
	OutputBindings.Init({ nullptr, 0 }, InModelInstance->GetOutputTensorDescs().Num());

	if (InputBindings.Num() == 0 || OutputBindings.Num() == 0)
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::AllocateBindingArrays(): No Input or Output Bindings found for the Model. Must be at least one input and one output binding for the model to run");
		return false;
	}

	return true;
}

bool UNNEPolicy::SetupBuffersAndBindings(const FInteractionDefinition& InDefinition, TSharedPtr<IModelInstanceRunSync> InModelInstance)
{
	// Add Empty Entries to the Input and Output Bindings, so we can update the entries instead of Allocating later.
	if (!AllocateBindingArrays(InModelInstance))
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::SetupBuffersAndBindings(): Failed to allocate input or output bindings");
		return false;
	}

	// Allocate memory for the buffers corresponding to Actions and Observations
	InitNonStateBuffers(InDefinition);

	// Prepare buffers and bindings for state tensors, e.g. as used in RNNs or LSTM networks
	if (!InitStateBuffersAndBindings(InModelInstance))
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::SetupBuffersAndBindings(): Failed to initialize state buffers");
		return false;
	}

	// Bind the remaining inputs and outputs to the Action and Observation buffers
	InitNonStateBindings(InDefinition);

	// Rebind an action output that predicts several actions per inference, e.g. ACT
	if (!InitActionChunking(InModelInstance))
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::SetupBuffersAndBindings(): Failed to initialize action chunking");
		return false;
	}

	if (!InitInputTensorShapes(InModelInstance))
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::SetupBuffersAndBindings(): Failed to initialize input tensor shapes");
		return false;
	}

	return true;
}

bool UNNEPolicy::Init(const FInteractionDefinition& InPolicyDefinition)
{
	// Store the policy definition for later use
	this->PolicyDefinition = InPolicyDefinition;

	if (!ModelData)
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::Init(): Failed to create network due to invalid model data");
		// Invalid Model Data
		bNetworkLoaded = false;
		return false;
	}

	// Create and store runtime in the class member to avoid shadowing and ensure lifetime
	this->Runtime = TUniquePtr<IRuntimeInterface>(this->GetRuntime(this->RuntimeName));
	if (!this->Runtime.IsValid() || !this->Runtime->IsValid())
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::Init(): Cannot find runtime {0}, please enable the corresponding plugin", RuntimeName);
		// Invalid Runtime
		bNetworkLoaded = false;
		return false;
	}

	TUniquePtr<IModelInterface> TempModelPtr = this->Runtime->CreateModel(ModelData);
	if (!TempModelPtr.IsValid() || !TempModelPtr->IsValid())
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::Init(): Failed to create the model");
		// Invalid Runtime
		bNetworkLoaded = false;
		return false;
	}

	// Store the model in the class member so it outlives the instance
	this->Model = MoveTemp(TempModelPtr);

	ModelInstance = this->Model->CreateModelInstance();
	if (!ModelInstance.IsValid())
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::Init(): Failed to create the model instance");
		bNetworkLoaded = false;
		return false;
	}

	if (!this->SetupBuffersAndBindings(InPolicyDefinition, ModelInstance))
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPolicy::Init(): Failed to setup buffers and bindings");
		bNetworkLoaded = false;
		return false;
	}

	bNetworkLoaded = true;
	return true;
}

bool UNNEPolicy::IsInferenceBusy() const
{
	return bInferenceInFlight.load(std::memory_order_relaxed);
}
