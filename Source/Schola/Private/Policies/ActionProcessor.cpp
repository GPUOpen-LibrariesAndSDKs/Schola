// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Policies/ActionProcessor.h"
#include "Common/LogSchola.h"

bool FActionProcessor::Init(int32 InChunkSize, int32 InActionDim)
{
	if (InChunkSize <= 0 || InActionDim <= 0)
	{
		UE_LOGFMT(LogSchola, Error, "FActionProcessor::Init(): Got chunk size {0} and action size {1}, both must be positive",
			InChunkSize, InActionDim);
		return false;
	}
	ChunkSize = InChunkSize;
	ActionDim = InActionDim;
	return true;
}

TInstancedStruct<FActionProcessor> FActionProcessor::CreateForModel(TConstArrayView<FString> OutputNames)
{
	TArray<FString> StructNames;
	for (const FString& OutputName : OutputNames)
	{
		if (OutputName.StartsWith(MarkerOutputPrefix, ESearchCase::CaseSensitive))
		{
			StructNames.Add(OutputName.RightChop(FCString::Strlen(MarkerOutputPrefix)));
		}
	}
	if (StructNames.Num() != 1)
	{
		UE_LOGFMT(LogSchola, Error, "FActionProcessor::CreateForModel(): A model that predicts action chunks needs exactly one output named {0}<processor> saying how to play them, got {1}. Export it with lerobot-to-onnx",
			MarkerOutputPrefix, StructNames.Num());
		return TInstancedStruct<FActionProcessor>();
	}

	// Object names ignore case, so the struct's own name is checked against the model's
	const UScriptStruct* Struct = FindFirstObject<UScriptStruct>(*StructNames[0], EFindFirstObjectOptions::NativeFirst);
	if (!Struct || !Struct->GetName().Equals(StructNames[0], ESearchCase::CaseSensitive) || !Struct->IsChildOf(StaticStruct()))
	{
		UE_LOGFMT(LogSchola, Error, "FActionProcessor::CreateForModel(): The model names action processor {0}, which is not a struct derived from FActionProcessor",
			StructNames[0]);
		return TInstancedStruct<FActionProcessor>();
	}
	return TInstancedStruct<FActionProcessor>(Struct);
}

bool FActionProcessor::Process(const FActionChunk& Chunk, TArrayView<float> OutAction)
{
	if (!IsActionValid(OutAction) || !IsChunkValid(Chunk))
	{
		return false;
	}
	CopyAction(Chunk.Actions, 0, OutAction);
	return true;
}

bool FActionProcessor::IsSameLayout(const TInstancedStruct<FActionProcessor>& A, const TInstancedStruct<FActionProcessor>& B)
{
	if (A.GetScriptStruct() != B.GetScriptStruct())
	{
		return false;
	}
	const FActionProcessor* ProcessorA = A.GetPtr();
	const FActionProcessor* ProcessorB = B.GetPtr();
	return !ProcessorA || (ProcessorA->ChunkSize == ProcessorB->ChunkSize && ProcessorA->ActionDim == ProcessorB->ActionDim);
}

bool FActionProcessor::CopyInPlace(TInstancedStruct<FActionProcessor>& Target, const TInstancedStruct<FActionProcessor>& Source)
{
	if (!IsSameLayout(Target, Source))
	{
		return false;
	}
	if (const UScriptStruct* Struct = Source.GetScriptStruct())
	{
		Struct->CopyScriptStruct(Target.GetMutableMemory(), Source.GetMemory());
	}
	return true;
}

bool FActionProcessor::IsActionValid(TArrayView<float> OutAction) const
{
	if (ActionDim <= 0)
	{
		UE_LOGFMT(LogSchola, Error, "FActionProcessor::Process(): Call Init before Process");
		return false;
	}
	if (OutAction.Num() != ActionDim)
	{
		UE_LOGFMT(LogSchola, Error, "FActionProcessor::Process(): OutAction has {0} floats, expected ActionDim {1}",
			OutAction.Num(), ActionDim);
		return false;
	}
	return true;
}

bool FActionProcessor::IsChunkValid(const FActionChunk& Chunk) const
{
	if (Chunk.Actions.Num() != NumChunkElements())
	{
		UE_LOGFMT(LogSchola, Error, "FActionProcessor::Process(): Chunk has {0} floats, expected {1}",
			Chunk.Actions.Num(), NumChunkElements());
		return false;
	}
	return true;
}

void FActionProcessor::CopyAction(TConstArrayView<float> Actions, int32 Index, TArrayView<float> OutAction) const
{
	check(Index >= 0 && (Index + 1) * ActionDim <= Actions.Num() && OutAction.Num() == ActionDim);
	FMemory::Memcpy(OutAction.GetData(), Actions.GetData() + Index * ActionDim, ActionDim * sizeof(float));
}
