// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "IREEModelData.h"
#include "Dom/JsonObject.h"
#include "IREE/IREERuntimeSession.h"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonWriter.h"
#include "Serialization/JsonSerializer.h"
#include "LogScholaIREE.h"

namespace
{
	bool ParseJsonIntArray(const TArray<TSharedPtr<FJsonValue>>& Values, TArray<int32>& OutShape)
	{
		OutShape.Reset();
		for (const TSharedPtr<FJsonValue>& Value : Values)
		{
			if (!Value.IsValid() || Value->Type != EJson::Number)
			{
				return false;
			}
			OutShape.Add(static_cast<int32>(Value->AsNumber()));
		}
		return true;
	}

	/** Prefixes that turn a state's ``name`` into its model input / output tensor name (must match ``iree_export.py``). */
	const TCHAR* const StateInPrefix = TEXT("state_in_");
	const TCHAR* const StateOutPrefix = TEXT("state_out_");

	/** Parses one ``{name, dtype, shape}`` object (the common part of observation, action and state specs). */
	bool ParseTensorInfo(const TSharedPtr<FJsonObject>& Spec, FScholaIREETensorInfo& OutTensor)
	{
		Spec->TryGetStringField(TEXT("name"), OutTensor.Name);
		Spec->TryGetStringField(TEXT("dtype"), OutTensor.Dtype);
		const TArray<TSharedPtr<FJsonValue>>* Shape = nullptr;
		return !Spec->TryGetArrayField(TEXT("shape"), Shape) || ParseJsonIntArray(*Shape, OutTensor.Shape);
	}

	/** Gets the object elements of the array field ``FieldName``; a missing field yields no elements. */
	bool GetSpecObjects(const TSharedPtr<FJsonObject>& Root, const TCHAR* FieldName, TArray<TSharedPtr<FJsonObject>>& OutObjects)
	{
		OutObjects.Reset();
		const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
		if (!Root->TryGetArrayField(FieldName, Values))
		{
			return true;
		}
		for (const TSharedPtr<FJsonValue>& Value : *Values)
		{
			const TSharedPtr<FJsonObject>* Object = nullptr;
			if (!Value.IsValid() || !Value->TryGetObject(Object) || !Object->IsValid())
			{
				return false;
			}
			OutObjects.Add(*Object);
		}
		return true;
	}

	/**
	 * Parses ``inputs`` / ``outputs`` (observation and action tensors only) into ``Inputs`` / ``Outputs``, and
	 * ``state`` into ``States`` together with each state's parsed ``state_metadata``.
	 */
	bool ParseTensorSpecs(const TSharedPtr<FJsonObject>& Root, FScholaIREEModelMetadata& OutMetadata)
	{
		TArray<TSharedPtr<FJsonObject>> InputSpecs, OutputSpecs, StateSpecs;
		if (!GetSpecObjects(Root, TEXT("inputs"), InputSpecs)
			|| !GetSpecObjects(Root, TEXT("outputs"), OutputSpecs)
			|| !GetSpecObjects(Root, TEXT("state"), StateSpecs))
		{
			return false;
		}

		for (const TSharedPtr<FJsonObject>& Spec : InputSpecs)
		{
			if (!ParseTensorInfo(Spec, OutMetadata.Inputs.AddDefaulted_GetRef()))
			{
				return false;
			}
		}
		for (const TSharedPtr<FJsonObject>& Spec : OutputSpecs)
		{
			if (!ParseTensorInfo(Spec, OutMetadata.Outputs.AddDefaulted_GetRef()))
			{
				return false;
			}
		}

		for (const TSharedPtr<FJsonObject>& Spec : StateSpecs)
		{
			FScholaIREETensorInfo Tensor;
			if (!ParseTensorInfo(Spec, Tensor))
			{
				return false;
			}

			FScholaIREEStateInfo& State = OutMetadata.States.AddDefaulted_GetRef();
			State.Name = Tensor.Name;
			State.Dtype = Tensor.Dtype;
			State.Shape = Tensor.Shape;

			const TSharedPtr<FJsonObject>* StateMetadata = nullptr;
			if (Spec->TryGetObjectField(TEXT("state_metadata"), StateMetadata) && StateMetadata->IsValid())
			{
				// max_seq_len / seq_dim may be JSON null, in which case they keep their INDEX_NONE defaults.
				(*StateMetadata)->TryGetBoolField(TEXT("has_seq_dim"), State.StateMetadata.bHasSeqDim);
				double Value = 0.0;
				if ((*StateMetadata)->TryGetNumberField(TEXT("max_seq_len"), Value))
				{
					State.StateMetadata.MaxSeqLen = static_cast<int32>(Value);
				}
				if ((*StateMetadata)->TryGetNumberField(TEXT("seq_dim"), Value))
				{
					State.StateMetadata.SeqDim = static_cast<int32>(Value);
				}
			}
		}
		return true;
	}
}

FString FScholaIREEStateInfo::GetInName() const
{
	return StateInPrefix + Name;
}

FString FScholaIREEStateInfo::GetOutName() const
{
	return StateOutPrefix + Name;
}

void UIREEModelData::PostLoad()
{
	Super::PostLoad();
	if (!MetadataJson.IsEmpty())
	{
		ParseMetadata();
	}
}

const TCHAR* const UIREEModelData::ReflectionKey = TEXT("schola.metadata");

bool UIREEModelData::LoadMetadataFromVmfb()
{
	if (!FIREERuntimeSession::ReadReflectionAttribute(VmfbBytes, ReflectionKey, MetadataJson))
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEModelData::LoadMetadataFromVmfb: no {0} reflection attribute in vmfb", ReflectionKey);
		return false;
	}
	return ParseMetadata();
}

bool UIREEModelData::ParseMetadata()
{
	bMetadataParsed = false;
	Metadata = FScholaIREEModelMetadata();

	TSharedPtr<FJsonObject> Root;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(MetadataJson);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEModelData::ParseMetadata: invalid JSON");
		return false;
	}

	Root->TryGetStringField(TEXT("entry_function"), Metadata.EntryFunction);
	Root->TryGetStringArrayField(TEXT("target_backends"), Metadata.TargetBackends);
	Root->TryGetStringField(TEXT("iree_version"), Metadata.IreeVersion);
	double BatchSize = static_cast<double>(Metadata.BatchSize);
	if (Root->TryGetNumberField(TEXT("batch_size"), BatchSize))
	{
		Metadata.BatchSize = static_cast<int32>(BatchSize);
	}

	if (!ParseTensorSpecs(Root, Metadata))
	{
		UE_LOGFMT(LogScholaIREE, Error, "UIREEModelData::ParseMetadata: failed to parse inputs / outputs / state");
		return false;
	}

	bMetadataParsed = true;
	return true;
}