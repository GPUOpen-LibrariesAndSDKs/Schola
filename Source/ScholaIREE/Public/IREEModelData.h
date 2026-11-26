// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "IREEModelData.generated.h"

/**
 * @brief One model input or output tensor as described by the metadata embedded in the ``.vmfb``.
 */
USTRUCT(BlueprintType)
struct SCHOLAIREE_API FScholaIREETensorInfo
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	FString Name;

	/** numpy-style dtype name, e.g. ``float32`` or ``int64``. Defaults to ``float32`` when the metadata omits it. */
	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	FString Dtype = TEXT("float32");

	/** Fixed tensor shape including the batch dimension. Empty when the metadata has no shape for this tensor. */
	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	TArray<int32> Shape;
};

/**
 * @brief Sequence information for a recurrent state. Mirrors ``StateMetadata`` in ``schola/core/model.py``.
 */
USTRUCT(BlueprintType)
struct SCHOLAIREE_API FScholaIREEStateMetadata
{
	GENERATED_BODY()

	/** Whether the state tensor has a sequence dimension. */
	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	bool bHasSeqDim = false;

	/** Maximum sequence length. Only meaningful when ``bHasSeqDim`` is true; ``INDEX_NONE`` otherwise. */
	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	int32 MaxSeqLen = INDEX_NONE;

	/** Index of the sequence dimension. Only meaningful when ``bHasSeqDim`` is true; ``INDEX_NONE`` otherwise. */
	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	int32 SeqDim = INDEX_NONE;
};

/**
 * @brief A recurrent state, passed in as ``state_in_<name>`` and returned as ``state_out_<name>``.
 * Mirrors ``StateSpec`` in ``schola/core/export/iree_export.py``. The input and output tensors share dtype and shape.
 */
USTRUCT(BlueprintType)
struct SCHOLAIREE_API FScholaIREEStateInfo
{
	GENERATED_BODY()

	/** Unprefixed state name. */
	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	FString Name;

	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	FString Dtype = TEXT("float32");

	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	TArray<int32> Shape;

	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	FScholaIREEStateMetadata StateMetadata;

	/** Name of the model input tensor for this state (``state_in_<name>``). */
	FString GetInName() const;

	/** Name of the model output tensor for this state (``state_out_<name>``). */
	FString GetOutName() const;
};

/**
 * @brief Parsed Schola metadata embedded in an IREE ``.vmfb`` via ``iree.reflection``.
 */
USTRUCT(BlueprintType)
struct SCHOLAIREE_API FScholaIREEModelMetadata
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	FString EntryFunction = TEXT("main");

	/** Observation inputs in call order. Recurrent state tensors are not included; see ``States``. */
	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	TArray<FScholaIREETensorInfo> Inputs;

	/** Action outputs in return order. Recurrent state tensors are not included; see ``States``. */
	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	TArray<FScholaIREETensorInfo> Outputs;

	/**
	 * Recurrent states. Each is passed to the entry function as ``state_in_<name>`` after the observations and
	 * returned as ``state_out_<name>`` after the actions.
	 */
	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	TArray<FScholaIREEStateInfo> States;

	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	TArray<FString> TargetBackends;

	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	FString IreeVersion;

	UPROPERTY(BlueprintReadOnly, Category = "IREE")
	int32 BatchSize = 1;
};

/**
 * @brief UObject asset holding a compiled IREE ``.vmfb`` and parsed embedded metadata.
 */
UCLASS(BlueprintType)
class SCHOLAIREE_API UIREEModelData : public UObject
{
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "IREE")
	TArray<uint8> VmfbBytes;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "IREE")
	FString MetadataJson;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "IREE")
	FScholaIREEModelMetadata Metadata;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "IREE")
	bool bMetadataParsed = false;

	/** Re-parses ``MetadataJson`` after load so assets imported before the metadata layout changed still pick up new fields. */
	virtual void PostLoad() override;

	/** Parse ``MetadataJson`` into ``Metadata``. */
	UFUNCTION(BlueprintCallable, Category = "Schola|IREE")
	bool ParseMetadata();

	/** Key inside a function's ``iree.reflection`` attribute that holds the metadata JSON. */
	static const TCHAR* const ReflectionKey;

	/** Read the metadata JSON embedded in ``VmfbBytes`` (``iree.reflection["schola.metadata"]``) and parse it. */
	UFUNCTION(BlueprintCallable, Category = "Schola|IREE")
	bool LoadMetadataFromVmfb();
};