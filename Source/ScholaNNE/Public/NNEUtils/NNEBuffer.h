// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "NNE.h"
#include "StructUtils/InstancedStruct.h"
#include <NNERuntimeRunSync.h>
#include "NNEBuffer.generated.h"

inline int GetTotalSize(TConstArrayView<int> Shape)
{
	int TotalSize = 1;
	for (int Dim : Shape)
	{
		TotalSize *= Dim;
	}
	return TotalSize;
}

/**
 * @brief Buffer for storing recurrent neural network state across time steps
 *
 * This buffer maintains state information for sequence-based neural network models,
 * storing a sequence of state vectors that can be shifted and updated.
 */
USTRUCT(BlueprintType)
struct SCHOLANNE_API FNNEStateBuffer
{
	GENERATED_BODY()

	/** Buffer storing the sequence of state vectors */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TArray<float> StateBuffer;

	/** Tensor shape for the state buffer (batch fixed to 1 where applicable). */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TArray<int> Shape = TArray<int>();

	/** Maximum Length of the state sequence */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Properties")
	int MaxSeqLen = 0;

	/** Index of the sequence dimension in the shape (or -1 if no sequence dimension). */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Properties")
	int SeqDim = -1;

	/** Flat size of one state vector (product of non-sequence dimensions). */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Properties")
	int StateDimSize = 1;

	FNNEStateBuffer() = default;

	/**
	 * @param[in] Shape Logical tensor shape; first dimension (batch dimension) will be set to 1, as batched state is not currently supported.
	 * @param[in] InMaxSeqLen Maximum sequence length when a sequence dimension is present.
	 */
	FNNEStateBuffer(TConstArrayView<int32> Shape, int InMaxSeqLen = 1)
	{
		this->Shape = Shape;
		this->MaxSeqLen = InMaxSeqLen;
		this->Shape[0] = 1; // Fix batch size at 1 for now, can be extended in the future
		this->SeqDim = this->Shape.FindLast(-1);
		// For now we only support the sequence dimension being the first dimension or not having a sequence dimension at all, but this can be extended in the future
		if (this->HasSequenceDimension())
		{
			this->Shape[SeqDim] = MaxSeqLen;
			TConstArrayView<int> NonSequenceShapeView = TConstArrayView<int>(Shape).Slice(SeqDim + 1, Shape.Num() - (SeqDim + 1));
			this->StateDimSize = GetTotalSize(NonSequenceShapeView);
		}
		else
		{
			this->StateDimSize = GetTotalSize(this->Shape);
		}
		this->StateBuffer.Init(0.0f, GetTotalSize(this->Shape));
	}

	/**
	 * @brief Shifts the state sequence by removing the oldest state and making room for a new one
	 *
	 * Moves all state vectors one position earlier in the sequence, discarding the first state
	 * and preparing the last position for a new state.
	 */
	void Shift()
	{
		// Shift overlapping regions safely
		for (int i = 0; i < MaxSeqLen - 1; i++)
		{
			FMemory::Memmove(
				StateBuffer.GetData() + i * StateDimSize,
				StateBuffer.GetData() + (i + 1) * StateDimSize,
				StateDimSize * sizeof(float));
		}
	}

	/** @return True if Shape contains a sequence axis (marked as -1 during construction). */
	bool HasSequenceDimension() const
	{
		return this->SeqDim != -1;
	}

	/** Zeroes the whole state sequence. */
	void Reset()
	{
		FMemory::Memzero(StateBuffer.GetData(), StateBuffer.Num() * sizeof(float));
	}

	/** Advances recurrent state (shifts sequence) when a sequence dimension exists. */
	void Update()
	{
		if (this->HasSequenceDimension())
		{
			// If there is a sequence dimension, we assume the new state is already in the last position of the buffer after shifting, so no need to copy
			this->Shift();
		}
	}

	/**
	 * @brief Creates an NNE tensor binding for the entire state buffer as input
	 * @return Tensor binding containing the full state sequence
	 */
	UE::NNE::FTensorBindingCPU MakeInputBinding() const
	{
		return { (void*)(StateBuffer.GetData()), GetTotalSize(this->Shape) * sizeof(float) };
	}

	/**
	 * @brief Creates an NNE tensor binding for the last state vector as output
	 * @return Tensor binding pointing to the most recent state position
	 */
	UE::NNE::FTensorBindingCPU MakeOutputBinding() const
	{
		if (this->SeqDim <= 0)
		{
			return { (void*)(StateBuffer.GetData()), GetTotalSize(this->Shape) * sizeof(float) };
		}
		else
		{
			return { (void*)(StateBuffer.GetData() + (MaxSeqLen - 1) * StateDimSize), StateDimSize * sizeof(float) };
		}
	}
};

struct FNNEBufferVisitor;

/**
 * @brief Base class for all NNE point buffers
 *
 * This is an abstract base class that provides the visitor pattern interface for
 * different types of buffers used to store neural network inputs and outputs.
 */
USTRUCT(BlueprintType)
struct SCHOLANNE_API FNNEPointBuffer
{
	GENERATED_BODY()

public:
	virtual ~FNNEPointBuffer() = default;

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	virtual void Accept(FNNEBufferVisitor& Visitor) const
	{
	}
};

/**
 * @brief Buffer for dictionary-structured neural network data
 *
 * Stores multiple named buffers organized as key-value pairs, corresponding to
 * dictionary observation or action spaces in reinforcement learning.
 */
USTRUCT(BlueprintType)
struct SCHOLANNE_API FNNEDictBuffer : public FNNEPointBuffer
{
	GENERATED_BODY(BlueprintType)

	/** Map of named buffers, each corresponding to a dictionary key */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TMap<FString, TInstancedStruct<FNNEPointBuffer>> Buffers;

	virtual ~FNNEDictBuffer() = default;

	FNNEDictBuffer() = default;

	/**
	 * @brief Constructor that initializes the dictionary buffer with a map of buffers
	 * @param[in] InBuffers Map of buffer names to buffer instances
	 */
	FNNEDictBuffer(const TMap<FString, TInstancedStruct<FNNEPointBuffer>>& InBuffers)
		: Buffers(InBuffers)
	{
	}

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	void Accept(FNNEBufferVisitor& Visitor) const override;
};

/**
 * @brief Buffer for discrete action or observation spaces
 *
 * Stores a single selected discrete value.
 */
USTRUCT(BlueprintType)
struct SCHOLANNE_API FNNEDiscreteBuffer : public FNNEPointBuffer
{
	GENERATED_BODY()

	/** Buffer storing the discrete value (not one-hot encoded) */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TArray<int64> Buffer;

	/**
	 * @brief Constructor that initializes the discrete buffer, with a fixed size of 1. Use MultiDiscreteBuffer for multiple discrete values.
	 */
	FNNEDiscreteBuffer()
	{
		Buffer.Init(0, 1);
	}

	virtual ~FNNEDiscreteBuffer() = default;

	/**
	 * @brief Creates an NNE tensor binding for this buffer
	 * @return Tensor binding containing the buffer data
	 */
	UE::NNE::FTensorBindingCPU MakeBinding() const
	{
		return { (void*)Buffer.GetData(), sizeof(int64) };
	}

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	void Accept(FNNEBufferVisitor& Visitor) const override;
};

/**
 * @brief Buffer for multi-discrete action or observation spaces
 *
 * Stores one selected value per multi-discrete dimension, where each dimension can have different numbers of possible values.
 * Used when an agent makes several independent discrete decisions simultaneously.
 */
USTRUCT(BlueprintType)
struct SCHOLANNE_API FNNEMultiDiscreteBuffer : public FNNEPointBuffer
{
	GENERATED_BODY()

	/** Buffer storing the selected value for each multi-discrete dimension */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TArray<int64> Buffer;

	FNNEMultiDiscreteBuffer() = default;

	/**
	 * @brief Constructor that initializes the multi-discrete buffer with a specific size
	 * @param[in] Size Number of multi-discrete dimensions to store
	 */
	FNNEMultiDiscreteBuffer(int Size)
	{
		Buffer.Init(0, Size);
	}

	virtual ~FNNEMultiDiscreteBuffer() = default;

	/**
	 * @brief Creates an NNE tensor binding for this buffer
	 * @return Tensor binding containing the buffer data
	 */
	UE::NNE::FTensorBindingCPU MakeBinding() const
	{
		return { (void*)Buffer.GetData(), Buffer.Num() * sizeof(int64) };
	}

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	void Accept(FNNEBufferVisitor& Visitor) const override;
};

/**
 * @brief Buffer for multi-binary action or observation spaces
 *
 * Stores multiple independent binary values, where each element represents a binary choice (on/off).
 * Common in scenarios with multiple simultaneous boolean decisions.
 */
USTRUCT(BlueprintType)
struct SCHOLANNE_API FNNEMultiBinaryBuffer : public FNNEPointBuffer
{
	GENERATED_BODY()

	/** Buffer storing multiple binary values as bools (true or false) */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TArray<bool> Buffer;

public:
	FNNEMultiBinaryBuffer() = default;

	/**
	 * @brief Constructor that initializes the multi-binary buffer with a specific size
	 * @param[in] Size Number of binary values
	 */
	FNNEMultiBinaryBuffer(int Size)
	{
		Buffer.Init(false, Size);
	}

	virtual ~FNNEMultiBinaryBuffer() = default;

	/**
	 * @brief Creates an NNE tensor binding for this buffer
	 * @return Tensor binding containing the buffer data
	 */
	UE::NNE::FTensorBindingCPU MakeBinding() const
	{
		return { (void*)Buffer.GetData(), Buffer.Num() * sizeof(bool) };
	}

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	void Accept(FNNEBufferVisitor& Visitor) const override;
};

/**
 * @brief Buffer for continuous box-bounded action or observation spaces
 *
 * Stores continuous values within bounded ranges, commonly used for continuous control
 * tasks where actions or observations are real-valued vectors with upper and lower bounds.
 */
USTRUCT(BlueprintType)
struct SCHOLANNE_API FNNEBoxBuffer : public FNNEPointBuffer
{
	GENERATED_BODY()

	/** Buffer storing continuous values */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TArray<float> Buffer;

public:
	FNNEBoxBuffer() = default;
	virtual ~FNNEBoxBuffer() = default;

	/**
	 * @brief Constructor that initializes the box buffer with a specific size
	 * @param[in] Size Dimensionality of the continuous space
	 */
	FNNEBoxBuffer(int Size)
	{
		Buffer.Init(0.0f, Size);
	}

	/**
	 * @brief Creates an NNE tensor binding for this buffer
	 * @return Tensor binding containing the buffer data
	 */
	UE::NNE::FTensorBindingCPU MakeBinding() const
	{
		return { (void*)Buffer.GetData(), Buffer.Num() * sizeof(float) };
	}

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	void Accept(FNNEBufferVisitor& Visitor) const override;
};
