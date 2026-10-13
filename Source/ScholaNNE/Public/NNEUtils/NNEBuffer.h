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
 * @brief Shape of one recurrent state tensor, resolved from the model's tensor description
 *
 * Holds everything needed to allocate an FNNEStateBuffer without the zeroed data itself,
 * so a policy can create new states without parsing the model again.
 */
USTRUCT(BlueprintType)
struct SCHOLANNE_API FNNEStateLayout
{
	GENERATED_BODY()

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

	FNNEStateLayout() = default;

	/**
	 * @param[in] InShape Logical tensor shape; first dimension (batch dimension) will be set to 1, as batched state is not currently supported.
	 * @param[in] InMaxSeqLen Maximum sequence length when a sequence dimension is present.
	 */
	explicit FNNEStateLayout(TConstArrayView<int32> InShape, int InMaxSeqLen = 1)
	{
		this->Shape = InShape;
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
	}

	/** @return True if Shape contains a sequence axis (marked as -1 in the model). */
	bool HasSequenceDimension() const
	{
		return this->SeqDim != -1;
	}

	/** @return Number of floats in the whole state sequence, i.e. the state_in tensor. */
	int GetNumElements() const
	{
		return GetTotalSize(this->Shape);
	}

	/** @return Number of floats in an FNNEStateBuffer: the state sequence plus one pending state vector for state_out. */
	int GetBufferNumElements() const
	{
		return this->GetNumElements() + this->StateDimSize;
	}
};

/**
 * @brief Buffer for storing recurrent neural network state across time steps
 *
 * This buffer maintains state information for sequence-based neural network models,
 * storing a sequence of state vectors that can be shifted and updated. It holds one
 * extra state vector after the sequence, which the model's state_out tensor writes to
 * so it never overlaps the state_in tensor. Update moves it into the sequence.
 */
USTRUCT(BlueprintType)
struct SCHOLANNE_API FNNEStateBuffer
{
	GENERATED_BODY()

	/** Buffer storing the sequence of state vectors, followed by the pending state vector */
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
	 * @brief Allocates a zeroed state sequence with the given layout
	 * @param[in] Layout The shape of the state tensor
	 */
	explicit FNNEStateBuffer(const FNNEStateLayout& Layout)
		: Shape(Layout.Shape), MaxSeqLen(Layout.MaxSeqLen), SeqDim(Layout.SeqDim), StateDimSize(Layout.StateDimSize)
	{
		this->StateBuffer.Init(0.0f, Layout.GetBufferNumElements());
	}

	/**
	 * @param[in] InShape Logical tensor shape; first dimension (batch dimension) will be set to 1, as batched state is not currently supported.
	 * @param[in] InMaxSeqLen Maximum sequence length when a sequence dimension is present.
	 */
	FNNEStateBuffer(TConstArrayView<int32> InShape, int InMaxSeqLen = 1)
		: FNNEStateBuffer(FNNEStateLayout(InShape, InMaxSeqLen))
	{
	}

	/**
	 * @brief Shifts every state vector one position earlier, including the pending one
	 *
	 * Discards the oldest state and moves the pending state into the last position of the
	 * sequence. Without a sequence dimension, the pending state replaces the whole state.
	 */
	void Shift()
	{
		FMemory::Memmove(StateBuffer.GetData(), StateBuffer.GetData() + StateDimSize, (StateBuffer.Num() - StateDimSize) * sizeof(float));
	}

	/** Zeroes the whole state sequence. */
	void Reset()
	{
		FMemory::Memzero(StateBuffer.GetData(), StateBuffer.Num() * sizeof(float));
	}

	/** Commits the pending state written by the model, dropping the oldest state of the sequence. */
	void Update()
	{
		this->Shift();
	}

	/**
	 * @brief Creates an NNE tensor binding for the state sequence as input
	 * @return Tensor binding containing the full state sequence, excluding the pending state
	 */
	UE::NNE::FTensorBindingCPU MakeInputBinding() const
	{
		return { (void*)(StateBuffer.GetData()), GetTotalSize(this->Shape) * sizeof(float) };
	}

	/**
	 * @brief Creates an NNE tensor binding for the pending state vector as output
	 * @return Tensor binding pointing after the state sequence, so it never overlaps the input binding
	 */
	UE::NNE::FTensorBindingCPU MakeOutputBinding() const
	{
		return { (void*)(StateBuffer.GetData() + StateBuffer.Num() - StateDimSize), StateDimSize * sizeof(float) };
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
