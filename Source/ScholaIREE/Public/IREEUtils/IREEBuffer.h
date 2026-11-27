// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "IREEUtils/IREETensorBinding.h"
#include "StructUtils/InstancedStruct.h"
#include "IREEBuffer.generated.h"

namespace ScholaIREE
{
	inline int GetTotalSize(TConstArrayView<int> Shape)
	{
		int TotalSize = 1;
		for (int Dim : Shape)
		{
			TotalSize *= Dim;
		}
		return TotalSize;
	}
} // namespace ScholaIREE

/**
 * @brief Shape of one recurrent state tensor, resolved from the model's tensor description
 *
 * Holds everything needed to allocate an FIREEStateBuffer without the zeroed data itself,
 * so a policy can create new states without parsing the model again.
 */
USTRUCT(BlueprintType)
struct SCHOLAIREE_API FIREEStateLayout
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

	FIREEStateLayout() = default;

	/**
	 * @param[in] InShape Logical tensor shape; first dimension (batch dimension) will be set to 1, as batched state is not currently supported.
	 * @param[in] InMaxSeqLen Maximum sequence length when a sequence dimension is present.
	 */
	explicit FIREEStateLayout(TConstArrayView<int32> InShape, int InMaxSeqLen = 1)
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
			this->StateDimSize = ScholaIREE::GetTotalSize(NonSequenceShapeView);
		}
		else
		{
			this->StateDimSize = ScholaIREE::GetTotalSize(this->Shape);
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
		return ScholaIREE::GetTotalSize(this->Shape);
	}

	/** @return Number of floats in an FIREEStateBuffer: the state sequence plus one pending state vector for state_out. */
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
struct SCHOLAIREE_API FIREEStateBuffer
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

	FIREEStateBuffer() = default;

	/**
	 * @brief Allocates a zeroed state sequence with the given layout
	 * @param[in] Layout The shape of the state tensor
	 */
	explicit FIREEStateBuffer(const FIREEStateLayout& Layout)
		: Shape(Layout.Shape), MaxSeqLen(Layout.MaxSeqLen), SeqDim(Layout.SeqDim), StateDimSize(Layout.StateDimSize)
	{
		this->StateBuffer.Init(0.0f, Layout.GetBufferNumElements());
	}

	/**
	 * @param[in] InShape Logical tensor shape; first dimension (batch dimension) will be set to 1, as batched state is not currently supported.
	 * @param[in] InMaxSeqLen Maximum sequence length when a sequence dimension is present.
	 */
	FIREEStateBuffer(TConstArrayView<int32> InShape, int InMaxSeqLen = 1)
		: FIREEStateBuffer(FIREEStateLayout(InShape, InMaxSeqLen))
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
	 * @brief Creates a tensor binding for the state sequence as input
	 * @return Tensor binding containing the full state sequence, excluding the pending state
	 */
	FIREETensorBinding MakeInputBinding() const
	{
		return { (void*)(StateBuffer.GetData()), static_cast<uint64>(ScholaIREE::GetTotalSize(this->Shape) * sizeof(float)) };
	}

	/**
	 * @brief Creates a tensor binding for the pending state vector as output
	 * @return Tensor binding pointing after the state sequence, so it never overlaps the input binding
	 */
	FIREETensorBinding MakeOutputBinding() const
	{
		return { (void*)(StateBuffer.GetData() + StateBuffer.Num() - StateDimSize), static_cast<uint64>(StateDimSize * sizeof(float)) };
	}
};

struct FIREEBufferVisitor;

/**
 * @brief Base class for all IREE point buffers
 *
 * This is an abstract base class that provides the visitor pattern interface for
 * different types of buffers used to store neural network inputs and outputs.
 */
USTRUCT(BlueprintType)
struct SCHOLAIREE_API FIREEPointBuffer
{
	GENERATED_BODY()

public:
	virtual ~FIREEPointBuffer() = default;

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	virtual void Accept(FIREEBufferVisitor& Visitor) const
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
struct SCHOLAIREE_API FIREEDictBuffer : public FIREEPointBuffer
{
	GENERATED_BODY()

	/** Map of named buffers, each corresponding to a dictionary key */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TMap<FString, TInstancedStruct<FIREEPointBuffer>> Buffers;

	virtual ~FIREEDictBuffer() = default;

	FIREEDictBuffer() = default;

	/**
	 * @brief Constructor that initializes the dictionary buffer with a map of buffers
	 * @param[in] InBuffers Map of buffer names to buffer instances
	 */
	FIREEDictBuffer(const TMap<FString, TInstancedStruct<FIREEPointBuffer>>& InBuffers)
		: Buffers(InBuffers)
	{
	}

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	void Accept(FIREEBufferVisitor& Visitor) const override;
};

/**
 * @brief Buffer for discrete action or observation spaces
 *
 * Stores a single selected discrete value.
 */
USTRUCT(BlueprintType)
struct SCHOLAIREE_API FIREEDiscreteBuffer : public FIREEPointBuffer
{
	GENERATED_BODY()

	/** Buffer storing the discrete value (not one-hot encoded) */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TArray<int64> Buffer;

	/**
	 * @brief Constructor that initializes the discrete buffer, with a fixed size of 1. Use MultiDiscreteBuffer for multiple discrete values.
	 */
	FIREEDiscreteBuffer()
	{
		Buffer.Init(0, 1);
	}

	virtual ~FIREEDiscreteBuffer() = default;

	/**
	 * @brief Creates a tensor binding for this buffer
	 * @return Tensor binding containing the buffer data
	 */
	FIREETensorBinding MakeBinding() const
	{
		return { (void*)Buffer.GetData(), sizeof(int64) };
	}

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	void Accept(FIREEBufferVisitor& Visitor) const override;
};

/**
 * @brief Buffer for multi-discrete action or observation spaces
 *
 * Stores one selected value per multi-discrete dimension, where each dimension can have different numbers of possible values.
 * Used when an agent makes several independent discrete decisions simultaneously.
 */
USTRUCT(BlueprintType)
struct SCHOLAIREE_API FIREEMultiDiscreteBuffer : public FIREEPointBuffer
{
	GENERATED_BODY()

	/** Buffer storing the selected value for each multi-discrete dimension */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TArray<int64> Buffer;

	FIREEMultiDiscreteBuffer() = default;

	/**
	 * @brief Constructor that initializes the multi-discrete buffer with a specific size
	 * @param[in] Size Number of multi-discrete dimensions to store
	 */
	FIREEMultiDiscreteBuffer(int Size)
	{
		Buffer.Init(0, Size);
	}

	virtual ~FIREEMultiDiscreteBuffer() = default;

	/**
	 * @brief Creates a tensor binding for this buffer
	 * @return Tensor binding containing the buffer data
	 */
	FIREETensorBinding MakeBinding() const
	{
		return { (void*)Buffer.GetData(), static_cast<uint64>(Buffer.Num() * sizeof(int64)) };
	}

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	void Accept(FIREEBufferVisitor& Visitor) const override;
};

/**
 * @brief Buffer for multi-binary action or observation spaces
 *
 * Stores multiple independent binary values, where each element represents a binary choice (on/off).
 * Common in scenarios with multiple simultaneous boolean decisions.
 */
USTRUCT(BlueprintType)
struct SCHOLAIREE_API FIREEMultiBinaryBuffer : public FIREEPointBuffer
{
	GENERATED_BODY()

	/** Buffer storing multiple binary values as bools (true or false) */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TArray<bool> Buffer;

public:
	FIREEMultiBinaryBuffer() = default;

	/**
	 * @brief Constructor that initializes the multi-binary buffer with a specific size
	 * @param[in] Size Number of binary values
	 */
	FIREEMultiBinaryBuffer(int Size)
	{
		Buffer.Init(false, Size);
	}

	virtual ~FIREEMultiBinaryBuffer() = default;

	/**
	 * @brief Creates a tensor binding for this buffer
	 * @return Tensor binding containing the buffer data
	 */
	FIREETensorBinding MakeBinding() const
	{
		return { (void*)Buffer.GetData(), static_cast<uint64>(Buffer.Num() * sizeof(bool)) };
	}

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	void Accept(FIREEBufferVisitor& Visitor) const override;
};

/**
 * @brief Buffer for continuous box-bounded action or observation spaces
 *
 * Stores continuous values within bounded ranges, commonly used for continuous control
 * tasks where actions or observations are real-valued vectors with upper and lower bounds.
 */
USTRUCT(BlueprintType)
struct SCHOLAIREE_API FIREEBoxBuffer : public FIREEPointBuffer
{
	GENERATED_BODY()

	/** Buffer storing continuous values */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Policy Data")
	TArray<float> Buffer;

public:
	FIREEBoxBuffer() = default;
	virtual ~FIREEBoxBuffer() = default;

	/**
	 * @brief Constructor that initializes the box buffer with a specific size
	 * @param[in] Size Dimensionality of the continuous space
	 */
	FIREEBoxBuffer(int Size)
	{
		Buffer.Init(0.0f, Size);
	}

	/**
	 * @brief Creates a tensor binding for this buffer
	 * @return Tensor binding containing the buffer data
	 */
	FIREETensorBinding MakeBinding() const
	{
		return { (void*)Buffer.GetData(), static_cast<uint64>(Buffer.Num() * sizeof(float)) };
	}

	/**
	 * @brief Accept method for the visitor pattern
	 * @param[in,out] Visitor The visitor object that will process this buffer
	 */
	void Accept(FIREEBufferVisitor& Visitor) const override;
};
