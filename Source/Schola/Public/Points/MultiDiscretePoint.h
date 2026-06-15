// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Points/Point.h"
#include "Points/PointVisitor.h"
#include "MultiDiscretePoint.generated.h"

/**
 * @struct FMultiDiscretePoint
 * @brief A point in a multi-discrete space with multiple integer values.
 * 
 * Multi-discrete points represent multiple independent discrete choices,
 * where each dimension can have a different discrete value. Useful for
 * representing multiple independent categorical selections.
 */
USTRUCT()
struct SCHOLA_API FMultiDiscretePoint : public FPoint
{
	GENERATED_BODY()
	
	/**
	 * @brief The integer values of this point.
	 */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "Point")
	TArray<int> Values;

	/**
	 * @brief Constructs an empty MultiDiscretePoint.
	 */
	FMultiDiscretePoint()
	{
	}

	/**
	 * @brief Constructs a MultiDiscretePoint from a raw array of integers.
	 * @param[in] Data The raw array of ints, as a const pointer.
	 * @param[in] Num The size of the array.
	 */
	FMultiDiscretePoint(const int* Data, int Num)
		: Values(Data, Num)
	{
	}

	/**
	 * @brief Constructs a MultiDiscretePoint from an array of integers.
	 * @param[in] InitialValues An array of ints to initialize the point with.
	 */
	FMultiDiscretePoint(const TArray<int>& InitialValues)
		: Values(InitialValues)
	{
	}

	/**
	 * @brief Constructs a MultiDiscretePoint from an initializer list of integers.
	 * @param[in] InitialValues An initializer list of ints to initialize the point with.
	 */
	FMultiDiscretePoint(std::initializer_list<int> InitialValues)
		: Values(InitialValues)
	{
	}

	/**
	 * @brief Virtual destructor.
	 */
	virtual ~FMultiDiscretePoint()
	{
	}

	/**
	 * @brief Accepts a mutable visitor for the visitor pattern.
	 * @param[in,out] Visitor The visitor to accept.
	 */
	void Accept(PointVisitor& Visitor) override;

	/**
	 * @brief Accepts a const visitor for the visitor pattern.
	 * @param[in,out] Visitor The const visitor to accept.
	 */
	void Accept(ConstPointVisitor& Visitor) const override;

	/**
	 * @brief Gets the integer value at the given index.
	 * @param[in] Index The dimension to get the value at.
	 * @return The integer value at the given index.
	 */
	int operator[](int Index) const
	{
		return this->Values[Index];
	}

	/**
	 * @brief Resets the values of the MultiDiscretePoint.
	 * 
	 * @note This doesn't reset the size of the array, so subsequent calls
	 * to Add will not reallocate memory.
	 */
	void Reset() override
	{
		this->Values.Reset(this->Values.Num());
	};

	/**
	 * @brief Adds an integer value to the point.
	 * @param[in] Value The integer value to add.
	 */
	void Add(int Value)
	{
		this->Values.Add(Value);
	}

	/**
	 * @brief Converts this point to a string representation.
	 * @return Bracketed list of integers, e.g. `[0, 1, 2]`.
	 */
	FString ToString() const override
	{
		FString Result = TEXT("[");
		for (int i = 0; i < this->Values.Num(); ++i)
		{
			if (i > 0)
			{
				Result += TEXT(", ");
			}
			Result += FString::Printf(TEXT("%d"), this->Values[i]);
		}
		Result += TEXT("]");
		return Result;
	}
};
