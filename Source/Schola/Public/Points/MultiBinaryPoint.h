// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Points/Point.h"
#include "Points/PointVisitor.h"
#include "MultiBinaryPoint.generated.h"

/**
 * @struct FMultiBinaryPoint
 * @brief A point in a multi-binary space with multiple boolean values.
 * 
 * Multi-binary points represent multiple independent binary choices,
 * where each dimension is a true/false value. Commonly used for
 * representing multiple on/off states or binary flags.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FMultiBinaryPoint : public FPoint
{
	GENERATED_BODY()
	
	/**
	 * @brief The boolean values of this point.
	 */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "Point")
	TArray<bool> Values;
	
	/**
	 * @brief Constructs an empty MultiBinaryPoint.
	 */
	FMultiBinaryPoint()
	{
	}

	/**
	 * @brief Constructs a MultiBinaryPoint from an array of booleans.
	 * @param[in] InitialValues An array of bools to initialize the point with.
	 */
	FMultiBinaryPoint(const TArray<bool>& InitialValues)
		: Values(InitialValues)
	{
	}

	/**
	 * @brief Constructs a MultiBinaryPoint from an initializer list.
	 * @param[in] InValues An initializer list of bools.
	 */
	FMultiBinaryPoint(std::initializer_list<bool> InValues)
		: Values(InValues)
	{
	}

	/**
	 * @brief Constructs a MultiBinaryPoint from a raw array of bools.
	 * @param[in] Data The raw array of bools, as a const pointer.
	 * @param[in] Num The size of the array.
	 */
	FMultiBinaryPoint(const bool* Data, int Num)
		: Values(Data, Num)
	{
	}

	/**
	 * @brief Virtual destructor.
	 */
	virtual ~FMultiBinaryPoint() {};
	
	/**
	 * @brief Gets the boolean value at the given index.
	 * @param[in] Index The dimension to get the value at.
	 * @return The boolean value at the given index.
	 */
	bool operator[](int Index) const
	{
		return this->Values[Index];
	}

	/**
	 * @brief Adds a boolean value to the point.
	 * @param[in] Value The boolean value to add.
	 */
	void Add(bool Value)
	{
		this->Values.Add(Value);
	}

	/**
	 * @brief Resets the values of the MultiBinaryPoint.
	 * 
	 * @note This doesn't reset the size of the array in memory, so subsequent calls
	 * to Add may not reallocate memory.
	 */
	void Reset() override
	{
		this->Values.Reset(Values.Num());
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
	 * @brief Converts this point to a string representation.
	 * @return Bracketed list of true/false, e.g. `[true, false]`.
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
			Result += this->Values[i] ? TEXT("true") : TEXT("false");
		}
		Result += TEXT("]");
		return Result;
	}
};