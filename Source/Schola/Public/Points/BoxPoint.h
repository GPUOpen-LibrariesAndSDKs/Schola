// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Points/Point.h"
#include "Points/PointVisitor.h"
#include "BoxPoint.generated.h"

/**
 * @struct FBoxPoint
 * @brief A point in a box (continuous) space.
 * 
 * Conceptually represents a floating point vector with continuous values.
 * Box points are used to represent continuous observations or actions
 * in a bounded multi-dimensional space.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FBoxPoint : public FPoint
{
	GENERATED_BODY()

	/** The float values of this point. */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "Point")
	TArray<float> Values;

	/**
	 * @brief The dimensional shape of this point.
	 * 
	 * If this is empty, the point is 1D with length Values.Num().
	 * Otherwise, defines the multi-dimensional structure of the values.
	 */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "Point")
	TArray<int> Shape;

	/**
	 * @brief Constructs an empty BoxPoint with no values.
	 */
	FBoxPoint()
	{
	}

	/**
	 * @brief Constructs a BoxPoint from an array of float values.
	 * @param[in] InValues The array of float values to initialize the point with.
	 */
	FBoxPoint(const TArray<float>& InValues)
		: Values(InValues), Shape({InValues.Num()})
	{

	}

	/**
	 * @brief Constructs a BoxPoint from an initializer list of float values.
	 * @param[in] InValues The initializer list of float values.
	 */
	FBoxPoint(std::initializer_list<float> InValues)
		: Values(InValues), Shape({static_cast<int>(InValues.size())})
	{
	}

	/**
	 * @brief Constructs a BoxPoint with specific values and shape.
	 * @param[in] InValues The array of float values.
	 * @param[in] InShape The dimensional shape for the point.
	 */
	FBoxPoint(const TArray<float>& InValues, const TArray<int>& InShape)
		: Values(InValues), Shape(InShape)
	{
	}

	/**
	 * @brief Construct a BoxPoint from a raw array of floats
	 * @param[in] Data The raw array of floats, as a const ptr
	 * @param[in] Num The size of the array
	 */
	FBoxPoint(const float* Data, int Num)
		: Values(Data, Num), Shape({ Num })
	{
	}

	/**
	 * @brief Constructs a preallocated BoxPoint with no initial values.
	 * @param[in] NumDims The amount of memory to preallocate for values.
	 */
	FBoxPoint(int NumDims) : Shape({ NumDims })
	{
		Values.Reserve(NumDims);
	}

	/**
	 * @brief Gets the value at the given index or dimension.
	 * @param[in] Index The dimension to get the value at.
	 * @return The float value at the given index.
	 */
	float operator[](int Index) const
	{
		return this->Values[Index];
	}

	/**
	 * @brief Virtual destructor.
	 */
	virtual ~FBoxPoint()
	{
	}

	/**
	 * @brief Adds a value to the BoxPoint, effectively adding a new dimension.
	 * @param[in] Value The float value to add.
	 */
	void Add(float Value)
	{
		this->Values.Add(Value);
	}

	/**
	 * @brief Resets the values of the BoxPoint, clearing the current values.
	 * 
	 * @note This doesn't reset the size of the array, so subsequent calls
	 * to Add will not reallocate memory.
	 */
	void Reset() override
	{
		this->Values.Reset(this->Values.Num());
	};

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
	 * @return Bracketed list of floats, e.g. `[0, 1.5]`.
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
			Result += FString::SanitizeFloat(this->Values[i]);
		}
		Result += TEXT("]");
		return Result;
	}
};
