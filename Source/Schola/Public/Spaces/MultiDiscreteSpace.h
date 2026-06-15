// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Spaces/Space.h"
#include "Spaces/SpaceVisitor.h"
#include "Points/MultiDiscretePoint.h"
#include "MultiDiscreteSpace.generated.h"


/**
 * @struct FMultiDiscreteSpace
 * @brief A struct representing a multi-discrete space (vector of integers) of possible observations or actions.
 * 
 * A multi-discrete space represents multiple independent discrete choices,
 * where each dimension can have a different number of options. This is useful
 * for representing multiple independent categorical selections, where each
 * category may have a different number of choices.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FMultiDiscreteSpace : public FSpace
{
	GENERATED_BODY()
public:
	/**
	 * @brief Array of exclusive upper bounds (maximum value + 1) for each dimension.
	 * 
	 * The lower bound for each dimension is always 0. For example, High[i]=2 gives
	 * valid values {0,1} for dimension i.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (DisplayName = "Maximum Values"))
	TArray<int> High = TArray<int>();

	/**
	 * @brief Constructs an empty MultiDiscreteSpace.
	 */
	FMultiDiscreteSpace();

	/**
	 * @brief Constructs a MultiDiscreteSpace from an array of upper bounds.
	 * @param[in] High Array of exclusive upper bounds for each dimension.
	 */
	FMultiDiscreteSpace(const TArray<int>& High);

	/**
	 * @brief Constructs a MultiDiscreteSpace from a raw array.
	 * @param[in] Data Raw array of upper bounds.
	 * @param[in] Size Size of the array.
	 */
	FMultiDiscreteSpace(const int* Data, int Size)
		: High(Data, Size) {};

	/**
	 * @brief Copy constructor.
	 * @param[in] Other The MultiDiscreteSpace to copy from.
	 */
	FMultiDiscreteSpace(FMultiDiscreteSpace & Other);

	/**
	 * @brief Merges another MultiDiscreteSpace into this one.
	 * @param[in] Other The space to merge.
	 */
	void Merge(const FMultiDiscreteSpace& Other);
	
	/**
	 * @brief Adds a dimension to this MultiDiscreteSpace.
	 * @param[in] DimSize The exclusive upper bound for the new dimension.
	 */
	void Add(int DimSize);

	/**
	 * @brief Virtual destructor.
	 */
	virtual ~FMultiDiscreteSpace();

	/**
	 * @brief Gets the index of the maximum value in an array.
	 * @param[in] Vector The vector to find the maximum value in.
	 * @return The index of the maximum value.
	 */
	int GetMaxValue(const TArray<float>& Vector) const;

	/**
	 * @brief Gets the number of dimensions in this space.
	 * @return The number of dimensions in the High array.
	 */
	int GetNumDimensions() const override;

	/**
	 * @brief Validates that a point conforms to this space.
	 * @param[in] InPoint The point to validate.
	 * @return Validation result indicating success or failure reason.
	 */
	ESpaceValidationResult Validate(const TInstancedStruct<FPoint>& InPoint) const override;

	/**
	 * @brief Gets the flattened size of this space.
	 * @return The sum of all High values.
	 */
	int GetFlattenedSize() const override;

	/**
	 * @brief Checks if this space is empty.
	 * @return True if High array is empty, false otherwise.
	 */
	bool IsEmpty() const override;

	/**
	 * @brief Converts this space to a string representation.
	 */
	FString ToString() const override;

	/**
	 * @brief Accepts a mutable visitor for the visitor pattern.
	 * @param[in,out] InVisitor The visitor to accept.
	 */
	virtual void Accept(SpaceVisitor& InVisitor)
	{
		InVisitor(*this);
	}

	/**
	 * @brief Accepts a const visitor for the visitor pattern.
	 * @param[in,out] InVisitor The const visitor to accept.
	 */
	virtual void Accept(ConstSpaceVisitor& InVisitor) const
	{
		InVisitor(*this);
	}
};