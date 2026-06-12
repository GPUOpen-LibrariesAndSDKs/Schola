// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Spaces/Space.h"
#include "Spaces/SpaceVisitor.h"
#include "Points/DiscretePoint.h"
#include "DiscreteSpace.generated.h"

/**
 * @struct FDiscreteSpace
 * @brief A struct representing a discrete space of possible observations or actions.
 * 
 * A discrete space represents a single choice from a finite set of options,
 * numbered from 0 to High-1. This is commonly used for discrete action spaces
 * where an agent selects from a fixed set of actions (e.g., move left, move right, jump).
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FDiscreteSpace : public FSpace
{
	GENERATED_BODY()
public:

	/**
	 * @brief The exclusive upper bound (maximum value + 1) for this discrete space.
	 * 
	 * The lower bound is always 0. For example, High=2 gives valid values {0,1}.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (DisplayName = "Maximum Value"))
	int High = 0;

	/**
	 * @brief Constructs an empty DiscreteSpace with High=0.
	 */
	FDiscreteSpace();

	/**
	 * @brief Constructs a DiscreteSpace with a specific upper bound.
	 * @param High The exclusive upper bound (number of discrete options).
	 */
	FDiscreteSpace(int High) : High(High) {};

	/**
	 * @brief Copies the contents of another DiscreteSpace into this one.
	 * @param[in] Other The DiscreteSpace to copy from.
	 */
	void Copy(const FDiscreteSpace& Other);

	/**
	 * @brief Gets the index of the maximum value in an array.
	 * @param[in] Vector The vector to find the maximum value in.
	 * @return The index of the maximum value.
	 */
	int GetMaxValue(const TArray<float>& Vector) const;

	/**
	 * @brief Gets the number of dimensions in this space.
	 * @return Always returns 1 for discrete spaces.
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
	 * @return The value of High.
	 */
	int GetFlattenedSize() const override;

	/**
	 * @brief Checks if this space is empty.
	 * @return True if High is 0, false otherwise.
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