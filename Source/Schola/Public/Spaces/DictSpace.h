// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Points/DictPoint.h"
#include "Spaces/Space.h"
#include "Spaces/SpaceVisitor.h"
#include "DictSpace.generated.h"


/**
 * @struct FDictSpace
 * @brief A struct representing a dictionary of named sub-spaces.
 * 
 * A dictionary space contains multiple named sub-spaces, allowing for complex
 * hierarchical structures. This is useful when observations or actions have
 * multiple distinct components that need to be organized by name, such as
 * "camera", "velocity", "position", etc.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FDictSpace : public FSpace
{
	GENERATED_BODY()

	/**
	 * @brief The map of named sub-spaces in this dictionary.
	 * 
	 * Each entry is a string key paired with a space value, allowing
	 * hierarchical organization of space definitions.
	 */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Space")
	TMap<FString, TInstancedStruct<FSpace>> Spaces;

	/**
	 * @brief Constructs an empty DictSpace.
	 */
	FDictSpace();

	/**
	 * @brief Constructs a DictSpace with the given map of sub-spaces.
	 * @param InSpaces The map of named sub-spaces to initialize this dictionary with.
	 */
	FDictSpace(const TMap<FString, TInstancedStruct<FSpace>>& InSpaces)
		: Spaces(InSpaces)
	{
	}

	/**
	 * @brief Gets the number of sub-spaces in this dictionary.
	 * @return The number of entries in the Spaces map.
	 */
	int Num();
	
	/**
	 * @brief Gets the number of dimensions in this dictionary space.
	 * @return The sum of dimensions of all sub-spaces.
	 */
	int GetNumDimensions() const override;
	
	/**
	 * @brief Checks if this space is empty.
	 * @return True if there are no sub-spaces, false otherwise.
	 */
	bool IsEmpty() const override;
	
	/**
	 * @brief Gets the total flattened size of all sub-spaces.
	 * @return The sum of flattened sizes of all sub-spaces.
	 */
	int GetFlattenedSize() const override;

	/**
	 * @brief Validates that a point conforms to this space.
	 * 
	 * Validates that the point is a DictPoint and that all its sub-points
	 * conform to the corresponding sub-spaces.
	 * 
	 * @param[in] InPoint The point to validate.
	 * @return Validation result indicating success or failure reason.
	 */
	ESpaceValidationResult Validate(const TInstancedStruct<FPoint>& InPoint) const override;

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
