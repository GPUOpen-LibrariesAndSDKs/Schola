// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Points/Point.h"
#include "StructUtils/InstancedStruct.h"
#include "Space.generated.h"

class SpaceVisitor;
class ConstSpaceVisitor;

/**
 * @enum ESpaceType
 * @brief Enumeration of concrete Space kinds supported by the Schola blueprint helpers.
 * 
 * This enum defines the different types of spaces that can define valid ranges
 * for observations and actions in reinforcement learning.
 */
 UENUM(BlueprintType)
 enum class ESpaceType : uint8
 {
	 MultiBinary          UMETA(DisplayName="MultiBinary"),        ///< Multi-binary space with boolean values
	 Discrete        UMETA(DisplayName="Discrete"),      ///< Discrete space with integer choices
	 MultiDiscrete   UMETA(DisplayName="MultiDiscrete"), ///< Multi-discrete space with multiple integer choices
	 Box             UMETA(DisplayName="Box"),           ///< Box space with continuous float ranges
	 Dict            UMETA(DisplayName="Dict"),          ///< Dictionary space containing named sub-spaces
	 Text            UMETA(DisplayName="Text")           ///< Text space with variable-length strings
 };

 
/**
 * @enum ESpaceValidationResult
 * @brief Enumeration of possible validation results when testing if a point is in a space.
 * 
 * This enum indicates the outcome of validating whether a point conforms to
 * the constraints defined by a space.
 */
UENUM(BlueprintType)
enum class ESpaceValidationResult : uint8
{
	Success			UMETA(DisplayName = "Success"),               ///< Point is valid for this space
	WrongDimensions UMETA(DisplayName = "Wrong Dimensions"),       ///< Point has incorrect dimensions
	OutOfBounds		UMETA(DisplayName = "Value Out of Bounds"),    ///< Point values are outside valid range
	WrongDataType   UMETA(DisplayName = "Data was of wrong type") ///< Point type doesn't match space type
};

/**
 * @brief Converts a space validation result to a boolean.
 * @param[in] InResult The validation result to convert.
 * @return True if the result is Success, false otherwise.
 */
inline bool ConvertSpaceValidationResultToBool(ESpaceValidationResult InResult)
{
	return InResult == ESpaceValidationResult::Success;
}


/**
 * @struct FSpace
 * @brief Base structure for all space types in the Schola framework.
 * 
 * A space defines the valid structure and range of values for observations
 * or actions in a reinforcement learning system. Spaces specify constraints
 * such as dimensionality, valid value ranges, and data types. This base
 * structure defines the common interface that all concrete space types
 * must implement.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FSpace
{
	GENERATED_BODY()

	/**
	 * @brief Gets the number of dimensions in this space.
	 * @return The number of dimensions.
	 */
	virtual int GetNumDimensions() const PURE_VIRTUAL(FSpace::GetNumDimensions, return 0;);

	/**
	 * @brief Checks if this space is empty.
	 * @return True if the space is empty, false otherwise.
	 */
	virtual bool IsEmpty() const PURE_VIRTUAL(FSpace::IsEmpty, return true;);

	/**
	 * @brief Tests if a point is valid for this space.
	 * 
	 * Validates that the point conforms to all constraints defined by this
	 * space, including correct dimensions, value ranges, and data type.
	 * 
	 * @param[in] InPoint The point to validate.
	 * @return A validation result indicating success or the type of failure.
	 */
	virtual ESpaceValidationResult Validate(const TInstancedStruct<FPoint>& InPoint) const PURE_VIRTUAL(FSpace::Validate, return ESpaceValidationResult::Success;);
	
	/**
	 * @brief Checks if a point is contained in this space.
	 * @param[in] InPoint The point to check.
	 * @return True if the point is contained in this space, false otherwise.
	 */
	bool Contains(const TInstancedStruct<FPoint>& InPoint) const
	{
		return ConvertSpaceValidationResultToBool(Validate(InPoint));
	}
	/**
	 * @brief Gets the size of the flattened representation of this space.
	 * 
	 * Returns the total number of scalar values when all dimensions are
	 * flattened into a single vector.
	 * 
	 * @return The flattened size.
	 */
	virtual int GetFlattenedSize() const PURE_VIRTUAL(FSpace::GetFlattenedSize, return 0;);

	/**
	 * @brief Converts this space to a string representation.
	 * @return A string representation of this space for debugging and logging.
	 */
	virtual FString ToString() const PURE_VIRTUAL(FSpace::ToString, return TEXT("Invalid Space"););

	/**
	 * @brief Virtual destructor.
	 */
	virtual ~FSpace() = default;

	/**
	 * @brief Accepts a mutable visitor for the visitor pattern.
	 * @param[in,out] InVisitor The visitor to accept.
	 */
	virtual void Accept(SpaceVisitor& InVisitor) {};

	/**
	 * @brief Accepts a const visitor for the visitor pattern.
	 * @param[in,out] InVisitor The const visitor to accept.
	 */
	virtual void Accept(ConstSpaceVisitor& InVisitor) const {};
	
};

/**
 * @brief Template specialization providing static struct information for FSpace.
 * @details This specialization enables FSpace to work with TInstancedStruct and other
 * Unreal Engine templated struct systems. It provides compile-time access to the
 * UScriptStruct for FSpace.
 */
template <>
struct TBaseStructure<FSpace>
{
	/**
	 * @brief Gets the UScriptStruct for FSpace.
	 * @return Pointer to FSpace's static struct definition.
	 */
	static SCHOLA_API UScriptStruct* Get()
	{
		return FSpace::StaticStruct();
	}
};