// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StructUtils/InstancedStruct.h"
#include "Point.generated.h"

class PointVisitor;
class ConstPointVisitor;


/**
 * @struct FPoint
 * @brief Base structure for all point types in the Schola framework.
 * 
 * A point is a data point that can represent observations or actions
 * in a reinforcement learning system. This base structure defines the
 * common interface that all concrete point types must implement.
 * Points can be passed to models for inference or used to communicate
 * state information.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FPoint
{
	GENERATED_BODY()

	/**
	 * @brief Virtual destructor for proper cleanup of derived types.
	 */
	virtual ~FPoint() = default;

	/**
	 * @brief Accepts a mutable visitor for the visitor pattern.
	 * @param[in,out] Visitor The visitor to accept.
	 */
	virtual void Accept(PointVisitor& Visitor) {};

	/**
	 * @brief Accepts a const visitor for the visitor pattern.
	 * @param[in,out] Visitor The const visitor to accept.
	 */
	virtual void Accept(ConstPointVisitor& Visitor) const {};

	/**
	 * @brief Resets the point to its default state.
	 */
	virtual void Reset() {};

	/**
	 * @brief Converts this point to a string representation.
	 * @return A string representation of this point for debugging and logging.
	 */
	virtual FString ToString() const PURE_VIRTUAL(FPointBase::ToString, return TEXT("Invalid Point"););

};

/**
 * @brief Template specialization providing static struct information for FPoint.
 * @details This specialization enables FPoint to work with TInstancedStruct and other
 * Unreal Engine templated struct systems. It provides compile-time access to the
 * UScriptStruct for FPoint.
 */
template <>
struct TBaseStructure<FPoint>
{
	/**
	 * @brief Gets the UScriptStruct for FPoint.
	 * @return Pointer to FPoint's static struct definition.
	 */
	static SCHOLA_API UScriptStruct* Get()
	{
		return FPoint::StaticStruct();
	}
};

/**
 * @enum EPointType
 * @brief Enumeration of concrete Point types supported by the Schola blueprint helpers.
 * 
 * This enum defines the different types of points that can be created and manipulated
 * in the Schola framework.
 */
 UENUM(BlueprintType)
 enum class EPointType : uint8
 {
	 MultiBinary          UMETA(DisplayName="MultiBinary"),        ///< Multi-binary point with boolean values
	 Discrete        UMETA(DisplayName="Discrete"),      ///< Discrete point with a single integer value
	 MultiDiscrete   UMETA(DisplayName="MultiDiscrete"), ///< Multi-discrete point with multiple integer values
	 Box             UMETA(DisplayName="Box"),           ///< Box point with continuous float values
	 Dict            UMETA(DisplayName="Dict"),          ///< Dictionary point containing named sub-points
	 Text            UMETA(DisplayName="Text")           ///< Text point holding a string value
 };
 