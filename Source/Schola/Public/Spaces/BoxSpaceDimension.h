// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "Spaces/Space.h"
#include "BoxSpaceDimension.generated.h"

/**
 * @struct FBoxSpaceDimension
 * @brief A struct representing a single dimension of a box (continuous) space.
 * 
 * A box space dimension defines a continuous range with optional upper and lower bounds
 * for one dimension of a multi-dimensional continuous space. Unset bounds are unbounded.
 * It provides utilities for normalizing and rescaling values within finite bounds.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FBoxSpaceDimension 
{
	GENERATED_BODY()

	/**
	 * @brief When true, Low is a finite lower bound. When false, this dimension is unbounded below.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (InlineEditConditionToggle))
	bool bHasLow = true;

	/**
	 * @brief The lower bound for this dimension. Ignored when bHasLow is false.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (EditCondition = "bHasLow"))
	float Low = -1.0;

	/**
	 * @brief When true, High is a finite upper bound. When false, this dimension is unbounded above.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (InlineEditConditionToggle))
	bool bHasHigh = true;

	/**
	 * @brief The upper bound for this dimension. Ignored when bHasHigh is false.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (EditCondition = "bHasHigh"))
	float High = 1.0;

	/**
	 * @brief Constructs a BoxSpaceDimension with default bounds [-1, 1].
	 */
	FBoxSpaceDimension();

	/**
	 * @brief Constructs a BoxSpaceDimension with specific finite bounds.
	 * @param[in] Low The lower bound.
	 * @param[in] High The upper bound.
	 */
	FBoxSpaceDimension(float Low, float High);

	/**
	 * @brief Creates a unit dimension in the range [0, 1].
	 * @return A BoxSpaceDimension with bounds [0, 1].
	 */
	static inline FBoxSpaceDimension ZeroOneUnitDimension() { return FBoxSpaceDimension(0, 1); };

	/**
	 * @brief Creates a unit dimension centered at 0 in the range [-0.5, 0.5].
	 * @return A BoxSpaceDimension with bounds [-0.5, 0.5].
	 */
	static inline FBoxSpaceDimension CenteredUnitDimension() { return FBoxSpaceDimension(-0.5, 0.5); };

	/**
	 * @brief Creates a fully unbounded dimension.
	 */
	static inline FBoxSpaceDimension Unbounded()
	{
		FBoxSpaceDimension Dimension;
		Dimension.bHasLow = false;
		Dimension.bHasHigh = false;
		return Dimension;
	}

	/**
	 * @brief Creates a dimension that is bounded below and unbounded above.
	 */
	static inline FBoxSpaceDimension LowerBounded(float InLow)
	{
		FBoxSpaceDimension Dimension;
		Dimension.Low = InLow;
		Dimension.bHasLow = true;
		Dimension.bHasHigh = false;
		return Dimension;
	}

	/**
	 * @brief Creates a dimension that is unbounded below and bounded above.
	 */
	static inline FBoxSpaceDimension UpperBounded(float InHigh)
	{
		FBoxSpaceDimension Dimension;
		Dimension.High = InHigh;
		Dimension.bHasLow = false;
		Dimension.bHasHigh = true;
		return Dimension;
	}

	/** @return True if this dimension has a finite lower bound. */
	inline bool IsLowerBounded() const { return bHasLow; }

	/** @return True if this dimension has a finite upper bound. */
	inline bool IsUpperBounded() const { return bHasHigh; }

	/** @return True if both lower and upper bounds are finite. */
	inline bool IsFullyBounded() const { return bHasLow && bHasHigh; }

	/** @return True if this dimension is unbounded. */
	inline bool IsUnbounded() const { return !bHasLow && !bHasHigh; }

	/**
	 * @brief Validates a scalar against this dimension's optional bounds.
	 * @param[in] Value The value to validate.
	 * @return Success, or OutOfBounds if NaN or outside a finite bound.
	 */
	ESpaceValidationResult ValidateDimension(float Value) const;

	/**
	 * @brief Rescales a normalized [0, 1] value to this dimension's bounds.
	 * @param[in] Value The normalized value to rescale.
	 * @return The rescaled value, or Value unchanged if this dimension is not fully bounded.
	 */
	float RescaleValue(float Value) const;

	/**
	 * @brief Rescales a value from another dimension's bounds to this dimension's bounds.
	 * @param[in] Value The value to rescale.
	 * @param[in] OldHigh The upper bound of the source dimension.
	 * @param[in] OldLow The lower bound of the source dimension.
	 * @return The rescaled value, or Value unchanged if either range is not fully finite.
	 */
	float RescaleValue(float Value, float OldHigh, float OldLow) const;

	/**
	 * @brief Normalizes a value from this dimension's bounds to [0, 1].
	 * @param[in] Value The value to normalize.
	 * @return The normalized value, or Value unchanged if this dimension is not fully bounded.
	 */
	float NormalizeValue(float Value) const;

	/**
	 * @brief Formats this dimension as `[low, high]`, using `-inf`/`inf` when unbounded.
	 */
	FString ToString() const;

	/**
	 * @brief Checks if two BoxSpaceDimensions are equal.
	 * @param[in] Other The dimension to compare to.
	 * @return True if both dimensions have the same bound presence and finite bound values.
	 */
	bool operator==(const FBoxSpaceDimension& Other) const;
};
