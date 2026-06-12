// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Spaces/Space.h"
#include "Points/BoxPoint.h"
#include "Spaces/SpaceVisitor.h"
#include "Spaces/BoxSpaceDimension.h"
#include "BoxSpace.generated.h"


/**
 * @struct FBoxSpace
 * @brief A struct representing a box (continuous) space of possible observations or actions.
 * 
 * A BoxSpace is a Cartesian product of BoxSpaceDimensions, where each dimension
 * defines a continuous range with upper and lower bounds. This is commonly used
 * for continuous control problems where actions or observations are real-valued
 * vectors within specified ranges.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FBoxSpace : public FSpace
{
	GENERATED_BODY()

public:
	
	/** The dimensions of this BoxSpace. Each dimension defines a continuous range with low and high bounds. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (TitleProperty = "[{Low}, {High}]"))
	TArray<FBoxSpaceDimension> Dimensions = TArray<FBoxSpaceDimension>();
	
	/** The shape of the Box Space. If empty, uses a 1D array for the Dimensions. Otherwise defines the multi-dimensional structure of the space. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (TitleProperty = "Shape"))
	TArray<int> Shape = TArray<int>();
	
	/**
	 * @brief Constructs an empty BoxSpace with no dimensions.
	 */
	FBoxSpace();

	/**
	 * @brief Constructs a BoxSpace with the given bounds arrays.
	 * @param[in] Low Array representing the lower bound of each dimension.
	 * @param[in] High Array representing the upper bound of each dimension.
	 * @param[in] Shape Optional dimensional shape for multi-dimensional structure.
	 * @note Low and High must have the same length.
	 */
	FBoxSpace(const TArray<float>& Low, const TArray<float>& High, const TArray<int>& Shape = TArray<int>());

	/**
	 * @brief Constructs a BoxSpace from an array of BoxSpaceDimensions.
	 * @param[in] Dimensions Array of BoxSpaceDimensions defining the space bounds.
	 * @param[in] Shape Optional dimensional shape for multi-dimensional structure.
	 */
	FBoxSpace(const TArray<FBoxSpaceDimension>& Dimensions, const TArray<int>& Shape = TArray<int>());

	/**
	 * @brief Constructs a BoxSpace from initializer lists.
	 * @param[in] Low Initializer list of lower bounds.
	 * @param[in] High Initializer list of upper bounds.
	 * @param[in] Shape Optional initializer list defining the dimensional shape.
	 */
	FBoxSpace(std::initializer_list<float> Low, std::initializer_list<float> High, std::initializer_list<int> Shape = std::initializer_list<int>());

	/**
	 * @brief Constructs a BoxSpace with a specific shape and uninitialized dimensions.
	 * @param[in] Shape The dimensional shape to preallocate.
	 */
	FBoxSpace(const TArray<int>& Shape);

	/**
	 * @brief Copies the contents of another BoxSpace into this one.
	 * @param[in] Other The BoxSpace to copy from.
	 */
	void Copy(const FBoxSpace& Other);

	/**
	 * @brief Gets a normalized version of this BoxSpace with all dimensions in [0, 1].
	 * @return A new BoxSpace with all dimensions normalized.
	 */
	FBoxSpace GetNormalizedObservationSpace() const;

	/**
	 * @brief Virtual destructor.
	 */
	virtual ~FBoxSpace();

	/**
	 * @brief Adds a dimension to this BoxSpace.
	 * @param[in] Low The lower bound of the new dimension.
	 * @param[in] High The upper bound of the new dimension.
	 */
	void Add(float Low, float High);

	/**
	 * @brief Adds a dimension to this BoxSpace.
	 * @param[in] Dimension The BoxSpaceDimension to add.
	 */
	void Add(const FBoxSpaceDimension& Dimension);

	/**
	 * @brief Gets the number of dimensions in this space.
	 * @return The number of dimensions.
	 */
	int GetNumDimensions() const override;
	
	/**
	 * @brief Gets the flattened size of this space.
	 * @return The total number of scalar values.
	 */
	int GetFlattenedSize() const override;
	
	/**
	 * @brief Checks if this space is empty.
	 * @return True if empty, false otherwise.
	 */
	bool IsEmpty() const override;

	/**
	 * @brief Validates that a point conforms to this space.
	 * @param[in] Point The point to validate.
	 * @return Validation result indicating success or failure reason.
	 */
	ESpaceValidationResult Validate(const TInstancedStruct<FPoint>& Point) const override;

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

	/**
	 * @brief Normalizes an observation in this space to the range [0, 1].
	 * @param[in] Observation The observation to normalize.
	 * @return A new BoxPoint with the normalized observation.
	 */
	FBoxPoint NormalizeObservation(const FBoxPoint& Observation) const;
};

