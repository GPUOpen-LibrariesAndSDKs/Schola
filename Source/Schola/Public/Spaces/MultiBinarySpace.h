// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Spaces/Space.h"
#include "Spaces/SpaceVisitor.h"
#include "Points/MultiBinaryPoint.h"
#include "MultiBinarySpace.generated.h"


/**
 * @struct FMultiBinarySpace
 * @brief A struct representing a multi-binary space (boolean vector) of possible observations or actions.
 * 
 * A multi-binary space represents multiple independent binary (true/false) values,
 * where each dimension is independent. This is commonly used for representing
 * multiple on/off states or binary flags in observations or actions.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FMultiBinarySpace : public FSpace
{
    GENERATED_BODY()

    /** The number of binary dimensions in this space. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition")
    int Shape = 0;

    /**
     * @brief Constructs an empty MultiBinarySpace with Shape=0.
     */
    FMultiBinarySpace();

    /**
     * @brief Constructs a MultiBinarySpace with a specific number of dimensions.
     * @param[in] InShape The number of binary dimensions.
     */
    FMultiBinarySpace(int InShape);

    /**
     * @brief Merges another MultiBinarySpace into this one.
     * @param[in] Other The space to merge.
     */
    void Merge(const FMultiBinarySpace& Other);

    /**
     * @brief Copies the contents of another MultiBinarySpace into this one.
     * @param[in] Other The space to copy from.
     */
    void Copy(const FMultiBinarySpace& Other);

    /**
     * @brief Gets the number of dimensions in this space.
     * @return The value of Shape.
     */
    virtual int GetNumDimensions() const override;
    
    /**
     * @brief Checks if this space is empty.
     * @return True if Shape is 0, false otherwise.
     */
    virtual bool IsEmpty() const override;
    
    /**
     * @brief Validates that a point conforms to this space.
     * @param[in] InPoint The point to validate.
     * @return Validation result indicating success or failure reason.
     */
    virtual ESpaceValidationResult Validate(const TInstancedStruct<FPoint>& InPoint) const override;
    
    /**
     * @brief Gets the flattened size of this space.
     * @return The value of Shape.
     */
    virtual int GetFlattenedSize() const override;

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