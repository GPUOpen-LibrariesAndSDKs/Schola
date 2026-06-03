// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Spaces/Space.h"
#include "Spaces/SpaceVisitor.h"
#include "Points/TextPoint.h"
#include "TextSpace.generated.h"

/**
 * @struct FTextSpace
 * @brief A struct representing a space of variable-length strings.
 * 
 * A text space represents a string whose length is bounded by [MinLength, MaxLength]
 * and whose characters are optionally restricted to a fixed character set. This mirrors
 * the gymnasium Text space and is commonly used for text-based observations or actions.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FTextSpace : public FSpace
{
	GENERATED_BODY()
public:

	/**
	 * @brief The maximum allowed length (inclusive) of strings in this space.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (DisplayName = "Maximum Length"))
	int MaxLength = 0;

	/**
	 * @brief Whether a minimum length constraint is set for this space.
	 * 
	 * When false, MinLength is ignored and the effective minimum length is 0.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (DisplayName = "Has Minimum Length", InlineEditConditionToggle))
	bool bHasMinLength = false;

	/**
	 * @brief The minimum allowed length (inclusive) of strings in this space.
	 * 
	 * Only enforced when bHasMinLength is true.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (DisplayName = "Minimum Length", EditCondition = "bHasMinLength"))
	int MinLength = 0;

	/**
	 * @brief The set of characters allowed in strings in this space.
	 * 
	 * An empty charset applies no restriction in C++, but maps to Gymnasium's
	 * default (alphanumeric) set when exchanged with Python.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (DisplayName = "Character Set"))
	FString Charset = TEXT("");

	/**
	 * @brief Constructs an empty TextSpace with MaxLength=0.
	 */
	FTextSpace();

	/**
	 * @brief Constructs a TextSpace with a specific maximum length.
	 * @param MaxLength The maximum allowed string length.
	 */
	FTextSpace(int MaxLength) : MaxLength(MaxLength) {};

	/**
	 * @brief Constructs a TextSpace from its length bounds and character set.
	 * @param MaxLength The maximum allowed string length.
	 * @param bHasMinLength Whether a minimum length constraint is enforced.
	 * @param MinLength The minimum allowed string length. Only enforced when bHasMinLength is true.
	 * @param Charset The set of allowed characters. An empty string applies no restriction.
	 */
	FTextSpace(int MaxLength, bool bHasMinLength, int MinLength, const FString& Charset)
		: MaxLength(MaxLength), bHasMinLength(bHasMinLength), MinLength(MinLength), Charset(Charset) {};

	/**
	 * @brief Copies the contents of another TextSpace into this one.
	 * @param[in] Other The TextSpace to copy from.
	 */
	void Copy(const FTextSpace& Other);

	/**
	 * @brief Gets the number of dimensions in this space.
	 * @return Always returns 1 for text spaces.
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
	 * @return The value of MaxLength.
	 */
	int GetFlattenedSize() const override;

	/**
	 * @brief Checks if this space is empty.
	 * @return True if MaxLength is 0, false otherwise.
	 */
	bool IsEmpty() const override;

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
	 * @param[in,out] InVisitor The visitor to accept.
	 */
	virtual void Accept(ConstSpaceVisitor& InVisitor) const
	{
		InVisitor(*this);
	}
};
