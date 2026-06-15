// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Spaces/Space.h"
#include "Spaces/SpaceVisitor.h"
#include "Points/TextPoint.h"
#include "TextSpace.generated.h"

/**
 * @namespace ScholaTextCharsets
 * @brief Common character-set presets for use with FTextSpace::Charset.
 */
namespace ScholaTextCharsets
{
	/** ASCII digits 0-9. */
	inline constexpr const TCHAR* Numeric = TEXT("0123456789");
	/** ASCII letters a-z and A-Z. */
	inline constexpr const TCHAR* Alphabetic = TEXT("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
	/** ASCII digits and letters (matches Gymnasium's default alphanumeric set). */
	inline constexpr const TCHAR* Alphanumeric = TEXT("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
}

/**
 * @enum ETextCharsetPreset
 * @brief Selectable character-set presets for building an FTextSpace charset.
 */
UENUM(BlueprintType)
enum class ETextCharsetPreset : uint8
{
	/** ASCII digits 0-9. */
	Numeric UMETA(DisplayName = "Numeric"),
	/** ASCII letters a-z and A-Z. */
	Alphabetic UMETA(DisplayName = "Alphabetic"),
	/** ASCII digits and letters (Gymnasium's default). */
	Alphanumeric UMETA(DisplayName = "Alphanumeric"),
};

/**
 * @struct FTextSpace
 * @brief A struct representing a space of variable-length strings.
 * 
 * A text space represents a string whose length is bounded by [MinLength, MaxLength]
 * and whose characters are restricted to a fixed character set (defaulting to alphanumeric).
 * This mirrors the gymnasium Text space and is commonly used for text-based observations or actions.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FTextSpace : public FSpace
{
	GENERATED_BODY()
public:

	/**
	 * @brief The maximum allowed length (inclusive) of strings in this space.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (DisplayName = "Maximum Length", ClampMin = "0"))
	int MaxLength = 0;

	/**
	 * @brief The minimum allowed length (inclusive) of strings in this space.
	 *
	 * Defaults to 0 so an empty (MaxLength == 0) space stays coherent (MinLength <= MaxLength).
	 * The length constructors and the Make Text Space node default this to 1 to match Gymnasium.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (DisplayName = "Minimum Length", ClampMin = "0"))
	int MinLength = 0;

	/**
	 * @brief The set of characters allowed in strings in this space.
	 *
	 * The charset is taken literally: it is the exact set of permitted characters and
	 * maps directly to Gymnasium's character_set. An empty charset is the empty set.
	 * Defaults to Gymnasium's alphanumeric set so a default-constructed space matches
	 * Gymnasium's Text default.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (DisplayName = "Character Set"))
	FString Charset = ScholaTextCharsets::Alphanumeric;

	/**
	 * @brief Constructs an empty TextSpace with MaxLength=0.
	 */
	FTextSpace();

	/**
	 * @brief Constructs a TextSpace with a specific maximum length.
	 * @param MaxLength The maximum allowed string length.
	 *
	 * MinLength defaults to 1 (matching Gymnasium) when MaxLength is positive, and 0 otherwise
	 * so the space stays coherent (MinLength <= MaxLength).
	 */
	FTextSpace(int MaxLength) : MaxLength(MaxLength), MinLength(MaxLength > 0 ? 1 : 0) {};

	/**
	 * @brief Constructs a TextSpace from its length bounds and character set.
	 * @param MaxLength The maximum allowed string length.
	 * @param MinLength The minimum allowed string length.
	 * @param Charset The set of allowed characters, taken literally. An empty string denotes the empty set.
	 */
	FTextSpace(int MaxLength, int MinLength, const FString& Charset)
		: MaxLength(MaxLength), MinLength(MinLength), Charset(Charset)
	{
		verifyf(MinLength >= 0, TEXT("FTextSpace MinLength (%d) must be non-negative"), MinLength);
		verifyf(MinLength <= MaxLength, TEXT("FTextSpace MinLength (%d) must be <= MaxLength (%d)"), MinLength, MaxLength);
	};

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
	 * @param[in,out] InVisitor The visitor to accept.
	 */
	virtual void Accept(ConstSpaceVisitor& InVisitor) const
	{
		InVisitor(*this);
	}
};
