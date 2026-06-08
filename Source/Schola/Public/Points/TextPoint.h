// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Points/Point.h"
#include "Points/PointVisitor.h"
#include "TextPoint.generated.h"

/**
 * @struct FTextPoint
 * @brief A point in a text space holding a single string value.
 * 
 * Text points represent a string sampled from a text space, constrained by
 * the space's length bounds and optional character set. Commonly used for
 * text-based observations or actions.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FTextPoint : public FPoint
{
	GENERATED_BODY()

	/**
	 * @brief The string value of this point.
	 */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "Point")
	FString Value = TEXT("");

	/**
	 * @brief Constructs an empty TextPoint with an empty string.
	 */
	FTextPoint()
		: Value(TEXT(""))
	{
	}

	/**
	 * @brief Constructs a TextPoint with a specific string value.
	 * @param[in] Value The string value to initialize the point with.
	 */
	FTextPoint(const FString& Value)
		: Value(Value)
	{
	}

	/**
	 * @brief Virtual destructor.
	 */
	virtual ~FTextPoint()
	{
	}

	/**
	 * @brief Accepts a mutable visitor for the visitor pattern.
	 * @param[in,out] Visitor The visitor to accept.
	 */
	void Accept(PointVisitor& Visitor) override;

	/**
	 * @brief Accepts a const visitor for the visitor pattern.
	 * @param[in,out] Visitor The const visitor to accept.
	 */
	void Accept(ConstPointVisitor& Visitor) const override;

	/**
	 * @brief Resets the value of the TextPoint to an empty string.
	 */
	void Reset() override
	{
		this->Value.Empty();
	};

	/**
	 * @brief Converts this point to a string representation.
	 * @return The string value of this point.
	 */
	FString ToString() const override
	{
		return this->Value;
	}
};
