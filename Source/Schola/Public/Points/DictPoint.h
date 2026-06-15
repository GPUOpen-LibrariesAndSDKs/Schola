// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Points/Point.h"
#include "StructUtils/InstancedStruct.h"
#include "Points/PointVisitor.h"
#include "DictPoint.generated.h"



/**
 * @struct FDictPoint
 * @brief A dictionary of named points.
 * 
 * This structure stores multiple points indexed by string keys, allowing
 * for complex hierarchical data structures. Useful for representing
 * observations or actions that have multiple named components.
 */
USTRUCT()
struct FDictPoint : public FPoint
{
	GENERATED_BODY()

	/**
	 * @brief The map of named points in this dictionary.
	 * 
	 * Each entry is a string key paired with a point value.
	 */
	UPROPERTY()
	TMap<FString, TInstancedStruct<FPoint>> Points;

	/**
	 * @brief Constructs an empty dictionary of points.
	 */
	FDictPoint()
	{

	}

	/**
	 * @brief Constructs a dictionary from an existing map of points.
	 * @param[in] InPoints The map of string keys to points to initialize with.
	 */
	FDictPoint(const TMap<FString, TInstancedStruct<FPoint>>& InPoints)
		: Points(InPoints)
	{

	}

	/**
	 * @brief Virtual destructor.
	 */
	virtual ~FDictPoint() {};

	/**
	 * @brief Constructs a dictionary from an initializer list of key-value pairs.
	 * @param[in] InPoints An initializer list of string-point pairs.
	 */
	FDictPoint(std::initializer_list<TPair<FString, TInstancedStruct<FPoint>>> InPoints)
	{
		for (const auto& Pair : InPoints)
		{
			Points.Add(Pair.Key, Pair.Value);
		}
	}

	/**
	 * @brief Accepts a mutable visitor for the visitor pattern.
	 * @param[in,out] Visitor The visitor to accept.
	 */
	void Accept(PointVisitor& Visitor) override
	{
		Visitor(*this);
	}

	/**
	 * @brief Accepts a const visitor for the visitor pattern.
	 * @param[in,out] Visitor The const visitor to accept.
	 */
	void Accept(ConstPointVisitor& Visitor) const override
	{
		Visitor(*this);
	}

	/**
	 * @brief Resets the dictionary, removing all points.
	 */
	void Reset() override
	{
		Points.Empty();
	}

	/**
	 * @brief Converts this dictionary point to a string representation.
	 * @return A multi-line string with one entry per line, for example:
	 * @code
	 * {
	 *     key_a: ...
	 *     key_b: ...
	 * }
	 * @endcode
	 */
	FString ToString() const override
	{
		if (Points.Num() == 0)
		{
			return TEXT("{\n}");
		}
		FString Result = TEXT("{\n");
		for (const auto& Pair : Points)
		{
			Result += FString::Printf(TEXT("    %s: %s,\n"), *Pair.Key, *Pair.Value.Get<FPoint>().ToString());
		}
		Result += TEXT("}");
		return Result;
	}

};