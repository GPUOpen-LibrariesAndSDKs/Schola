// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/BoxSpaceDimension.h"


FBoxSpaceDimension::FBoxSpaceDimension()
{
}

FBoxSpaceDimension::FBoxSpaceDimension(float Low, float High)
	: bHasLow(true), Low(Low), bHasHigh(true), High(High)
{
}

ESpaceValidationResult FBoxSpaceDimension::ValidateDimension(float Value) const
{
	if (FMath::IsNaN(Value))
	{
		return ESpaceValidationResult::OutOfBounds;
	}
	if (bHasLow && Value < Low)
	{
		return ESpaceValidationResult::OutOfBounds;
	}
	if (bHasHigh && Value > High)
	{
		return ESpaceValidationResult::OutOfBounds;
	}
	return ESpaceValidationResult::Success;
}

float FBoxSpaceDimension::RescaleValue(float NormalizedValue) const
{
	if (!IsFullyBounded())
	{
		return NormalizedValue;
	}
	return (NormalizedValue * (this->High - this->Low)) + this->Low;
}

float FBoxSpaceDimension::NormalizeValue(float Value) const
{
	if (!IsFullyBounded())
	{
		return Value;
	}
	// Convert a value from the range of this dimension to [0,1]
	return (Value - this->Low) / (this->High - this->Low);
}

float FBoxSpaceDimension::RescaleValue(float Value, float OldHigh, float OldLow) const
{
	if (!IsFullyBounded())
	{
		return Value;
	}

	// Normalize the value to be between [0,1] based on it's previous range
	float NormalizedValue = (Value - OldLow) / (OldHigh - OldLow);

	// Now blow it back up to the range of this dimension
	return this->RescaleValue(NormalizedValue);
}

FString FBoxSpaceDimension::ToString() const
{
	const FString LowStr = bHasLow ? FString::Printf(TEXT("%.6g"), Low) : TEXT("-inf");
	const FString HighStr = bHasHigh ? FString::Printf(TEXT("%.6g"), High) : TEXT("inf");
	return FString::Printf(TEXT("[%s, %s]"), *LowStr, *HighStr);
}

bool FBoxSpaceDimension::operator==(const FBoxSpaceDimension& Other) const
{
	if (bHasLow != Other.bHasLow || bHasHigh != Other.bHasHigh)
	{
		return false;
	}
	if (bHasLow && Low != Other.Low)
	{
		return false;
	}
	if (bHasHigh && High != Other.High)
	{
		return false;
	}
	return true;
}
