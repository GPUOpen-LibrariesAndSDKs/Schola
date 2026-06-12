// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/DiscreteSpace.h"



FDiscreteSpace::FDiscreteSpace()
	: High(0)
{
}

void FDiscreteSpace::Copy(const FDiscreteSpace& Other)
{
	this->High = Other.High;
}


ESpaceValidationResult FDiscreteSpace::Validate(const TInstancedStruct<FPoint>& InPoint) const
{
	
	const FDiscretePoint* TypedObservation = InPoint.GetPtr<FDiscretePoint>();
	if (!TypedObservation)
	{
		return ESpaceValidationResult::WrongDataType;
	}

	if(TypedObservation->Value < 0 || TypedObservation->Value >= this->High)
	{
		return ESpaceValidationResult::OutOfBounds;
	}

	return ESpaceValidationResult::Success;
}

int FDiscreteSpace::GetNumDimensions() const
{
	return 1;
}

int FDiscreteSpace::GetMaxValue(const TArray<float>& Vector) const
{
	float CurrMax = Vector[0];
	int	  Index = 0;
	int	  CurrIndex = 0;
	for (const float& Value : Vector)
	{
		if (Value > CurrMax)
		{
			CurrMax = Value;
			Index = CurrIndex;
		}
		CurrIndex += 1;
	}
	return Index;
}

int FDiscreteSpace::GetFlattenedSize() const
{
	return this->High;
}

bool FDiscreteSpace::IsEmpty() const
{
	return this->High == 0;
}

FString FDiscreteSpace::ToString() const
{
	return FString::Printf(TEXT("DiscreteSpace(High=%d)"), High);
}