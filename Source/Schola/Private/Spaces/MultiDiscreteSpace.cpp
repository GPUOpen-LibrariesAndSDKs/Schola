// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/MultiDiscreteSpace.h"



FMultiDiscreteSpace::FMultiDiscreteSpace()
{
}

FMultiDiscreteSpace::FMultiDiscreteSpace(const TArray<int>& High)
{
	this->High = High;
}

FMultiDiscreteSpace::FMultiDiscreteSpace(FMultiDiscreteSpace& Other)
{
	this->High = Other.High;
}

FMultiDiscreteSpace::~FMultiDiscreteSpace()
{

}

void FMultiDiscreteSpace::Add(int DimSize)
{
	this->High.Add(DimSize);
}

void FMultiDiscreteSpace::Merge(const FMultiDiscreteSpace& Other)
{
	this->High.Append(Other.High);
}


ESpaceValidationResult FMultiDiscreteSpace::Validate(const TInstancedStruct<FPoint>& InPoint) const
{
	const FMultiDiscretePoint* TypedObservation = InPoint.GetPtr<FMultiDiscretePoint>();

	if (!TypedObservation)
	{
		return ESpaceValidationResult::WrongDataType;
	}

    if(this->High.Num() != TypedObservation->Values.Num())
    {
        return ESpaceValidationResult::WrongDimensions;
    }

    for(int i = 0; i < this->High.Num(); i++)
    {
        if(TypedObservation->Values[i] < 0 || TypedObservation->Values[i] >= this->High[i])
        {
            return ESpaceValidationResult::OutOfBounds;
        }
    }

	return ESpaceValidationResult::Success;
}

int FMultiDiscreteSpace::GetNumDimensions() const
{
	return this->High.Num();
}

int FMultiDiscreteSpace::GetMaxValue(const TArray<float>& Vector) const
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

int FMultiDiscreteSpace::GetFlattenedSize() const
{
	int Size = 0;
	for (int Dim : this->High)
	{
		Size += Dim;
	}
	return Size;
}

bool FMultiDiscreteSpace::IsEmpty() const
{
	return this->High.Num() == 0;
}

FString FMultiDiscreteSpace::ToString() const
{
	FString HighPart;
	for (int32 i = 0; i < High.Num(); ++i)
	{
		if (i > 0)
		{
			HighPart += TEXT(", ");
		}
		HighPart += FString::FromInt(High[i]);
	}
	return FString::Printf(TEXT("MultiDiscreteSpace(High=[%s])"), *HighPart);
}