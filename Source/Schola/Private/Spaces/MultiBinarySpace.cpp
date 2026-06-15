// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/MultiBinarySpace.h"

FMultiBinarySpace::FMultiBinarySpace()
{
}

FMultiBinarySpace::FMultiBinarySpace(int Shape)
{
	this->Shape = Shape;
}

void FMultiBinarySpace::Merge(const FMultiBinarySpace& Other)
{
	this->Shape += Other.Shape;
}

void FMultiBinarySpace::Copy(const FMultiBinarySpace& Other)
{
	this->Shape = Other.Shape;
}


int FMultiBinarySpace::GetNumDimensions() const
{
	return Shape;
}

ESpaceValidationResult FMultiBinarySpace::Validate(const TInstancedStruct<FPoint>& InPoint) const
{
	if (!InPoint.GetPtr<FMultiBinaryPoint>())
    {
        return ESpaceValidationResult::WrongDataType;
    }
	const FMultiBinaryPoint& P = InPoint.Get<FMultiBinaryPoint>();
    if (Shape != P.Values.Num())
    {
        return ESpaceValidationResult::WrongDimensions;
    }
    return ESpaceValidationResult::Success;
}

int FMultiBinarySpace::GetFlattenedSize() const
{
	return this->Shape;
}

bool FMultiBinarySpace::IsEmpty() const
{
    return this->Shape == 0;
}

FString FMultiBinarySpace::ToString() const
{
	return FString::Printf(TEXT("MultiBinarySpace(Shape=%d)"), Shape);
}