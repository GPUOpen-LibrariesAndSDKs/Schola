// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/TextSpace.h"



FTextSpace::FTextSpace()
	: MaxLength(0)
{
}

void FTextSpace::Copy(const FTextSpace& Other)
{
	this->MaxLength = Other.MaxLength;
	this->bHasMinLength = Other.bHasMinLength;
	this->MinLength = Other.MinLength;
	this->Charset = Other.Charset;
}


ESpaceValidationResult FTextSpace::Validate(const TInstancedStruct<FPoint>& InPoint) const
{

	const FTextPoint* TypedObservation = InPoint.GetPtr<FTextPoint>();
	if (!TypedObservation)
	{
		return ESpaceValidationResult::WrongDataType;
	}

	const int Length = TypedObservation->Value.Len();
	if (Length > this->MaxLength)
	{
		return ESpaceValidationResult::OutOfBounds;
	}

	if (this->bHasMinLength && Length < this->MinLength)
	{
		return ESpaceValidationResult::OutOfBounds;
	}

	if (!this->Charset.IsEmpty())
	{
		for (const TCHAR Character : TypedObservation->Value)
		{
			int32 Index = INDEX_NONE;
			if (!this->Charset.FindChar(Character, Index))
			{
				return ESpaceValidationResult::OutOfBounds;
			}
		}
	}

	return ESpaceValidationResult::Success;
}

int FTextSpace::GetNumDimensions() const
{
	return 1;
}

int FTextSpace::GetFlattenedSize() const
{
	return this->MaxLength;
}

bool FTextSpace::IsEmpty() const
{
	return this->MaxLength == 0;
}
