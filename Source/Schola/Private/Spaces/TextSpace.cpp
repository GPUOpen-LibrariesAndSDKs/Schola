// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/TextSpace.h"

FTextSpace::FTextSpace()
	: MaxLength(0)
{
}

void FTextSpace::Copy(const FTextSpace& Other)
{
	this->MaxLength = Other.MaxLength;
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

	if (Length < this->MinLength)
	{
		return ESpaceValidationResult::OutOfBounds;
	}

	// An empty charset maps to Gymnasium's default (alphanumeric) set when the space
	// is exchanged with Python, where "no restriction" cannot be represented. Validate
	// against that same set so a point accepted here is also accepted across the boundary.
	const FString EffectiveCharset = this->Charset.IsEmpty()
		? FString(ScholaTextCharsets::Alphanumeric)
		: this->Charset;

	for (const TCHAR Character : TypedObservation->Value)
	{
		int32 Index = INDEX_NONE;
		if (!EffectiveCharset.FindChar(Character, Index))
		{
			return ESpaceValidationResult::OutOfBounds;
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
