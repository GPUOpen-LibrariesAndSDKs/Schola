// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/DictSpace.h"

FDictSpace::FDictSpace()
{

}

int FDictSpace::Num()
{
	return Spaces.Num();
}

int FDictSpace::GetNumDimensions() const
{
	return Spaces.Num();
}

bool FDictSpace::IsEmpty() const
{
	return Spaces.IsEmpty();
}

int FDictSpace::GetFlattenedSize() const
{
	int Size = 0;
	for (const TPair<FString, TInstancedStruct<FSpace>>& Pair : Spaces)
	{
		const FSpace* SpacePtr = Pair.Value.GetPtr<FSpace>();
		if (SpacePtr)
		{
			Size += SpacePtr->GetFlattenedSize();
		}
	}
	return Size;
}

ESpaceValidationResult FDictSpace::Validate(const TInstancedStruct<FPoint>& InPoint) const
{
	
	if (!InPoint.GetPtr<FDictPoint>())
	{
		return ESpaceValidationResult::WrongDataType;
	}
	const FDictPoint& DictPoint = InPoint.Get<FDictPoint>();
	// Check that the number of keys in the point matches the number of keys in the space
	if (DictPoint.Points.Num() != Spaces.Num())
	{
		return ESpaceValidationResult::WrongDimensions;
	}

	// Check that each sub-point matches the corresponding sub-space
	for (const TPair<FString, TInstancedStruct<FSpace>>& Pair : Spaces)
	{
		const FString& Key = Pair.Key;
		const FSpace* SubSpacePtr = Pair.Value.GetPtr<FSpace>();
		const TInstancedStruct<FPoint>* SubPoint = DictPoint.Points.Find(Key);
		if (!SubPoint)
		{
			return ESpaceValidationResult::WrongDimensions;
		}
		if (!SubSpacePtr)
		{
			return ESpaceValidationResult::WrongDataType;
		}
		ESpaceValidationResult Result = SubSpacePtr->Validate(*SubPoint);
		if (Result != ESpaceValidationResult::Success)
		{
			return Result;
		}
	}
	return ESpaceValidationResult::Success;
}

FString FDictSpace::ToString() const
{
	FString Body;
	for (const TPair<FString, TInstancedStruct<FSpace>>& Pair : Spaces)
	{
		const FSpace* SubSpace = Pair.Value.GetPtr<FSpace>();
		const FString SubStr = SubSpace ? SubSpace->ToString() : TEXT("(null)");
		Body += FString::Printf(TEXT("\t%s: %s,\n"), *Pair.Key, *SubStr);
	}
	return FString::Printf(TEXT("DictSpace(\n{%s})"), *Body);
}
