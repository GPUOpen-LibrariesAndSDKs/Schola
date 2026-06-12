// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Points/Blueprint/DiscretePointBlueprintLibrary.h"
#include "Points/DiscretePoint.h"
#include "Common/BlueprintErrorUtils.h"

TInstancedStruct<FDiscretePoint> UDiscretePointBlueprintLibrary::Int32ToDiscretePoint(int32 InValue)
{
	return TInstancedStruct<FDiscretePoint>::Make(InValue);
}

int32 UDiscretePointBlueprintLibrary::DiscretePointToInt32(const TInstancedStruct<FDiscretePoint>& InDiscretePoint)
{
    // Type check: ensure the InstancedStruct is actually a FDiscretePoint
    if (!InDiscretePoint.IsValid())
    {
        RaiseInvalidInstancedStructError(TEXT("DiscretePointToInt32"));
        return 0;
    }

    const FDiscretePoint* TypedPoint = InDiscretePoint.GetPtr<FDiscretePoint>();
    
    if (!TypedPoint)
    {
        RaiseInstancedStructTypeMismatchError(InDiscretePoint, TEXT("FDiscretePoint"), TEXT("DiscretePointToInt32"));
        return 0;
    }

    return TypedPoint->Value;

}

