// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.



#include "Points/Blueprint/MultiDiscretePointBlueprintLibrary.h"
#include "Points/MultiDiscretePoint.h"
#include "Common/BlueprintErrorUtils.h"



TInstancedStruct<FMultiDiscretePoint> UMultiDiscretePointBlueprintLibrary::ArrayToMultiDiscretePoint(const TArray<int32>& InValues)
{

	return TInstancedStruct<FMultiDiscretePoint>::Make(InValues);

}



TArray<int32> UMultiDiscretePointBlueprintLibrary::MultiDiscretePointToArray(const TInstancedStruct<FMultiDiscretePoint>& InMultiDiscretePoint)
{
    // Type check: ensure the InstancedStruct is actually a FMultiDiscretePoint

    if (!InMultiDiscretePoint.IsValid())
    {

        RaiseInvalidInstancedStructError(TEXT("MultiDiscretePointToArray"));

        return TArray<int32>();

    }

    const FMultiDiscretePoint* TypedPoint = InMultiDiscretePoint.GetPtr<FMultiDiscretePoint>();

    if (!TypedPoint)
    {

        RaiseInstancedStructTypeMismatchError(InMultiDiscretePoint, TEXT("FMultiDiscretePoint"), TEXT("MultiDiscretePointToArray"));

        return TArray<int32>();

    }

    return TypedPoint->Values;

}

