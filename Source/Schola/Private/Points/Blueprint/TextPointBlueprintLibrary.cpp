// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Points/Blueprint/TextPointBlueprintLibrary.h"
#include "Points/TextPoint.h"
#include "Common/BlueprintErrorUtils.h"

TInstancedStruct<FTextPoint> UTextPointBlueprintLibrary::StringToTextPoint(const FString& InValue)
{
    return TInstancedStruct<FTextPoint>::Make(InValue);
}

FString UTextPointBlueprintLibrary::TextPointToString(const TInstancedStruct<FTextPoint>& InTextPoint)
{
    // Type check: ensure the InstancedStruct is actually a FTextPoint
    if (!InTextPoint.IsValid())
    {
        RaiseInvalidInstancedStructError(TEXT("TextPointToString"));
        return FString();
    }

    const FTextPoint* TypedPoint = InTextPoint.GetPtr<FTextPoint>();

    if (!TypedPoint)
    {
        RaiseInstancedStructTypeMismatchError(InTextPoint, TEXT("FTextPoint"), TEXT("TextPointToString"));
        return FString();
    }

    return TypedPoint->Value;
}
