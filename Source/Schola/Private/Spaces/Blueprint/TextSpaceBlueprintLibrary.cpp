// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/Blueprint/TextSpaceBlueprintLibrary.h"
#include "Spaces/TextSpace.h"
#include "Common/BlueprintErrorUtils.h"

TInstancedStruct<FTextSpace> UTextSpaceBlueprintLibrary::MakeTextSpace(int32 InMaxLength, bool bInHasMinLength, int32 InMinLength, const FString& InCharset)
{
    return TInstancedStruct<FTextSpace>::Make(InMaxLength, bInHasMinLength, InMinLength, InCharset);
}

void UTextSpaceBlueprintLibrary::BreakTextSpace(const TInstancedStruct<FTextSpace>& InTextSpace, int32& OutMaxLength, bool& bOutHasMinLength, int32& OutMinLength, FString& OutCharset)
{
    OutMaxLength = 0;
    bOutHasMinLength = false;
    OutMinLength = 0;
    OutCharset = FString();

    // Type check: ensure the InstancedStruct is actually a FTextSpace
    if (!InTextSpace.IsValid())
    {
        RaiseInvalidInstancedStructError(TEXT("BreakTextSpace"));
        return;
    }

    const FTextSpace* TypedSpace = InTextSpace.GetPtr<FTextSpace>();

    if (!TypedSpace)
    {
        RaiseInstancedStructTypeMismatchError(InTextSpace, TEXT("FTextSpace"), TEXT("BreakTextSpace"));
        return;
    }

    OutMaxLength = TypedSpace->MaxLength;
    bOutHasMinLength = TypedSpace->bHasMinLength;
    OutMinLength = TypedSpace->MinLength;
    OutCharset = TypedSpace->Charset;
}
