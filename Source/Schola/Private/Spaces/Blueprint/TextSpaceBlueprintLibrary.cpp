// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/Blueprint/TextSpaceBlueprintLibrary.h"
#include "Spaces/TextSpace.h"
#include "Common/BlueprintErrorUtils.h"

TInstancedStruct<FTextSpace> UTextSpaceBlueprintLibrary::MakeTextSpace(int32 InMaxLength, int32 InMinLength, const FString& InCharset)
{
    // A space with MinLength > MaxLength can never contain a valid string. Clamp and warn rather
    // than producing an incoherent space. (UPARAM/clamp metadata can bound each value individually
    // but cannot express the relational MinLength <= MaxLength constraint, so it is enforced here.)
    if (InMinLength > InMaxLength)
    {
        RaiseBlueprintError(TEXT("MakeTextSpace"), FString::Printf(TEXT("Min Length (%d) cannot exceed Max Length (%d). Clamping Min Length to Max Length."), InMinLength, InMaxLength));
        InMinLength = InMaxLength;
    }
    return TInstancedStruct<FTextSpace>::Make(InMaxLength, InMinLength, InCharset);
}

TInstancedStruct<FTextSpace> UTextSpaceBlueprintLibrary::MakeTextSpaceFromPreset(int32 InMaxLength, int32 InMinLength, ETextCharsetPreset InCharsetPreset)
{
    return MakeTextSpace(InMaxLength, InMinLength, GetTextCharsetPreset(InCharsetPreset));
}

void UTextSpaceBlueprintLibrary::BreakTextSpace(const TInstancedStruct<FTextSpace>& InTextSpace, int32& OutMaxLength, int32& OutMinLength, FString& OutCharset)
{
    OutMaxLength = 0;
    OutMinLength = 1;
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
    OutMinLength = TypedSpace->MinLength;
    OutCharset = TypedSpace->Charset;
}

FString UTextSpaceBlueprintLibrary::GetTextCharsetPreset(ETextCharsetPreset InPreset)
{
    switch (InPreset)
    {
    case ETextCharsetPreset::Numeric:
        return ScholaTextCharsets::Numeric;
    case ETextCharsetPreset::Alphabetic:
        return ScholaTextCharsets::Alphabetic;
    case ETextCharsetPreset::Alphanumeric:
    default:
        return ScholaTextCharsets::Alphanumeric;
    }
}
