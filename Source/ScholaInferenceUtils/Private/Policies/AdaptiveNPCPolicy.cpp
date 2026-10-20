// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Policies/AdaptiveNPCPolicy.h"
#include "Spaces/BoxSpace.h"

namespace
{
// FSpace::Validate checks a point against a space; it does not validate the space definition itself.
bool ValidateBoxDefinition(const TInstancedStruct<FSpace>& Definition, bool bActionSpace, FString& Error)
{
    const TCHAR* Label = bActionSpace ? TEXT("Action") : TEXT("Observation");
    if (!Definition.IsValid() || Definition.GetScriptStruct() != FBoxSpace::StaticStruct())
    {
        Error = FString::Printf(TEXT("%s space must be a valid FBoxSpace; other space types are not supported yet."), Label);
        return false;
    }

    const FBoxSpace& Box = Definition.Get<FBoxSpace>();
    if (Box.IsEmpty())
    {
        Error = FString::Printf(TEXT("%s space must contain at least one dimension."), Label);
        return false;
    }

    // Empty Shape means a vector. A specified shape must match the number of stored bounds.
    if (!Box.Shape.IsEmpty())
    {
        int64 Size = 1;
        for (int Dimension : Box.Shape)
        {
            // Bound multiplication before doing it, including for hostile/overflowing Blueprint shapes.
            if (Dimension <= 0 || Size > Box.Dimensions.Num() / Dimension)
            {
                Error = FString::Printf(TEXT("%s space shape must have positive dimensions and match its number of bounds."), Label);
                return false;
            }
            Size *= Dimension;
        }
        if (Size != Box.Dimensions.Num())
        {
            Error = FString::Printf(TEXT("%s space shape must match its number of bounds."), Label);
            return false;
        }
    }

    for (int32 Index = 0; Index < Box.Dimensions.Num(); ++Index)
    {
        const FBoxSpaceDimension& Dimension = Box.Dimensions[Index];
        if ((Dimension.bHasLow && !FMath::IsFinite(Dimension.Low))
            || (Dimension.bHasHigh && !FMath::IsFinite(Dimension.High)))
        {
            Error = FString::Printf(TEXT("%s space dimension %d has a nonfinite enabled bound."), Label, Index);
            return false;
        }
        if (Dimension.IsFullyBounded() && Dimension.Low > Dimension.High)
        {
            Error = FString::Printf(TEXT("%s space dimension %d has its lower bound above its upper bound."), Label, Index);
            return false;
        }
        if (bActionSpace && (!Dimension.IsFullyBounded() || Dimension.Low == Dimension.High
            || !FMath::IsFinite(Dimension.High - Dimension.Low)))
        {
            Error = FString::Printf(TEXT("Action space dimension %d requires finite lower and upper bounds with a positive finite range."), Index);
            return false;
        }
    }
    return true;
}
}

bool UAdaptiveNPCPolicy::Init(const FInteractionDefinition& InDefinition)
{
    FString Error;
    if (!ValidateBoxDefinition(InDefinition.ObsSpaceDefn, false, Error)
        || !ValidateBoxDefinition(InDefinition.ActionSpaceDefn, true, Error))
    {
        LastError = MoveTemp(Error);
        if (!bInitialized)
        {
            SessionState = EAdaptiveSessionState::Failed;
        }
        return false;
    }

    InteractionDefinition = InDefinition;
    bInitialized = true;
    SessionState = EAdaptiveSessionState::Initialized;
    LastError.Reset();
    return true;
}

bool UAdaptiveNPCPolicy::GetInteractionDefinition(FInteractionDefinition& OutDefinition) const
{
    if (!bInitialized)
    {
        return false;
    }
    OutDefinition = InteractionDefinition;
    return true;
}

bool UAdaptiveNPCPolicy::Think(const TInstancedStruct<FPoint>& Observation, TInstancedStruct<FPoint>& Action)
{
    LastError = TEXT("Adaptive inference is not available yet. Local initialization does not establish a learner connection.");
    return false;
}

bool UAdaptiveNPCPolicy::BatchedThink(const TArray<TInstancedStruct<FPoint>>& Observations,
                                    TArray<TInstancedStruct<FPoint>>& Actions)
{
    LastError = TEXT("Adaptive inference is not available yet. Local initialization does not establish a learner connection.");
    return false;
}
