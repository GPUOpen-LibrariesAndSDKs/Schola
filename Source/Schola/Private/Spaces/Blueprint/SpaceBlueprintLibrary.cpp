// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/Blueprint/SpaceBlueprintLibrary.h"

#include "Spaces/MultiBinarySpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DictSpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/TextSpace.h"
#include "Common/BlueprintErrorUtils.h"
#include "Common/InstancedStructUtils.h"

ESpaceType USpaceBlueprintLibrary::Space_Type(const FInstancedStruct& InSpace)
{
	if (InSpace.GetScriptStruct() && InSpace.GetScriptStruct()->IsChildOf(FMultiBinarySpace::StaticStruct()))
    {
        return ESpaceType::MultiBinary;
    }

	if (InSpace.GetScriptStruct() && InSpace.GetScriptStruct()->IsChildOf(FDiscreteSpace::StaticStruct()))
	{
		return ESpaceType::Discrete;
	}
    
    if (InSpace.GetScriptStruct() && InSpace.GetScriptStruct()->IsChildOf(FMultiDiscreteSpace::StaticStruct()))
	{
		return ESpaceType::MultiDiscrete;
	}
    
    if (InSpace.GetScriptStruct() && InSpace.GetScriptStruct()->IsChildOf(FBoxSpace::StaticStruct()))
	{
		return ESpaceType::Box;
	}

    if (InSpace.GetScriptStruct() && InSpace.GetScriptStruct()->IsChildOf(FDictSpace::StaticStruct()))
	{
		return ESpaceType::Dict;
	}

    if (InSpace.GetScriptStruct() && InSpace.GetScriptStruct()->IsChildOf(FTextSpace::StaticStruct()))
	{
		return ESpaceType::Text;
	}

    return ESpaceType::MultiBinary;
}

bool USpaceBlueprintLibrary::Space_IsOfType(const FInstancedStruct& InSpace, ESpaceType InType)
{
	return Space_Type(InSpace) == InType;
}

bool USpaceBlueprintLibrary::Space_Contains(const FInstancedStruct& InSpace, const FInstancedStruct& InPoint)
{
	if (!InSpace.IsValid() || !InPoint.IsValid())
	{
		RaiseInvalidInstancedStructError(TEXT("USpaceBlueprintLibrary::Space_Contains()"));
		return false;
	}

	if (!InSpace.GetPtr<FSpace>())
	{
		RaiseInstancedStructTypeMismatchError(InSpace, TEXT("FSpace"), TEXT("USpaceBlueprintLibrary::Space_Contains()"));
		return false;
	}
	
	if (!InPoint.GetPtr<FPoint>())
	{
		RaiseInstancedStructTypeMismatchError(InPoint, TEXT("FPoint"), TEXT("USpaceBlueprintLibrary::Space_Contains()"));
		return false;
	}

	return InSpace.GetPtr<FSpace>()->Contains(ToTypedInstancedStruct<FPoint>(InPoint));
}

FString USpaceBlueprintLibrary::SpaceToString(const FInstancedStruct& InSpace)
{
	if (!InSpace.IsValid())
	{
		RaiseInvalidInstancedStructError(TEXT("SpaceToString"));
		return FString();
	}

	const UScriptStruct* ScriptStruct = InSpace.GetScriptStruct();
	if (!ScriptStruct || !ScriptStruct->IsChildOf(FSpace::StaticStruct()))
	{
		RaiseInstancedStructTypeMismatchError(InSpace, TEXT("FSpace"), TEXT("SpaceToString"));
		return FString();
	}

	const FSpace* Space = InSpace.GetPtr<FSpace>();
	return Space ? Space->ToString() : FString();
}
