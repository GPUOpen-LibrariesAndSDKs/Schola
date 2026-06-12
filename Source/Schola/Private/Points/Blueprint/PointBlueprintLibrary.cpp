// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Points/Blueprint/PointBlueprintLibrary.h"

#include "Common/BlueprintErrorUtils.h"
#include "Points/MultiBinaryPoint.h"
#include "Points/DiscretePoint.h"
#include "Points/BoxPoint.h"
#include "Points/DictPoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Points/TextPoint.h"

EPointType UPointBlueprintLibrary::Point_Type(const FInstancedStruct& InPoint)
{
	if (InPoint.GetScriptStruct() && InPoint.GetScriptStruct()->IsChildOf(FMultiBinaryPoint::StaticStruct()))
	{
		return EPointType::MultiBinary;
	}

    if (InPoint.GetScriptStruct() && InPoint.GetScriptStruct()->IsChildOf(FMultiDiscretePoint::StaticStruct()))
	{
		return EPointType::MultiDiscrete;
	}

	if (InPoint.GetScriptStruct() && InPoint.GetScriptStruct()->IsChildOf(FDiscretePoint::StaticStruct()))
	{
		return EPointType::Discrete;
	}

    if (InPoint.GetScriptStruct() && InPoint.GetScriptStruct()->IsChildOf(FBoxPoint::StaticStruct()))
	{
		return EPointType::Box;
	}

	if (InPoint.GetScriptStruct() && InPoint.GetScriptStruct()->IsChildOf(FDictPoint::StaticStruct()))
	{
		return EPointType::Dict;
	}

    if (InPoint.GetScriptStruct() && InPoint.GetScriptStruct()->IsChildOf(FTextPoint::StaticStruct()))
	{
		return EPointType::Text;
	}

    return EPointType::MultiBinary;
}

bool UPointBlueprintLibrary::Point_IsOfType(const FInstancedStruct& InPoint, EPointType InType)
{
	return Point_Type(InPoint) == InType;
}

FString UPointBlueprintLibrary::PointToString(const FInstancedStruct& InPoint)
{
	if (!InPoint.IsValid())
	{
		RaiseInvalidInstancedStructError(TEXT("PointToString"));
		return FString();
	}

	const UScriptStruct* ScriptStruct = InPoint.GetScriptStruct();
	if (!ScriptStruct || !ScriptStruct->IsChildOf(FPoint::StaticStruct()))
	{
		RaiseInstancedStructTypeMismatchError(InPoint, TEXT("FPoint"), TEXT("PointToString"));
		return FString();
	}

	const FPoint* Point = InPoint.GetPtr<FPoint>();
	return Point ? Point->ToString() : FString();
}
