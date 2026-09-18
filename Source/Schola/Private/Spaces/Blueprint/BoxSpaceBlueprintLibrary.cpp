// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/Blueprint/BoxSpaceBlueprintLibrary.h"
#include "Spaces/BoxSpace.h"

TInstancedStruct<FBoxSpace> UBoxSpaceBlueprintLibrary::ArraysToBoxSpace(const TArray<float>& InLow, const TArray<float>& InHigh, const TArray<int32>& InShape)
{
	return TInstancedStruct<FBoxSpace>::Make<FBoxSpace>(InLow, InHigh, InShape);
}

TInstancedStruct<FBoxSpace> UBoxSpaceBlueprintLibrary::VectorToBoxSpace(const FVector& InLow, const FVector& InHigh)
{
	TArray<float> Low;
	Low.Add(InLow.X);
	Low.Add(InLow.Y);
	Low.Add(InLow.Z);

	TArray<float> High;
	High.Add(InHigh.X);
	High.Add(InHigh.Y);
	High.Add(InHigh.Z);

	TArray<int32> Shape;
	Shape.Add(3);

	return TInstancedStruct<FBoxSpace>::Make<FBoxSpace>(Low, High, Shape);
}

TInstancedStruct<FBoxSpace> UBoxSpaceBlueprintLibrary::RotatorSpace()
{
	TArray<float> Low;
	Low.Add(-180.0f);
	Low.Add(-180.0f);
	Low.Add(-180.0f);

	TArray<float> High;
	High.Add(180.0f);
	High.Add(180.0f);
	High.Add(180.0f);

	TArray<int32> Shape;
	Shape.Add(3);

	return TInstancedStruct<FBoxSpace>::Make<FBoxSpace>(Low, High, Shape);
}

TInstancedStruct<FBoxSpace> UBoxSpaceBlueprintLibrary::TransformToBoxSpace(
	const FVector& InLocationLow, 
	const FVector& InLocationHigh,
	const FVector& InScaleLow,
	const FVector& InScaleHigh)
{
	TArray<float> Low;
	// Location
	Low.Add(InLocationLow.X);
	Low.Add(InLocationLow.Y);
	Low.Add(InLocationLow.Z);
	// Rotation
	Low.Add(-180.0f);
	Low.Add(-180.0f);
	Low.Add(-180.0f);
	// Scale
	Low.Add(InScaleLow.X);
	Low.Add(InScaleLow.Y);
	Low.Add(InScaleLow.Z);

	TArray<float> High;
	// Location
	High.Add(InLocationHigh.X);
	High.Add(InLocationHigh.Y);
	High.Add(InLocationHigh.Z);
	// Rotation
	High.Add(180.0f);
	High.Add(180.0f);
	High.Add(180.0f);
	// Scale
	High.Add(InScaleHigh.X);
	High.Add(InScaleHigh.Y);
	High.Add(InScaleHigh.Z);

	TArray<int32> Shape;
	Shape.Add(9);

	return TInstancedStruct<FBoxSpace>::Make<FBoxSpace>(Low, High, Shape);
}

FBoxSpaceDimension UBoxSpaceBlueprintLibrary::MakeUnboundedBoxSpaceDimension()
{
	return FBoxSpaceDimension::Unbounded();
}

FBoxSpaceDimension UBoxSpaceBlueprintLibrary::MakeLowerBoundedBoxSpaceDimension(float InLow)
{
	return FBoxSpaceDimension::LowerBounded(InLow);
}

FBoxSpaceDimension UBoxSpaceBlueprintLibrary::MakeUpperBoundedBoxSpaceDimension(float InHigh)
{
	return FBoxSpaceDimension::UpperBounded(InHigh);
}

TInstancedStruct<FBoxSpace> UBoxSpaceBlueprintLibrary::MakeUnboundedBoxSpace(const TArray<int32>& InShape)
{
	TInstancedStruct<FBoxSpace> Result;
	Result.InitializeAs<FBoxSpace>(InShape);
	FBoxSpace& Space = Result.GetMutable<FBoxSpace>();
	const FBoxSpaceDimension UnboundedDimension = FBoxSpaceDimension::Unbounded();
	for (FBoxSpaceDimension& Dimension : Space.Dimensions)
	{
		Dimension = UnboundedDimension;
	}
	return Result;
}