// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Sensors/FakeCameraSensor.h"
#include "LogScholaInteractors.h"

int32 UFakeCameraSensor::GetObservationSize() const
{
	const int32 SafeNumChannels = FMath::Max(NumChannels, 1);
	const int32 SafeWidth = FMath::Max(Width, 1);
	const int32 SafeHeight = FMath::Max(Height, 1);
	return SafeNumChannels * SafeHeight * SafeWidth;
}

void UFakeCameraSensor::GetObservationSpace_Implementation(FInstancedStruct& OutObservationSpace) const
{
	const int32 SafeNumChannels = FMath::Max(NumChannels, 1);
	const int32 SafeWidth = FMath::Max(Width, 1);
	const int32 SafeHeight = FMath::Max(Height, 1);

	OutObservationSpace.InitializeAs<FBoxSpace>();
	FBoxSpace& SpaceDefinition = OutObservationSpace.GetMutable<FBoxSpace>();
	SpaceDefinition.Dimensions.Init(FBoxSpaceDimension(0.0, 1.0), SafeNumChannels * SafeHeight * SafeWidth);
	SpaceDefinition.Shape = { SafeNumChannels, SafeHeight, SafeWidth };
}

void UFakeCameraSensor::CollectObservations_Implementation(FInstancedStruct& OutObservations)
{
	const int32 SafeNumChannels = FMath::Max(NumChannels, 1);
	const int32 SafeWidth = FMath::Max(Width, 1);
	const int32 SafeHeight = FMath::Max(Height, 1);
	const int32 ExpectedSize = SafeNumChannels * SafeHeight * SafeWidth;

	OutObservations.InitializeAs<FBoxPoint>();
	FBoxPoint& OutBoxPoint = OutObservations.GetMutable<FBoxPoint>();
	OutBoxPoint.Shape = { SafeNumChannels, SafeHeight, SafeWidth };
	OutBoxPoint.Values.Init(0.0f, ExpectedSize);

	if (CustomImage.Num() == 0)
	{
		return;
	}

	if (CustomImage.Num() != ExpectedSize)
	{
		UE_LOGFMT(
			LogScholaInteractors,
			Warning,
			"UFakeCameraSensor::CollectObservations_Implementation(): CustomImage size {0} does not match expected size {1} (C={2}, H={3}, W={4}). Copying the overlapping prefix and zero-filling the rest.",
			CustomImage.Num(),
			ExpectedSize,
			SafeNumChannels,
			SafeHeight,
			SafeWidth);
	}

	const int32 CopyCount = FMath::Min(ExpectedSize, CustomImage.Num());
	FMemory::Memcpy(OutBoxPoint.Values.GetData(), CustomImage.GetData(), CopyCount * sizeof(float));
}

FString UFakeCameraSensor::GenerateId() const
{
	return FString::Printf(
		TEXT("FakeCamera_C%d_W%d_H%d"),
		FMath::Max(NumChannels, 1),
		FMath::Max(Width, 1),
		FMath::Max(Height, 1));
}
