// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "MjcfConversion.h"
#include "Math/ScaleMatrix.h"

FVector FMjcfConversion::ConvertPosition(const FVector& Position)
{
	return FVector(Position.X, -Position.Y, Position.Z) * CentimetersPerMeter;
}

FQuat FMjcfConversion::ConvertRotation(const FQuat& Rotation)
{
	return FQuat(-Rotation.X, Rotation.Y, -Rotation.Z, Rotation.W);
}

FMatrix FMjcfConversion::MeshToUnreal(const FVector& MeshScale)
{
	return FScaleMatrix(FVector(MeshScale.X, -MeshScale.Y, MeshScale.Z) * CentimetersPerMeter);
}
