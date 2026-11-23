// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "MjcfConversion.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace MjcfConversionTest
{
	constexpr double Tolerance = 1e-9;

	/** A placement as MuJoCo writes it: a position and rotation relative to the parent frame. */
	struct FPlacement
	{
		FVector Position;
		FQuat Rotation;
	};

	FVector Place(const FPlacement& Placement, const FVector& Point)
	{
		return Placement.Position + Placement.Rotation.RotateVector(Point);
	}

	FPlacement Convert(const FPlacement& Placement)
	{
		return { FMjcfConversion::ConvertPosition(Placement.Position), FMjcfConversion::ConvertRotation(Placement.Rotation) };
	}

	FQuat AxisAngle(const FVector& Axis, double Degrees)
	{
		return FQuat(Axis.GetSafeNormal(), FMath::DegreesToRadians(Degrees));
	}
} // namespace MjcfConversionTest

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfConversionReferenceTest, "Schola.Mjcf.Conversion.Reference", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfConversionReferenceTest::RunTest(const FString& Parameters)
{
	using namespace MjcfConversionTest;

	TestEqual(TEXT("Meters become centimeters and Y is mirrored"), FMjcfConversion::ConvertPosition(FVector(1, 2, 3)), FVector(100, -200, 300), Tolerance);

	// MuJoCo turns +X toward +Y for a positive rotation about +Z; Y is mirrored in Unreal, so +X must turn toward -Y.
	const FQuat Turn = FMjcfConversion::ConvertRotation(AxisAngle(FVector::UpVector, 90));
	TestEqual(TEXT("A quarter turn about Z takes +X to -Y"), Turn.RotateVector(FVector::ForwardVector), FVector(0, -1, 0), Tolerance);
	TestEqual(TEXT("A quarter turn about Z keeps +Z"), Turn.RotateVector(FVector::UpVector), FVector::UpVector, Tolerance);

	const FQuat Pitch = FMjcfConversion::ConvertRotation(AxisAngle(FVector::RightVector, 90));
	TestEqual(TEXT("A quarter turn about Y still takes +X to -Z"), Pitch.RotateVector(FVector::ForwardVector), FVector(0, 0, -1), Tolerance);

	const FMatrix Mesh = FMjcfConversion::MeshToUnreal(FVector(0.5, 1, 2));
	TestEqual(TEXT("Mesh scale is applied per axis before the axis conversion"), Mesh.TransformPosition(FVector(1, 2, 3)), FVector(50, -200, 600), Tolerance);
	TestTrue(TEXT("The mesh conversion mirrors"), Mesh.Determinant() < 0);
	TestTrue(TEXT("A negative mesh scale mirrors back"), FMjcfConversion::MeshToUnreal(FVector(1, -1, 1)).Determinant() > 0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfConversionPlacementTest, "Schola.Mjcf.Conversion.PlacementMatchesPlacedGeometry", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfConversionPlacementTest::RunTest(const FString& Parameters)
{
	using namespace MjcfConversionTest;

	const FPlacement Body{ FVector(0.3, -1.2, 0.8), AxisAngle(FVector(1, 2, 3), 47) };
	const FPlacement Geom{ FVector(-0.05, 0.4, 0.1), AxisAngle(FVector(-2, 0.5, 1), -130) };
	const TArray<FVector> Scales = { FVector::OneVector, FVector(0.001, 0.002, 0.003), FVector(-1, 1, 1), FVector(2, -0.5, -3) };
	const TArray<FVector> Vertices = { FVector::ZeroVector, FVector(1, 0, 0), FVector(0.25, -0.75, 1.5), FVector(-3, 2, -1) };

	for (const FVector& Scale : Scales)
	{
		for (const FVector& Vertex : Vertices)
		{
			const FVector InMujoco = Place(Body, Place(Geom, Scale * Vertex));
			const FVector Expected = FMjcfConversion::ConvertPosition(InMujoco);
			const FVector Actual = Place(Convert(Body), Place(Convert(Geom), FMjcfConversion::MeshToUnreal(Scale).TransformPosition(Vertex)));
			TestEqual(FString::Printf(TEXT("Vertex %s with scale %s"), *Vertex.ToString(), *Scale.ToString()), Actual, Expected, 1e-6);
		}
	}

	const FQuat ComposedThenConverted = FMjcfConversion::ConvertRotation(Body.Rotation * Geom.Rotation);
	const FQuat ConvertedThenComposed = FMjcfConversion::ConvertRotation(Body.Rotation) * FMjcfConversion::ConvertRotation(Geom.Rotation);
	TestTrue(TEXT("Converting nested rotations matches nesting converted rotations"), ComposedThenConverted.Equals(ConvertedThenComposed, Tolerance));
	return true;
}

#endif
