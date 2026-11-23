// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "MjcfStl.h"
#include "Serialization/MemoryWriter.h"

#include <limits>

#if WITH_DEV_AUTOMATION_TESTS

namespace MjcfStlTest
{
	using FTriangle = TArray<FVector3f, TFixedAllocator<3>>;

	/** The box from (0, 0, 0) to (1, 2, 3), wound counter-clockwise seen from outside. */
	TArray<FTriangle> Box()
	{
		TArray<FTriangle> Triangles;
		const auto AddQuad = [&Triangles](FVector3f A, FVector3f B, FVector3f C, FVector3f D, FVector3f Outward) {
			if ((FVector3f::CrossProduct(B - A, C - A) | Outward) < 0)
			{
				Swap(B, D);
			}
			Triangles.Add({ A, B, C });
			Triangles.Add({ A, C, D });
		};
		const FVector3f S(1, 2, 3);
		AddQuad({ 0, 0, 0 }, { 0, S.Y, 0 }, { 0, S.Y, S.Z }, { 0, 0, S.Z }, { -1, 0, 0 });
		AddQuad({ S.X, 0, 0 }, { S.X, S.Y, 0 }, { S.X, S.Y, S.Z }, { S.X, 0, S.Z }, { 1, 0, 0 });
		AddQuad({ 0, 0, 0 }, { S.X, 0, 0 }, { S.X, 0, S.Z }, { 0, 0, S.Z }, { 0, -1, 0 });
		AddQuad({ 0, S.Y, 0 }, { S.X, S.Y, 0 }, { S.X, S.Y, S.Z }, { 0, S.Y, S.Z }, { 0, 1, 0 });
		AddQuad({ 0, 0, 0 }, { S.X, 0, 0 }, { S.X, S.Y, 0 }, { 0, S.Y, 0 }, { 0, 0, -1 });
		AddQuad({ 0, 0, S.Z }, { S.X, 0, S.Z }, { S.X, S.Y, S.Z }, { 0, S.Y, S.Z }, { 0, 0, 1 });
		return Triangles;
	}

	TArray<uint8> Binary(const TArray<FTriangle>& Triangles, const ANSICHAR* Header = "binary")
	{
		TArray<uint8> Bytes;
		Bytes.SetNumZeroed(80);
		FMemory::Memcpy(Bytes.GetData(), Header, FCStringAnsi::Strlen(Header));
		FMemoryWriter Ar(Bytes);
		Ar.Seek(80);
		uint32 Count = Triangles.Num();
		Ar << Count;
		for (FTriangle Triangle : Triangles)
		{
			FVector3f Normal = FVector3f::ZeroVector;
			uint16 Attributes = 0;
			Ar << Normal << Triangle[0] << Triangle[1] << Triangle[2] << Attributes;
		}
		return Bytes;
	}

	TArray<uint8> Text(const FString& Contents)
	{
		const FTCHARToUTF8 Utf8(*Contents);
		return TArray<uint8>(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
	}

	/** Returns the triangles of Mesh by corner position, for comparing meshes regardless of how corners are indexed. */
	TArray<FTriangle> Unindex(const FMjcfTriangleMesh& Mesh)
	{
		TArray<FTriangle> Triangles;
		for (int32 Triangle = 0; Triangle < Mesh.NumTriangles(); ++Triangle)
		{
			Triangles.Add({ Mesh.Positions[Mesh.Indices[3 * Triangle]], Mesh.Positions[Mesh.Indices[3 * Triangle + 1]], Mesh.Positions[Mesh.Indices[3 * Triangle + 2]] });
		}
		return Triangles;
	}

	void ExpectError(FAutomationTestBase& Test, const TCHAR* What, const TArray<uint8>& Bytes, const TCHAR* Fragment)
	{
		FString Error;
		const TOptional<FMjcfTriangleMesh> Mesh = FMjcfStlReader::Read(Bytes, Error);
		Test.TestFalse(FString::Printf(TEXT("%s: read fails"), What), Mesh.IsSet());
		Test.TestTrue(FString::Printf(TEXT("%s: error contains '%s', got '%s'"), What, Fragment, *Error), Error.Contains(Fragment));
	}
} // namespace MjcfStlTest

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfStlReadsTest, "Schola.Mjcf.Stl.Reads", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfStlReadsTest::RunTest(const FString& Parameters)
{
	using namespace MjcfStlTest;

	const TArray<FTriangle> Triangles = Box();
	FString Error;
	const TOptional<FMjcfTriangleMesh> FromBinary = FMjcfStlReader::Read(Binary(Triangles), Error);
	if (!TestTrue(TEXT("Binary reads: ") + Error, FromBinary.IsSet()))
	{
		return false;
	}
	TestEqual(TEXT("Shared corners are merged"), FromBinary->Positions.Num(), 8);
	TestTrue(TEXT("Binary keeps every triangle and its winding"), Unindex(*FromBinary) == Triangles);

	const TOptional<FMjcfTriangleMesh> SolidHeader = FMjcfStlReader::Read(Binary(Triangles, "solid exported as binary"), Error);
	TestTrue(TEXT("A binary file whose header starts with 'solid' is read as binary: ") + Error, SolidHeader.IsSet() && Unindex(*SolidHeader) == Triangles);

	TArray<uint8> Padded = Binary(Triangles);
	Padded.AddZeroed(7);
	TestTrue(TEXT("Bytes after the last triangle are ignored"), FMjcfStlReader::Read(Padded, Error).IsSet());

	TArray<FTriangle> WithDegenerate = Triangles;
	WithDegenerate.Add({ FVector3f(0, 0, 0), FVector3f(0, 0, 0), FVector3f(1, 0, 0) });
	WithDegenerate.Add({ FVector3f(-0.0f, 0, 0), FVector3f(0, 0, 0), FVector3f(1, 0, 0) });
	const TOptional<FMjcfTriangleMesh> Cleaned = FMjcfStlReader::Read(Binary(WithDegenerate), Error);
	TestTrue(TEXT("Triangles with repeated corners, including -0 and 0, are dropped"), Cleaned.IsSet() && Cleaned->NumTriangles() == 12);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfStlErrorsTest, "Schola.Mjcf.Stl.Errors", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfStlErrorsTest::RunTest(const FString& Parameters)
{
	using namespace MjcfStlTest;

	TArray<uint8> Truncated = Binary(Box());
	Truncated.SetNum(Truncated.Num() - 10);
	ExpectError(*this, TEXT("Truncated binary"), Truncated, TEXT("truncated"));
	ExpectError(*this, TEXT("Too short"), Text(TEXT("abc")), TEXT("too short"));
	ExpectError(*this, TEXT("No triangles"), Binary({}), TEXT("no triangles"));
	ExpectError(*this, TEXT("ASCII"), Text(TEXT("solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid x\n")), TEXT("ASCII STL, which isn't supported"));

	TArray<FTriangle> NotFinite = Box();
	NotFinite[3][1].Y = std::numeric_limits<float>::quiet_NaN();
	ExpectError(*this, TEXT("NaN corner"), Binary(NotFinite), TEXT("Triangle 4"));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfStlNormalsTest, "Schola.Mjcf.Stl.Normals", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfStlNormalsTest::RunTest(const FString& Parameters)
{
	using namespace MjcfStlTest;

	FString Error;
	const TOptional<FMjcfTriangleMesh> Cube = FMjcfStlReader::Read(Binary(Box()), Error);
	if (!TestTrue(TEXT("Box reads: ") + Error, Cube.IsSet()))
	{
		return false;
	}
	const TArray<FVector3f> CubeNormals = Cube->ComputeCornerNormals(40);
	bool bFlat = true;
	for (int32 Corner = 0; Corner < Cube->Indices.Num(); ++Corner)
	{
		const int32 Triangle = Corner / 3;
		const FVector3f& A = Cube->Positions[Cube->Indices[3 * Triangle]];
		const FVector3f Face = FVector3f::CrossProduct(Cube->Positions[Cube->Indices[3 * Triangle + 1]] - A, Cube->Positions[Cube->Indices[3 * Triangle + 2]] - A).GetSafeNormal();
		bFlat &= CubeNormals[Corner].Equals(Face, 1e-6f);
	}
	TestTrue(TEXT("Box edges are sharper than the crease, so every corner keeps its face's outward normal"), bFlat);

	// Two triangles meeting along the X axis, each tilted 10 degrees from +Z.
	const float Rise = FMath::Tan(FMath::DegreesToRadians(10.0f));
	TArray<FTriangle> Fold;
	Fold.Add({ FVector3f(0, 0, 0), FVector3f(1, 0, 0), FVector3f(0, 1, Rise) });
	Fold.Add({ FVector3f(1, 0, 0), FVector3f(0, 0, 0), FVector3f(0, -1, Rise) });
	const TOptional<FMjcfTriangleMesh> Shallow = FMjcfStlReader::Read(Binary(Fold), Error);
	if (TestTrue(TEXT("Fold reads: ") + Error, Shallow.IsSet()))
	{
		const TArray<FVector3f> Smooth = Shallow->ComputeCornerNormals(40);
		TestTrue(TEXT("Shared corners of a shallow fold get one smooth normal"), Smooth[0].Equals(FVector3f::UnitZ(), 1e-6f) && Smooth[4].Equals(FVector3f::UnitZ(), 1e-6f));
		const TArray<FVector3f> Sharp = Shallow->ComputeCornerNormals(5);
		TestFalse(TEXT("A smaller crease angle keeps the fold sharp"), Sharp[0].Equals(Sharp[4], 1e-3f));
	}
	return true;
}

#endif
