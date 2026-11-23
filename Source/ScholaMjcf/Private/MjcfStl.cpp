// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "MjcfStl.h"
#include "Misc/FileHelper.h"
#include "Serialization/MemoryReader.h"

namespace
{
	constexpr int64 HeaderSize = 84;
	constexpr int64 TriangleSize = 50;

	/** Returns Value with -0 replaced by +0, which compares equal to it but hashes differently. */
	float WithoutNegativeZero(float Value)
	{
		uint32 Bits;
		FMemory::Memcpy(&Bits, &Value, sizeof(Bits));
		return (Bits & 0x7fffffff) == 0 ? 0.0f : Value;
	}

	/** Collects triangles, merging corners at equal positions and dropping triangles that merging collapses. */
	class FTriangleCollector
	{
	public:
		void Add(const FVector3f (&Corners)[3])
		{
			int32 Indices[3];
			for (int32 Corner = 0; Corner < 3; ++Corner)
			{
				const FVector3f Position(WithoutNegativeZero(Corners[Corner].X), WithoutNegativeZero(Corners[Corner].Y), WithoutNegativeZero(Corners[Corner].Z));
				Indices[Corner] = IndexOf.FindOrAdd(Position, Mesh.Positions.Num());
				if (Indices[Corner] == Mesh.Positions.Num())
				{
					Mesh.Positions.Add(Position);
				}
			}
			if (Indices[0] != Indices[1] && Indices[1] != Indices[2] && Indices[2] != Indices[0])
			{
				Mesh.Indices.Append(Indices, 3);
			}
		}

		TOptional<FMjcfTriangleMesh> Finish(FString& OutError)
		{
			if (Mesh.Indices.IsEmpty())
			{
				OutError = TEXT("The file contains no triangles");
				return {};
			}
			return MoveTemp(Mesh);
		}

	private:
		FMjcfTriangleMesh Mesh;
		TMap<FVector3f, int32> IndexOf;
	};

	bool StartsWithSolid(TConstArrayView<uint8> Bytes)
	{
		int32 Start = 0;
		while (Start < Bytes.Num() && FChar::IsWhitespace(Bytes[Start]))
		{
			++Start;
		}
		return Bytes.Num() - Start >= 5 && FCStringAnsi::Strnicmp(reinterpret_cast<const ANSICHAR*>(Bytes.GetData() + Start), "solid", 5) == 0;
	}
} // namespace

TArray<FVector3f> FMjcfTriangleMesh::ComputeCornerNormals(float CreaseAngleDegrees) const
{
	const int32 Triangles = NumTriangles();
	TArray<FVector3f> AreaNormals;
	TArray<FVector3f> UnitNormals;
	AreaNormals.Reserve(Triangles);
	UnitNormals.Reserve(Triangles);
	for (int32 Triangle = 0; Triangle < Triangles; ++Triangle)
	{
		const FVector3f& A = Positions[Indices[3 * Triangle]];
		const FVector3f& B = Positions[Indices[3 * Triangle + 1]];
		const FVector3f& C = Positions[Indices[3 * Triangle + 2]];
		AreaNormals.Add(FVector3f::CrossProduct(B - A, C - A));
		UnitNormals.Add(AreaNormals.Last().GetSafeNormal());
	}

	// The triangles around each position, as ranges of TrianglesAt.
	TArray<int32> FirstAt;
	FirstAt.SetNumZeroed(Positions.Num() + 1);
	for (const int32 Index : Indices)
	{
		++FirstAt[Index + 1];
	}
	for (int32 Position = 0; Position < Positions.Num(); ++Position)
	{
		FirstAt[Position + 1] += FirstAt[Position];
	}
	TArray<int32> TrianglesAt;
	TrianglesAt.SetNumUninitialized(Indices.Num());
	TArray<int32> Filled = FirstAt;
	for (int32 Corner = 0; Corner < Indices.Num(); ++Corner)
	{
		TrianglesAt[Filled[Indices[Corner]]++] = Corner / 3;
	}

	const float MinCosine = FMath::Cos(FMath::DegreesToRadians(CreaseAngleDegrees));
	TArray<FVector3f> Normals;
	Normals.SetNumUninitialized(Indices.Num());
	for (int32 Corner = 0; Corner < Indices.Num(); ++Corner)
	{
		const int32 Triangle = Corner / 3;
		const int32 Position = Indices[Corner];
		FVector3f Sum = FVector3f::ZeroVector;
		for (int32 Around = FirstAt[Position]; Around < FirstAt[Position + 1]; ++Around)
		{
			const int32 Other = TrianglesAt[Around];
			if ((UnitNormals[Triangle] | UnitNormals[Other]) >= MinCosine)
			{
				Sum += AreaNormals[Other];
			}
		}
		const FVector3f Normal = Sum.GetSafeNormal();
		Normals[Corner] = Normal.IsZero() ? FVector3f::UnitZ() : Normal;
	}
	return Normals;
}

TOptional<FMjcfTriangleMesh> FMjcfStlReader::ReadFile(const FString& File, FString& OutError)
{
	TArray<uint8> Bytes;
	if (!FFileHelper::LoadFileToArray(Bytes, *File, FILEREAD_Silent))
	{
		OutError = TEXT("The file could not be read");
		return {};
	}
	return Read(Bytes, OutError);
}

TOptional<FMjcfTriangleMesh> FMjcfStlReader::Read(TConstArrayView<uint8> Bytes, FString& OutError)
{
	// Binary STL is an 80-byte header of free text, a triangle count, and per triangle a normal, three corners, and two
	// attribute bytes, all little-endian. Some binary headers start with "solid" like ASCII files do, so only the size is
	// trusted to identify the format.
	FMemoryReaderView Ar(Bytes);
	Ar.SetByteSwapping(!PLATFORM_LITTLE_ENDIAN);
	Ar.Seek(80);
	uint32 NumTriangles = 0;
	Ar << NumTriangles;
	if (Ar.IsError())
	{
		OutError = FString::Printf(TEXT("The file is too short to be an STL file (%d bytes)"), Bytes.Num());
		return {};
	}
	const int64 NeededSize = HeaderSize + int64(NumTriangles) * TriangleSize;
	if (Bytes.Num() < NeededSize)
	{
		OutError = StartsWithSolid(Bytes)
			? FString(TEXT("The file is ASCII STL, which isn't supported; re-export it as binary STL"))
			: FString::Printf(TEXT("The file is truncated: its header lists %u triangles, which need %lld bytes, but it has %d"), NumTriangles, NeededSize, Bytes.Num());
		return {};
	}

	FTriangleCollector Collector;
	for (uint32 Triangle = 0; Triangle < NumTriangles; ++Triangle)
	{
		FVector3f Normal;
		FVector3f Corners[3];
		uint16 Attributes;
		Ar << Normal << Corners[0] << Corners[1] << Corners[2] << Attributes;
		for (const FVector3f& Corner : Corners)
		{
			if (!FMath::IsFinite(Corner.X) || !FMath::IsFinite(Corner.Y) || !FMath::IsFinite(Corner.Z))
			{
				OutError = FString::Printf(TEXT("Triangle %u has a corner that is not a finite number"), Triangle + 1);
				return {};
			}
		}
		Collector.Add(Corners);
	}
	return Collector.Finish(OutError);
}
