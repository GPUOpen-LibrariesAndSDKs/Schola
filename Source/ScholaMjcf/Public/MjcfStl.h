// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

/** Triangles read from a mesh file, in the file's units and axes. */
struct SCHOLAMJCF_API FMjcfTriangleMesh
{
	/** Each distinct corner position once. */
	TArray<FVector3f> Positions;

	/** Three indices into Positions per triangle, counter-clockwise when seen from outside. */
	TArray<int32> Indices;

	int32 NumTriangles() const { return Indices.Num() / 3; }

	/**
	 * Returns one normal per entry in Indices. Each corner averages, weighted by area, the normals of the triangles around its
	 * position that face within CreaseAngleDegrees of its own triangle, so curved surfaces look smooth while sharper edges stay
	 * sharp.
	 */
	TArray<FVector3f> ComputeCornerNormals(float CreaseAngleDegrees) const;
};

/** Reads binary STL files, which store only triangle corners: no units, materials, or texture coordinates. ASCII STL is rejected. */
class SCHOLAMJCF_API FMjcfStlReader
{
public:
	/** Reads a binary STL file. Returns nothing and sets OutError if the file can't be read or is malformed. */
	static TOptional<FMjcfTriangleMesh> ReadFile(const FString& File, FString& OutError);

	/** Reads the contents of a binary STL file. */
	static TOptional<FMjcfTriangleMesh> Read(TConstArrayView<uint8> Bytes, FString& OutError);
};
