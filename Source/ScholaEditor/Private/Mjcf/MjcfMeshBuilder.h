// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

class UStaticMesh;
struct FMjcfTriangleMesh;

/** Converts meshes to Unreal and creates static mesh assets for meshes that Interchange doesn't import, such as STL. */
namespace MjcfMeshBuilder
{
	/** Replaces Interchange's OBJ axis conversion with the MJCF one, which also converts meters and applies the mesh scale. */
	void ConvertObjMesh(UStaticMesh& StaticMesh, const FVector& MeshScale);

	/**
	 * Creates the static mesh asset Folder/Name from triangles in MuJoCo's units and axes, or replaces its geometry if it
	 * exists. CornerUVs holds one texture coordinate per entry in Mesh.Indices, or is empty.
	 */
	UStaticMesh* SaveTriangleMesh(const FMjcfTriangleMesh& Mesh, const FVector& MeshScale, TConstArrayView<FVector2f> CornerUVs,
		const FString& Folder, const FString& Name, bool& bOutCreated);

	/**
	 * Creates the static mesh asset for a plane geom facing +Z, HalfSize meters across each way, whose texture coordinates
	 * run from 0 to UVRepeat across it.
	 */
	UStaticMesh* SavePlane(const FVector2D& HalfSize, const FVector2D& UVRepeat, const FString& Folder, const FString& Name, bool& bOutCreated);
} // namespace MjcfMeshBuilder
