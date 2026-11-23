// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Mjcf/MjcfMeshBuilder.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Engine/StaticMesh.h"
#include "Math/ScaleMatrix.h"
#include "Mjcf/MjcfAssetUtils.h"
#include "MjcfConversion.h"
#include "MjcfStl.h"
#include "StaticMeshAttributes.h"
#include "StaticMeshOperations.h"

namespace
{
	/**
	 * Interchange's OBJ translator mirrors Y and keeps the file's units and winding. Meshes built here start out the same way,
	 * so that every mesh takes the same conversion.
	 */
	FMatrix ImportedToFile()
	{
		return FScaleMatrix(FVector(1, -1, 1));
	}

	/** MuJoCo's renderer keeps edges sharper than roughly this angle. */
	constexpr float CreaseAngleDegrees = 40.0f;

	const FName MaterialSlot(TEXT("Material"));

	UStaticMesh* SaveStaticMesh(FMeshDescription&& Description, const FString& Folder, const FString& Name, bool& bOutCreated)
	{
		UStaticMesh* StaticMesh = MjcfAssetUtils::FindOrCreateAsset<UStaticMesh>(Folder, Name, bOutCreated);
		if (!StaticMesh)
		{
			return nullptr;
		}
		StaticMesh->PreEditChange(nullptr);
		StaticMesh->SetNumSourceModels(1);
		FStaticMeshSourceModel& SourceModel = StaticMesh->GetSourceModel(0);
		SourceModel.BuildSettings.bRecomputeNormals = false;
		SourceModel.BuildSettings.bRecomputeTangents = true;
		SourceModel.BuildSettings.bGenerateLightmapUVs = false;
		*StaticMesh->CreateMeshDescription(0) = MoveTemp(Description);
		StaticMesh->CommitMeshDescription(0);
		StaticMesh->GetStaticMaterials().Reset();
		StaticMesh->GetStaticMaterials().Add(FStaticMaterial(nullptr, MaterialSlot, MaterialSlot));
		StaticMesh->PostEditChange();
		StaticMesh->MarkPackageDirty();
		if (bOutCreated)
		{
			FAssetRegistryModule::AssetCreated(StaticMesh);
		}
		return StaticMesh;
	}
} // namespace

void MjcfMeshBuilder::ConvertObjMesh(UStaticMesh& StaticMesh, const FVector& MeshScale)
{
	const FMatrix Transform = ImportedToFile() * FMjcfConversion::MeshToUnreal(MeshScale);
	for (int32 Lod = 0; Lod < StaticMesh.GetNumSourceModels(); ++Lod)
	{
		if (FMeshDescription* MeshDescription = StaticMesh.GetMeshDescription(Lod))
		{
			FStaticMeshOperations::ApplyTransform(*MeshDescription, Transform, /*bApplyCorrectNormalTransform=*/true);
			StaticMesh.CommitMeshDescription(Lod);
		}
	}
	StaticMesh.PostEditChange();
}

UStaticMesh* MjcfMeshBuilder::SaveTriangleMesh(const FMjcfTriangleMesh& Mesh, const FVector& MeshScale, TConstArrayView<FVector2f> CornerUVs,
	const FString& Folder, const FString& Name, bool& bOutCreated)
{
	check(CornerUVs.IsEmpty() || CornerUVs.Num() == Mesh.Indices.Num());
	const TArray<FVector3f> Normals = Mesh.ComputeCornerNormals(CreaseAngleDegrees);

	FMeshDescription Description;
	FStaticMeshAttributes Attributes(Description);
	Attributes.Register();
	TVertexAttributesRef<FVector3f> Positions = Attributes.GetVertexPositions();
	TVertexInstanceAttributesRef<FVector3f> InstanceNormals = Attributes.GetVertexInstanceNormals();
	TVertexInstanceAttributesRef<FVector2f> UVs = Attributes.GetVertexInstanceUVs();
	const FPolygonGroupID Group = Description.CreatePolygonGroup();
	Attributes.GetPolygonGroupMaterialSlotNames()[Group] = MaterialSlot;

	Description.ReserveNewVertices(Mesh.Positions.Num());
	Description.ReserveNewVertexInstances(Mesh.Indices.Num());
	Description.ReserveNewTriangles(Mesh.NumTriangles());
	TArray<FVertexID> Vertices;
	Vertices.Reserve(Mesh.Positions.Num());
	for (const FVector3f& Position : Mesh.Positions)
	{
		const FVertexID Vertex = Description.CreateVertex();
		Positions[Vertex] = FVector3f(Position.X, -Position.Y, Position.Z);
		Vertices.Add(Vertex);
	}
	for (int32 Triangle = 0; Triangle < Mesh.NumTriangles(); ++Triangle)
	{
		FVertexInstanceID Corners[3];
		for (int32 Corner = 0; Corner < 3; ++Corner)
		{
			const int32 Index = 3 * Triangle + Corner;
			Corners[Corner] = Description.CreateVertexInstance(Vertices[Mesh.Indices[Index]]);
			InstanceNormals[Corners[Corner]] = FVector3f(Normals[Index].X, -Normals[Index].Y, Normals[Index].Z);
			// Like the OBJ translator, flip V, since Unreal's texture coordinates start at the top of the image.
			UVs.Set(Corners[Corner], 0, CornerUVs.IsEmpty() ? FVector2f::ZeroVector : FVector2f(CornerUVs[Index].X, 1.0f - CornerUVs[Index].Y));
		}
		Description.CreateTriangle(Group, Corners);
	}

	FStaticMeshOperations::ApplyTransform(Description, ImportedToFile() * FMjcfConversion::MeshToUnreal(MeshScale), /*bApplyCorrectNormalTransform=*/true);
	return SaveStaticMesh(MoveTemp(Description), Folder, Name, bOutCreated);
}

UStaticMesh* MjcfMeshBuilder::SavePlane(const FVector2D& HalfSize, const FVector2D& UVRepeat, const FString& Folder, const FString& Name, bool& bOutCreated)
{
	FMjcfTriangleMesh Plane;
	const FVector2f Half(HalfSize);
	Plane.Positions = { FVector3f(-Half.X, -Half.Y, 0), FVector3f(Half.X, -Half.Y, 0), FVector3f(Half.X, Half.Y, 0), FVector3f(-Half.X, Half.Y, 0) };
	Plane.Indices = { 0, 1, 2, 0, 2, 3 };
	TArray<FVector2f> CornerUVs;
	for (const int32 Index : Plane.Indices)
	{
		const FVector3f& Position = Plane.Positions[Index];
		CornerUVs.Add(FVector2f((Position.X / Half.X + 1) / 2 * UVRepeat.X, (Position.Y / Half.Y + 1) / 2 * UVRepeat.Y));
	}
	return SaveTriangleMesh(Plane, FVector::OneVector, CornerUVs, Folder, Name, bOutCreated);
}
