// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

class UMaterialInterface;
class UTexture2D;
struct FMjcfGeom;
struct FMjcfMaterial;
struct FMjcfModel;
struct FMjcfTexture;

/**
 * Creates the material instances and textures that a model's geoms are drawn with, each once per import. Materials are
 * instances of Interchange's PBR materials, with MuJoCo's colors converted from sRGB.
 */
class FMjcfMaterialBuilder
{
public:
	/**
	 * @param Folder Where the assets go; materials and textures get subfolders.
	 * @param OutAssets Receives each asset when it is first used.
	 */
	FMjcfMaterialBuilder(const FMjcfModel& Model, const FString& Folder, TArray<UObject*>& OutAssets);

	/** Returns the color MuJoCo draws Geom with: its own when it differs from the default, otherwise its material's. */
	static FVector4 GeomColor(const FMjcfModel& Model, const FMjcfGeom& Geom);

	/** Returns the material to draw Geom with, or null after logging an error if it couldn't be created. */
	UMaterialInterface* GetMaterial(const FMjcfGeom& Geom);

private:
	UMaterialInterface* CreateMaterial(const FString& Name, const FMjcfMaterial* Material, const FVector4& Rgba);
	UTexture2D* GetTexture(const FString& Name);
	UTexture2D* CreateBuiltinTexture(const FMjcfTexture& Texture, const FString& AssetName);
	UTexture2D* ImportTextureFile(const FMjcfTexture& Texture, const FString& AssetName);

	const FMjcfModel& Model;
	FString MaterialFolder;
	FString TextureFolder;
	TArray<UObject*>& Assets;
	TSet<FString> UsedNames;
	TMap<FString, UMaterialInterface*> Materials;
	TMap<FString, UTexture2D*> Textures;
};
