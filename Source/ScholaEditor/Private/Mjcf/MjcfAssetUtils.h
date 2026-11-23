// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

class UInterchangePipelineBase;

namespace MjcfAssetUtils
{
	/**
	 * Returns Name, or Name with a numeric suffix when an earlier asset in this import already took it. UsedNames is
	 * case-insensitive, like Unreal asset names.
	 */
	FString ClaimName(const FString& Name, TSet<FString>& UsedNames);

	/** Returns Name with the characters that Unreal doesn't allow in asset names replaced. */
	FString SanitizeName(const FString& Name);

	/**
	 * Returns the asset Folder/Name of class Class, loading it if it exists on disk and creating it otherwise, so that
	 * importing the same file again updates the assets in place. Returns null after logging an error if a different kind
	 * of asset already has that name.
	 */
	UObject* FindOrCreateAsset(UClass* Class, const FString& Folder, const FString& Name, bool& bOutCreated);

	template <typename AssetType>
	AssetType* FindOrCreateAsset(const FString& Folder, const FString& Name, bool& bOutCreated)
	{
		return static_cast<AssetType*>(FindOrCreateAsset(AssetType::StaticClass(), Folder, Name, bOutCreated));
	}

	/** Imports File with Interchange as Folder/Name and returns the first asset of class AssetType it creates, or null. */
	UObject* ImportWithInterchange(UClass* Class, const FString& File, const FString& Folder, const FString& Name, UInterchangePipelineBase* Pipeline = nullptr);

	template <typename AssetType>
	AssetType* ImportWithInterchange(const FString& File, const FString& Folder, const FString& Name, UInterchangePipelineBase* Pipeline = nullptr)
	{
		return static_cast<AssetType*>(ImportWithInterchange(AssetType::StaticClass(), File, Folder, Name, Pipeline));
	}
} // namespace MjcfAssetUtils
