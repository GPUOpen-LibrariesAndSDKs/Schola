// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "MjcfModel.h"

const FVector4 FMjcfGeom::DefaultRgba(0.5, 0.5, 0.5, 1);

namespace
{
	template <typename ElementType>
	const ElementType* FindByName(const TArray<ElementType>& Elements, const FString& Name)
	{
		return Elements.FindByPredicate([&Name](const ElementType& Element) { return Element.Name.Equals(Name, ESearchCase::CaseSensitive); });
	}
} // namespace

const FMjcfMesh* FMjcfModel::FindMesh(const FString& Name) const
{
	return FindByName(Meshes, Name);
}

const FMjcfTexture* FMjcfModel::FindTexture(const FString& Name) const
{
	return FindByName(Textures, Name);
}

const FMjcfMaterial* FMjcfModel::FindMaterial(const FString& Name) const
{
	return FindByName(Materials, Name);
}
