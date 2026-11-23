// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Mjcf/MjcfAssetUtils.h"
#include "InterchangeManager.h"
#include "InterchangePipelineBase.h"
#include "LogScholaEditor.h"
#include "Misc/PackageName.h"
#include "ObjectTools.h"
#include "UObject/Package.h"

FString MjcfAssetUtils::ClaimName(const FString& Name, TSet<FString>& UsedNames)
{
	FString Candidate = Name;
	for (int32 Suffix = 2; UsedNames.Contains(Candidate); ++Suffix)
	{
		Candidate = FString::Printf(TEXT("%s_%d"), *Name, Suffix);
	}
	UsedNames.Add(Candidate);
	return Candidate;
}

FString MjcfAssetUtils::SanitizeName(const FString& Name)
{
	return ObjectTools::SanitizeObjectName(Name);
}

UObject* MjcfAssetUtils::FindOrCreateAsset(UClass* Class, const FString& Folder, const FString& Name, bool& bOutCreated)
{
	bOutCreated = false;
	const FString PackageName = Folder / Name;
	UPackage* Package = FindPackage(nullptr, *PackageName);
	if (!Package && FPackageName::DoesPackageExist(PackageName))
	{
		Package = LoadPackage(nullptr, *PackageName, LOAD_None);
	}
	if (!Package)
	{
		Package = CreatePackage(*PackageName);
	}

	if (UObject* Existing = StaticFindObject(UObject::StaticClass(), Package, *Name))
	{
		if (Existing->IsA(Class))
		{
			return Existing;
		}
		UE_LOGFMT(LogScholaEditor, Error, "Could not create {0} '{1}' because a {2} already has that name", Class->GetName(), PackageName, Existing->GetClass()->GetName());
		return nullptr;
	}
	bOutCreated = true;
	return NewObject<UObject>(Package, Class, *Name, RF_Public | RF_Standalone | RF_Transactional);
}

UObject* MjcfAssetUtils::ImportWithInterchange(UClass* Class, const FString& File, const FString& Folder, const FString& Name, UInterchangePipelineBase* Pipeline)
{
	FImportAssetParameters Parameters;
	Parameters.bIsAutomated = true;
	Parameters.DestinationName = Name;
	if (Pipeline)
	{
		Parameters.OverridePipelines.Add(FSoftObjectPath(Pipeline));
	}

	const UE::Interchange::FScopedSourceData SourceData(File);
	TArray<UObject*> ImportedObjects;
	UInterchangeManager::GetInterchangeManager().ImportAsset(Folder, SourceData.GetSourceData(), Parameters, ImportedObjects);
	for (UObject* Object : ImportedObjects)
	{
		if (Object && Object->IsA(Class))
		{
			return Object;
		}
	}
	return nullptr;
}
