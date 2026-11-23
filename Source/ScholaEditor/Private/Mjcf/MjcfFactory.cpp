// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Mjcf/MjcfFactory.h"
#include "Engine/Blueprint.h"
#include "Engine/StaticMesh.h"
#include "InterchangeGenericAssetsPipeline.h"
#include "InterchangeGenericMaterialPipeline.h"
#include "InterchangeGenericMeshPipeline.h"
#include "LogScholaEditor.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "Mjcf/MjcfAssetUtils.h"
#include "Mjcf/MjcfBlueprintBuilder.h"
#include "Mjcf/MjcfMaterialBuilder.h"
#include "Mjcf/MjcfMeshBuilder.h"
#include "MjcfParser.h"
#include "MjcfStl.h"
#include "UObject/StrongObjectPtr.h"

namespace
{
	void LogDiagnostic(const FMjcfDiagnostic& Diagnostic)
	{
		switch (Diagnostic.Severity)
		{
			case EMjcfSeverity::Error:
				UE_LOGFMT(LogScholaEditor, Error, "{0}", Diagnostic.ToString());
				break;
			case EMjcfSeverity::Warning:
				UE_LOGFMT(LogScholaEditor, Warning, "{0}", Diagnostic.ToString());
				break;
			default:
				UE_LOGFMT(LogScholaEditor, Log, "{0}", Diagnostic.ToString());
				break;
		}
	}

	/**
	 * Fixed import settings, so that project import settings can't change the geometry the MJCF conversion is applied to.
	 * MuJoCo models declare collision with their own geoms, so none is generated from the visual meshes, and MuJoCo ignores
	 * the materials in OBJ files, so they aren't imported.
	 */
	TStrongObjectPtr<UInterchangeGenericAssetsPipeline> MakeImportPipeline()
	{
		TStrongObjectPtr<UInterchangeGenericAssetsPipeline> Pipeline(NewObject<UInterchangeGenericAssetsPipeline>(GetTransientPackage()));
		Pipeline->MeshPipeline->bCombineStaticMeshes = true;
		Pipeline->MeshPipeline->bCollision = false;
		Pipeline->MaterialPipeline->bImportMaterials = false;
		return Pipeline;
	}

	/** Imports the model's OBJ and STL meshes into Folder and returns them by MJCF mesh name. */
	TMap<FString, UStaticMesh*> ImportMeshes(const FMjcfModel& Model, const FString& Folder, TSet<FString>& UsedNames, TArray<UObject*>& OutAssets)
	{
		const TStrongObjectPtr<UInterchangeGenericAssetsPipeline> Pipeline = MakeImportPipeline();
		TMap<FString, UStaticMesh*> Meshes;
		for (const FMjcfMesh& Mesh : Model.Meshes)
		{
			const FString Extension = FPaths::GetExtension(Mesh.File);
			const bool bObj = Extension.Equals(TEXT("obj"), ESearchCase::IgnoreCase);
			if (!bObj && !Extension.Equals(TEXT("stl"), ESearchCase::IgnoreCase))
			{
				UE_LOGFMT(LogScholaEditor, Warning, "Skipped mesh \"{0}\": only OBJ and STL meshes are supported, got '{1}'", Mesh.Name, Mesh.File);
				continue;
			}
			if (!FPaths::FileExists(Mesh.File))
			{
				UE_LOGFMT(LogScholaEditor, Error, "Could not import mesh \"{0}\" because '{1}' does not exist", Mesh.Name, Mesh.File);
				continue;
			}

			const FString SanitizedName = MjcfAssetUtils::SanitizeName(Mesh.Name);
			const FString AssetName = MjcfAssetUtils::ClaimName(SanitizedName, UsedNames);
			if (AssetName != SanitizedName)
			{
				UE_LOGFMT(LogScholaEditor, Warning, "Imported mesh \"{0}\" as \"{1}\" because another mesh already uses the asset name \"{2}\"", Mesh.Name, AssetName, SanitizedName);
			}

			UStaticMesh* StaticMesh = nullptr;
			if (bObj)
			{
				StaticMesh = MjcfAssetUtils::ImportWithInterchange<UStaticMesh>(Mesh.File, Folder, AssetName, Pipeline.Get());
				if (!StaticMesh)
				{
					UE_LOGFMT(LogScholaEditor, Error, "Could not import mesh \"{0}\" because Interchange created no static mesh from '{1}'", Mesh.Name, Mesh.File);
					continue;
				}
				MjcfMeshBuilder::ConvertObjMesh(*StaticMesh, Mesh.Scale);
			}
			else
			{
				FString Error;
				const TOptional<FMjcfTriangleMesh> Triangles = FMjcfStlReader::ReadFile(Mesh.File, Error);
				if (!Triangles)
				{
					UE_LOGFMT(LogScholaEditor, Error, "Could not import mesh \"{0}\" from '{1}': {2}", Mesh.Name, Mesh.File, Error);
					continue;
				}
				bool bCreated = false;
				StaticMesh = MjcfMeshBuilder::SaveTriangleMesh(*Triangles, Mesh.Scale, {}, Folder, AssetName, bCreated);
				if (!StaticMesh)
				{
					continue;
				}
			}
			Meshes.Add(Mesh.Name, StaticMesh);
			OutAssets.Add(StaticMesh);
		}
		return Meshes;
	}
} // namespace

UMjcfFactory::UMjcfFactory()
{
	SupportedClass = UBlueprint::StaticClass();
	bCreateNew = false;
	bEditorImport = true;
	Formats.Add(TEXT("mjcf;MuJoCo Model"));
	Formats.Add(TEXT("xml;MuJoCo Model"));
}

bool UMjcfFactory::FactoryCanImport(const FString& Filename)
{
	return FMjcfParser::IsMjcfFile(Filename);
}

UObject* UMjcfFactory::FactoryCreateFile(UClass* InClass, UObject* InParent, FName InName, EObjectFlags Flags, const FString& Filename, const TCHAR* Parms, FFeedbackContext* Warn, bool& bOutOperationCanceled)
{
	bOutOperationCanceled = false;
	// Asset Tools only clears this after all dropped files are imported.
	AdditionalImportedObjects.Reset();

	TArray<FMjcfDiagnostic> Diagnostics;
	const TOptional<FMjcfModel> Model = FMjcfParser::ParseFile(Filename, Diagnostics);
	for (const FMjcfDiagnostic& Diagnostic : Diagnostics)
	{
		LogDiagnostic(Diagnostic);
	}
	if (!Model)
	{
		UE_LOGFMT(LogScholaEditor, Error, "Could not import '{0}' because the MJCF file has errors", Filename);
		return nullptr;
	}

	// The Blueprint goes where the file was dropped, and everything it uses goes in a folder named after the file.
	const FString Folder = FPackageName::GetLongPackagePath(InParent->GetPackage()->GetName()) / InName.ToString();
	TSet<FString> UsedNames;
	TArray<UObject*> Assets;
	const TMap<FString, UStaticMesh*> Meshes = ImportMeshes(*Model, Folder, UsedNames, Assets);
	FMjcfMaterialBuilder Materials(*Model, Folder, Assets);
	UBlueprint* Blueprint = FMjcfBlueprintBuilder(*Model, Meshes, Materials, Folder, UsedNames, Assets).Build(InParent, InName);
	if (!Blueprint)
	{
		return nullptr;
	}
	AdditionalImportedObjects = MoveTemp(Assets);
	return Blueprint;
}
