// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Factories/IREEModelDataImportFactory.h"

#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

#if SCHOLA_WITH_IREE
#include "IREEModelData.h"
#endif

// On non-Windows platforms ScholaIREE is unavailable; the factory exists (UHT generates it from the header) but imports nothing.
UIREEModelDataImportFactory::UIREEModelDataImportFactory()
{
	bEditorImport = true;
	bCreateNew = false;
	bText = false;
#if SCHOLA_WITH_IREE
	SupportedClass = UIREEModelData::StaticClass();
	Formats.Add(TEXT("vmfb;Schola IREE VM FlatBuffer"));
#else
	SupportedClass = UObject::StaticClass();
#endif
}

bool UIREEModelDataImportFactory::FactoryCanImport(const FString& Filename)
{
#if SCHOLA_WITH_IREE
	return FPaths::GetExtension(Filename).Equals(TEXT("vmfb"), ESearchCase::IgnoreCase);
#else
	return false;
#endif
}

UObject* UIREEModelDataImportFactory::FactoryCreateFile(
	UClass* InClass,
	UObject* InParent,
	FName InName,
	EObjectFlags Flags,
	const FString& Filename,
	const TCHAR* Parms,
	FFeedbackContext* Warn,
	bool& bOutOperationCanceled)
{
	bOutOperationCanceled = false;

#if !SCHOLA_WITH_IREE
	return nullptr;
#else
	TArray<uint8> VmfbBytes;
	if (!FFileHelper::LoadFileToArray(VmfbBytes, *Filename))
	{
		if (Warn)
		{
			Warn->Logf(ELogVerbosity::Error, TEXT("Failed to read IREE vmfb: %s"), *Filename);
		}
		return nullptr;
	}

	UIREEModelData* Asset = NewObject<UIREEModelData>(InParent, InClass, InName, Flags);
	Asset->VmfbBytes = MoveTemp(VmfbBytes);

	// Metadata is embedded in the vmfb as iree.reflection["schola.metadata"] (see Python save_model_as_iree).
	if (!Asset->LoadMetadataFromVmfb() && Warn)
	{
		Warn->Logf(ELogVerbosity::Warning, TEXT("No embedded schola.metadata found in %s"), *Filename);
	}

	return Asset;
#endif
}