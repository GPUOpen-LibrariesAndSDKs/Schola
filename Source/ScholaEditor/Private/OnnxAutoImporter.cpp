// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "OnnxAutoImporter.h"

#include "AutoReimport/AutoReimportManager.h"
#include "AssetImportTask.h"
#include "AssetToolsModule.h"
#include "DirectoryWatcherModule.h"
#include "Factories/Factory.h"
#include "HAL/FileManager.h"
#include "IAssetTools.h"
#include "IDirectoryWatcher.h"
#include "LogScholaEditor.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "NNE.h"
#include "Editor/UnrealEdEngine.h"
#include "UObject/UObjectGlobals.h"
#include "UnrealEdGlobals.h"

namespace ScholaOnnxAutoImporter
{
	static const TCHAR* NneEditorModuleNames[] = {
		TEXT("NNEEditor"),
		TEXT("NNEEditorTools"),
	};
}

FScholaOnnxAutoImporter::~FScholaOnnxAutoImporter()
{
	Stop();
}

void FScholaOnnxAutoImporter::Start()
{
	if (!DirectoryWatcherHandle.IsValid())
	{
		FDirectoryWatcherModule& DirectoryWatcherModule =
			FModuleManager::LoadModuleChecked<FDirectoryWatcherModule>(TEXT("DirectoryWatcher"));
		IDirectoryWatcher* DirectoryWatcher = DirectoryWatcherModule.Get();
		if (!DirectoryWatcher)
		{
			UE_LOGFMT(LogScholaEditor, Warning,
				"FScholaOnnxAutoImporter::Start(): DirectoryWatcher unavailable; ONNX auto-import disabled");
			return;
		}

		WatchedContentDirectory = FPaths::ConvertRelativePathToFull(FPaths::ProjectContentDir());
		DirectoryWatcher->RegisterDirectoryChangedCallback_Handle(
			WatchedContentDirectory,
			IDirectoryWatcher::FDirectoryChanged::CreateRaw(this, &FScholaOnnxAutoImporter::OnDirectoryChanged),
			DirectoryWatcherHandle);
	}

	if (!TickerHandle.IsValid())
	{
		TickerHandle = FTSTicker::GetCoreTicker().AddTicker(
			FTickerDelegate::CreateRaw(this, &FScholaOnnxAutoImporter::Tick),
			TickerIntervalSeconds);
	}

	UE_LOGFMT(LogScholaEditor, Log,
		"FScholaOnnxAutoImporter::Start(): Watching {0} for ONNX exports", WatchedContentDirectory);
}

void FScholaOnnxAutoImporter::Stop()
{
	if (TickerHandle.IsValid())
	{
		FTSTicker::GetCoreTicker().RemoveTicker(TickerHandle);
		TickerHandle.Reset();
	}

	if (DirectoryWatcherHandle.IsValid())
	{
		if (FModuleManager::Get().IsModuleLoaded(TEXT("DirectoryWatcher")))
		{
			FDirectoryWatcherModule& DirectoryWatcherModule =
				FModuleManager::GetModuleChecked<FDirectoryWatcherModule>(TEXT("DirectoryWatcher"));
			if (IDirectoryWatcher* DirectoryWatcher = DirectoryWatcherModule.Get())
			{
				DirectoryWatcher->UnregisterDirectoryChangedCallback_Handle(
					WatchedContentDirectory,
					DirectoryWatcherHandle);
			}
		}
		DirectoryWatcherHandle.Reset();
	}

	PendingImports.Empty();
	InFlightImports.Empty();
	PendingImportTimes.Empty();
	PendingFileSizes.Empty();
}

void FScholaOnnxAutoImporter::OnDirectoryChanged(const TArray<FFileChangeData>& FileChanges)
{
	for (const FFileChangeData& Change : FileChanges)
	{
		if (Change.Action == FFileChangeData::FCA_Modified
			|| Change.Action == FFileChangeData::FCA_Added)
		{
			const FString AbsolutePath = FPaths::ConvertRelativePathToFull(Change.Filename);
			if (FPaths::GetExtension(AbsolutePath).Equals(TEXT("onnx"), ESearchCase::IgnoreCase))
			{
				SuppressAutoReimportPrompt(AbsolutePath, Change.Action);
				QueueImport(AbsolutePath);
			}
		}
	}
}

void FScholaOnnxAutoImporter::QueueImport(const FString& AbsoluteFilePath)
{
	if (!AbsoluteFilePath.StartsWith(WatchedContentDirectory))
	{
		return;
	}

	if (InFlightImports.Contains(AbsoluteFilePath))
	{
		return;
	}

	PendingImports.Add(AbsoluteFilePath);
	PendingImportTimes.FindOrAdd(AbsoluteFilePath) = FPlatformTime::Seconds();
	PendingFileSizes.Remove(AbsoluteFilePath);
}

void FScholaOnnxAutoImporter::SuppressAutoReimportPrompt(
	const FString& AbsoluteFilePath,
	FFileChangeData::EFileChangeAction Action) const
{
	if (!GUnrealEd || !GUnrealEd->AutoReimportManager)
	{
		return;
	}

	switch (Action)
	{
	case FFileChangeData::FCA_Added:
		GUnrealEd->AutoReimportManager->IgnoreNewFile(AbsoluteFilePath);
		break;
	case FFileChangeData::FCA_Modified:
		GUnrealEd->AutoReimportManager->IgnoreFileModification(AbsoluteFilePath);
		break;
	default:
		break;
	}
}

bool FScholaOnnxAutoImporter::Tick(float DeltaTime)
{
	if (PendingImports.Num() == 0)
	{
		return true;
	}

	const double Now = FPlatformTime::Seconds();
	TArray<FString> ReadyImports;
	for (const FString& PendingPath : PendingImports)
	{
		const double* QueuedAt = PendingImportTimes.Find(PendingPath);
		if (!QueuedAt)
		{
			continue;
		}

		double QueuedAtValue = *QueuedAt;
		if (!IsOnnxFileStable(PendingPath, QueuedAtValue))
		{
			PendingImportTimes[PendingPath] = QueuedAtValue;
			continue;
		}

		if ((Now - QueuedAtValue) < ImportDebounceSeconds)
		{
			continue;
		}

		ReadyImports.Add(PendingPath);
	}

	for (const FString& ReadyPath : ReadyImports)
	{
		PendingImports.Remove(ReadyPath);
		PendingImportTimes.Remove(ReadyPath);
		PendingFileSizes.Remove(ReadyPath);

		if (ProcessImport(ReadyPath))
		{
			InFlightImports.Remove(ReadyPath);
		}
	}

	return true;
}

bool FScholaOnnxAutoImporter::IsOnnxFileStable(const FString& AbsoluteFilePath, double& InOutQueuedAt)
{
	const int64 CurrentSize = IFileManager::Get().FileSize(*AbsoluteFilePath);
	if (CurrentSize == INDEX_NONE)
	{
		return false;
	}

	const int64* PreviousSize = PendingFileSizes.Find(AbsoluteFilePath);
	if (!PreviousSize)
	{
		PendingFileSizes.Add(AbsoluteFilePath, CurrentSize);
		return false;
	}

	if (*PreviousSize != CurrentSize)
	{
		PendingFileSizes[AbsoluteFilePath] = CurrentSize;
		InOutQueuedAt = FPlatformTime::Seconds();
		return false;
	}

	return true;
}

bool FScholaOnnxAutoImporter::TryGetGameDestinationPath(
	const FString& AbsoluteFilePath,
	FString& OutDestinationPath) const
{
	if (!AbsoluteFilePath.StartsWith(WatchedContentDirectory))
	{
		return false;
	}

	FString RelativePath = AbsoluteFilePath.Mid(WatchedContentDirectory.Len());
	FPaths::NormalizeFilename(RelativePath);
	const FString RelativeDir = FPaths::GetPath(RelativePath).Replace(TEXT("\\"), TEXT("/"));

	OutDestinationPath = RelativeDir.IsEmpty() ? TEXT("/Game") : FString::Printf(TEXT("/Game/%s"), *RelativeDir);
	return true;
}

bool FScholaOnnxAutoImporter::ProcessImport(const FString& AbsoluteFilePath)
{
	if (!FPaths::FileExists(AbsoluteFilePath))
	{
		return true;
	}

	for (const TCHAR* ModuleName : ScholaOnnxAutoImporter::NneEditorModuleNames)
	{
		if (!FModuleManager::Get().IsModuleLoaded(ModuleName))
		{
			FModuleManager::Get().LoadModule(ModuleName);
		}
	}

	FString DestinationPath;
	if (!TryGetGameDestinationPath(AbsoluteFilePath, DestinationPath))
	{
		UE_LOGFMT(LogScholaEditor, Warning,
			"FScholaOnnxAutoImporter: Skipping ONNX outside project Content: {0}", AbsoluteFilePath);
		return true;
	}

	InFlightImports.Add(AbsoluteFilePath);

	UAssetImportTask* ImportTask = NewObject<UAssetImportTask>();
	ImportTask->Filename = AbsoluteFilePath;
	ImportTask->DestinationPath = DestinationPath;
	ImportTask->bReplaceExisting = true;
	ImportTask->bAutomated = true;
	ImportTask->bSave = true;
	ImportTask->Factory = nullptr;

	IAssetTools& AssetTools = FModuleManager::LoadModuleChecked<FAssetToolsModule>(TEXT("AssetTools")).Get();
	AssetTools.ImportAssetTasks({ ImportTask });

	if (ImportTask->ImportedObjectPaths.Num() > 0)
	{
		UE_LOGFMT(LogScholaEditor, Log,
			"FScholaOnnxAutoImporter: Imported ONNX {0} to {1}",
			AbsoluteFilePath,
			ImportTask->ImportedObjectPaths[0]);
	}
	else
	{
		UE_LOGFMT(LogScholaEditor, Warning,
			"FScholaOnnxAutoImporter: Failed to import ONNX {0} to {1}",
			AbsoluteFilePath,
			DestinationPath);
	}

	return true;
}
