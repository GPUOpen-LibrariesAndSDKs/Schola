// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "OnnxAutoImporter.h"

#include "AutoReimport/AutoReimportManager.h"
#include "AssetImportTask.h"
#include "AssetToolsModule.h"
#include "DirectoryWatcherModule.h"
#include "HAL/FileManager.h"
#include "IAssetTools.h"
#include "IDirectoryWatcher.h"
#include "LogScholaEditor.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "NNE.h"
#include "Editor/UnrealEdEngine.h"
#include "UnrealEdGlobals.h"

namespace
{
	static const FName NneEditorModuleName(TEXT("NNEEditor"));
}

FScholaOnnxAutoImporter::~FScholaOnnxAutoImporter()
{
	Stop();
}

bool FScholaOnnxAutoImporter::IsNneImportAvailable()
{
	if (!FModuleManager::Get().IsModuleLoaded(NneEditorModuleName))
	{
		if (FModuleManager::Get().LoadModule(NneEditorModuleName) == nullptr)
		{
			return false;
		}
	}

	return UE::NNE::GetAllRuntimeNames().Num() > 0;
}

void FScholaOnnxAutoImporter::Start()
{
	if (!IsNneImportAvailable())
	{
		static bool bLoggedNneUnavailableWarning = false;
		if (!bLoggedNneUnavailableWarning)
		{
			bLoggedNneUnavailableWarning = true;
			UE_LOGFMT(LogScholaEditor, Warning,
				"FScholaOnnxAutoImporter: NNE plugin or runtime not enabled; ONNX auto-import disabled. "
				"Enable NNE and at least one NNE runtime, or import .onnx files manually via the Content Browser.");
		}
		return;
	}

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

bool FScholaOnnxAutoImporter::IsUnderWatchedContentDirectory(const FString& AbsoluteFilePath) const
{
	return FPaths::IsUnderDirectory(AbsoluteFilePath, WatchedContentDirectory);
}

void FScholaOnnxAutoImporter::QueueImport(const FString& AbsoluteFilePath)
{
	const FString NormalizedPath = FPaths::ConvertRelativePathToFull(AbsoluteFilePath);
	if (!IsUnderWatchedContentDirectory(NormalizedPath))
	{
		return;
	}

	if (const FPendingOnnxImport* Existing = PendingImports.Find(NormalizedPath))
	{
		if (Existing->Phase == EOnnxImportPhase::Importing)
		{
			return;
		}
	}

	FPendingOnnxImport& Pending = PendingImports.FindOrAdd(NormalizedPath);
	Pending.QueuedAt = FPlatformTime::Seconds();
	Pending.StableSince = 0.0;
	Pending.LastObservedSize = INDEX_NONE;
	Pending.bHasObservedSize = false;
	Pending.Phase = EOnnxImportPhase::WaitingForStableFile;
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
	TArray<FString> PathsToRemove;

	for (auto& Pair : PendingImports)
	{
		const FString& PendingPath = Pair.Key;
		FPendingOnnxImport& Pending = Pair.Value;

		if (Pending.Phase == EOnnxImportPhase::Importing)
		{
			continue;
		}

		const int64 CurrentSize = IFileManager::Get().FileSize(*PendingPath);
		if (CurrentSize == INDEX_NONE)
		{
			PathsToRemove.Add(PendingPath);
			continue;
		}

		if (!Pending.bHasObservedSize || Pending.LastObservedSize != CurrentSize)
		{
			Pending.LastObservedSize = CurrentSize;
			Pending.bHasObservedSize = true;
			Pending.StableSince = 0.0;
			continue;
		}

		if (Pending.StableSince <= 0.0)
		{
			Pending.StableSince = Now;
			continue;
		}

		if ((Now - Pending.StableSince) < ImportDebounceSeconds)
		{
			continue;
		}

		Pending.Phase = EOnnxImportPhase::Importing;
		if (ProcessImport(PendingPath))
		{
			PathsToRemove.Add(PendingPath);
		}
		else
		{
			Pending.Phase = EOnnxImportPhase::WaitingForStableFile;
			Pending.StableSince = 0.0;
			Pending.bHasObservedSize = false;
		}
	}

	for (const FString& Path : PathsToRemove)
	{
		PendingImports.Remove(Path);
	}

	return true;
}

bool FScholaOnnxAutoImporter::TryGetGameDestinationPath(
	const FString& AbsoluteFilePath,
	FString& OutDestinationPath) const
{
	const FString NormalizedPath = FPaths::ConvertRelativePathToFull(AbsoluteFilePath);
	if (!IsUnderWatchedContentDirectory(NormalizedPath))
	{
		return false;
	}

	FString RelativePath = NormalizedPath;
	if (!FPaths::MakePathRelativeTo(RelativePath, *WatchedContentDirectory))
	{
		return false;
	}

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

	FString DestinationPath;
	if (!TryGetGameDestinationPath(AbsoluteFilePath, DestinationPath))
	{
		UE_LOGFMT(LogScholaEditor, Warning,
			"FScholaOnnxAutoImporter: Skipping ONNX outside project Content: {0}", AbsoluteFilePath);
		return true;
	}

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
		return true;
	}

	UE_LOGFMT(LogScholaEditor, Warning,
		"FScholaOnnxAutoImporter: Failed to import ONNX {0} to {1}",
		AbsoluteFilePath,
		DestinationPath);
	return false;
}
