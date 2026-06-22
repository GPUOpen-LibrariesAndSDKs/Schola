// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Containers/Ticker.h"
#include "IDirectoryWatcher.h"

/**
 * Watches the project Content directory for Schola-exported ONNX files and imports
 * them into the Content Browser when the NNE plugin and at least one NNE runtime
 * are available.
 */
class FScholaOnnxAutoImporter
{
public:
	FScholaOnnxAutoImporter() = default;
	~FScholaOnnxAutoImporter();

	void Start();
	void Stop();

private:
	enum class EOnnxImportPhase : uint8
	{
		WaitingForStableFile,
		Importing,
	};

	struct FPendingOnnxImport
	{
		double QueuedAt = 0.0;
		double StableSince = 0.0;
		int64 LastObservedSize = INDEX_NONE;
		bool bHasObservedSize = false;
		EOnnxImportPhase Phase = EOnnxImportPhase::WaitingForStableFile;
	};

	void OnDirectoryChanged(const TArray<FFileChangeData>& FileChanges);
	bool Tick(float DeltaTime);
	void QueueImport(const FString& AbsoluteFilePath);
	void SuppressAutoReimportPrompt(const FString& AbsoluteFilePath, FFileChangeData::EFileChangeAction Action) const;
	bool ProcessImport(const FString& AbsoluteFilePath);
	bool IsUnderWatchedContentDirectory(const FString& AbsoluteFilePath) const;
	bool TryGetGameDestinationPath(const FString& AbsoluteFilePath, FString& OutDestinationPath) const;
	static bool IsNneImportAvailable();

	FDelegateHandle DirectoryWatcherHandle;
	FTSTicker::FDelegateHandle TickerHandle;
	FString WatchedContentDirectory;

	TMap<FString, FPendingOnnxImport> PendingImports;

	static constexpr double ImportDebounceSeconds = 1.0;
	static constexpr float TickerIntervalSeconds = 0.25f;
};
