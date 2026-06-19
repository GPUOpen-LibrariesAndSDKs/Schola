// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Containers/Ticker.h"
#include "IDirectoryWatcher.h"

/**
 * Watches the project Content directory for Schola-exported ONNX files and imports
 * them into the Content Browser when NNE editor import support is available.
 */
class FScholaOnnxAutoImporter
{
public:
	FScholaOnnxAutoImporter() = default;
	~FScholaOnnxAutoImporter();

	void Start();
	void Stop();

private:
	void OnDirectoryChanged(const TArray<FFileChangeData>& FileChanges);
	bool Tick(float DeltaTime);
	void QueueImport(const FString& AbsoluteFilePath);
	void SuppressAutoReimportPrompt(const FString& AbsoluteFilePath, FFileChangeData::EFileChangeAction Action) const;
	bool ProcessImport(const FString& AbsoluteFilePath);
	bool IsOnnxFileStable(const FString& AbsoluteFilePath, double& InOutQueuedAt);
	bool TryGetGameDestinationPath(const FString& AbsoluteFilePath, FString& OutDestinationPath) const;

	FDelegateHandle DirectoryWatcherHandle;
	FTSTicker::FDelegateHandle TickerHandle;
	FString WatchedContentDirectory;

	/** ONNX files waiting for debounce / write stability before import. */
	TSet<FString> PendingImports;

	/** ONNX files currently being imported to suppress duplicate work. */
	TSet<FString> InFlightImports;

	/** Last filesystem notification time per ONNX file. */
	TMap<FString, double> PendingImportTimes;

	/** Last observed file size while waiting for write completion. */
	TMap<FString, int64> PendingFileSizes;

	static constexpr double ImportDebounceSeconds = 1.0;
	static constexpr float TickerIntervalSeconds = 0.25f;
};
