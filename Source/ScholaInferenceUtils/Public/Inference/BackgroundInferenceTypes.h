// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BackgroundInferenceTypes.generated.h"

/** Outcome of a Background request or submission. Every accepted request reaches exactly one terminal status. */
UENUM(BlueprintType)
enum class ENPCBackgroundStatus : uint8
{
	/** Unknown, already-consumed, or stale handle; or a malformed submission (nothing retained). */
	Invalid,
	/** Submission rejected by admission bounds (rows, bytes, or shutdown); nothing retained. Retry later. */
	Busy,
	/** Accepted and not yet terminal. */
	Pending,
	/** All rows inferred; actions available once. */
	Completed,
	/** A backend call containing at least one row failed. No actions are delivered. */
	Failed,
	/** Cancelled by the caller or by scheduler shutdown. */
	Cancelled,
	/** Exceeded its maximum age before or during execution. No actions are delivered. */
	Expired
};

/**
 * How formed batches are distributed to worker execution contexts.
 * All strategies give identical results; they differ in balance, locality, and coordination cost.
 */
UENUM(BlueprintType)
enum class ENPCBackgroundDispatch : uint8
{
	/** Workers pull the next ready batch from one queue (default; best measured under skewed cost). */
	SharedQueue,
	/** Batches formed together are split into contiguous worker ranges. */
	Contiguous,
	/** Batch k goes to worker k mod WorkerCount. */
	BatchRoundRobin,
	/** Rows are spread cyclically over workers, which form their own batches (legacy RoundRobin). */
	RowRoundRobin,
	/** Rows with the same key always run on worker key mod WorkerCount (legacy MassEntity routing). */
	KeyAffinity,
	/** Per-worker queues; idle workers steal from the most loaded queue. */
	WorkStealing,
	/** Each batch goes to the less loaded of two sampled workers by estimated queued service time. */
	LeastLoaded
};

/** Opaque handle to one submitted Background request. Zero is never a valid handle. */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FNPCBackgroundRequest
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference")
	int64 Id = 0;

	bool IsSet() const { return Id != 0; }
	bool operator==(const FNPCBackgroundRequest& Other) const { return Id == Other.Id; }
};

UENUM(BlueprintType)
enum class ENPCBackgroundWorkerPriority : uint8 { BelowNormal, Normal };

/** Scheduler configuration. */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FNPCBackgroundSchedulerSettings
{
	GENERATED_BODY()

	/** Dedicated worker threads, each owning one backend execution context. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference", meta = (ClampMin = 1, ClampMax = 64))
	int32 WorkerCount = 2;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference")
	ENPCBackgroundWorkerPriority WorkerPriority = ENPCBackgroundWorkerPriority::BelowNormal;

	/** Largest rows per backend call (also limited by the backend). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference", meta = (ClampMin = 1))
	int32 MaxBatchSize = 64;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference")
	ENPCBackgroundDispatch Dispatch = ENPCBackgroundDispatch::SharedQueue;

	/** Cap on accepted rows not yet consumed or released (pending, executing, or completed). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference", meta = (ClampMin = 1))
	int32 MaxPendingRows = 4096;

	/** Cap on observation + action bytes held by accepted, unreleased requests. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference", meta = (ClampMin = 1))
	int64 MaxPendingBytes = 8 * 1024 * 1024;

	/** Formed-but-unfinished batches allowed per worker. Rows beyond this wait and merge into larger batches. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference", meta = (ClampMin = 1))
	int32 MaxInFlightBatchesPerWorker = 2;

	/** Extra wait to fill a partial batch while capacity is free. 0 = eager (recommended). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference", meta = (ClampMin = 0))
	float MaxAssemblyWaitMs = 0.0f;

	/** Default request age limit from submission; rows older than this are expired. 0 = no limit. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference", meta = (ClampMin = 0))
	float DefaultMaxAgeMs = 0.0f;
};

/**
 * Scheduler diagnostics. Counters are cumulative since creation or the last ResetStats.
 * Logical requests, backend calls, and rows are reported separately; BackendCalls are real model invocations.
 */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FNPCBackgroundStats
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 SubmittedRequests = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 BusyRejections = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 InvalidSubmissions = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 AcceptedRows = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 CompletedRows = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 FailedRows = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 CancelledRows = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 ExpiredRows = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 BackendCalls = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 BackendReshapeCalls = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 BackendStagingGrowthEvents = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 PaddingRows = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") double BackendReshapeSeconds = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") double BackendServiceSeconds = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 BackendRows = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 FailedBackendCalls = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 StolenBatches = 0;
	/** Mean rows per backend call divided by the effective max batch size. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") float MeanBatchFill = 0.0f;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int32 PendingRows = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 PendingBytes = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int32 PeakPendingRows = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 PeakPendingBytes = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int32 PeakInFlightBatches = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int32 PeakActiveWorkers = 0;
	/** Pooled/consumer buffer capacity growth events; separate tracing checks other allocations. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int64 PoolGrowthEvents = 0;
	/** Submission to batch formation, per row (ms). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") float AssemblyWaitP99Ms = 0.0f;
	/** Largest observed submit-to-formation delay, including rows waiting for capacity. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") double AssemblyWaitMaxMs = 0.0;
	/** Batch formation to execution start, per row (ms). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") float QueueWaitP99Ms = 0.0f;
	/** Backend call duration, per call (ms). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") float ServiceP50Ms = 0.0f;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") float ServiceP99Ms = 0.0f;
	/** Submission to terminal status, per request (ms). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") float CompletionP50Ms = 0.0f;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") float CompletionP99Ms = 0.0f;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") TArray<int64> WorkerBatches;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") TArray<int64> WorkerRows;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") TArray<float> WorkerBusySeconds;
};
