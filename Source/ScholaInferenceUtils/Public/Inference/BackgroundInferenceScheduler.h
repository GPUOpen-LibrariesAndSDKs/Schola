// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Inference/BackgroundInferenceTypes.h"
#include "Inference/BatchInferenceBackend.h"

/** Per-submission options. */
struct FNPCBackgroundSubmitOptions
{
	/** Optional stable identity per row (NPC id or packed Mass handle). Empty, or exactly one per row. Used by KeyAffinity routing and returned with the result. */
	TConstArrayView<int64> Keys;
	/** Age limit from submission. Negative uses the scheduler default; 0 disables expiry. */
	float MaxAgeMs = -1.0f;
	/** Caller data returned with the result (for example, a world/config generation). */
	uint64 UserTag = 0;
};

/** Terminal outcome of a consumed request. Buffers are swapped in, so reusing one result object avoids allocation. */
struct FNPCBackgroundResult
{
	ENPCBackgroundStatus Status = ENPCBackgroundStatus::Invalid;
	int32 Rows = 0;
	int32 ActionSize = 0;
	/** Rows * ActionSize floats, row-major, in submission order. Only written when Status is Completed. */
	TArray<float> Actions;
	/** Keys as submitted (empty if none). */
	TArray<int64> Keys;
	/** Worker index that executed each row, or INDEX_NONE for rows that never executed. */
	TArray<int32> RowWorkers;
	uint64 UserTag = 0;
	/** FPlatformTime::Seconds() at submission and at terminal status. */
	double SubmitSeconds = 0.0;
	double CompleteSeconds = 0.0;
};

class FNPCBackgroundSchedulerImpl;

/**
 * Bounded, batched Background inference service for one compatible model bucket.
 *
 * Callers submit rows of float observations and later consume actions without blocking. Rows from
 * many submissions are merged into tensor batches: a batch forms as soon as a worker slot is free
 * (eager, work-conserving); while all in-flight batch slots are used, arriving rows accumulate and
 * form larger batches. Batches are distributed to dedicated worker threads according to the
 * configured ENPCBackgroundDispatch strategy. Each worker owns one backend context.
 *
 * Bounds: accepted-but-unreleased rows and bytes, and formed-but-unfinished batches, never exceed
 * their caps; submissions beyond them return Busy immediately. Every accepted request reaches exactly
 * one terminal status and its result is delivered at most once.
 *
 * Thread safety: all public methods may be called from any thread. Wait() blocks and must not be
 * used on the game thread in production. Destroying the scheduler (or Shutdown) joins its workers
 * and cancels unfinished requests.
 */
class SCHOLAINFERENCEUTILS_API FNPCBackgroundScheduler
{
public:
	/**
	 * Validates settings, creates one backend context per worker on the calling thread, and starts the workers.
	 * @return Null with OutError set if settings are invalid or a context cannot be created.
	 */
	static TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Create(const FNPCBackgroundSchedulerSettings& Settings,
		const TSharedRef<IBatchInferenceBackend>& Backend, FString* OutError = nullptr);

	~FNPCBackgroundScheduler();

	/**
	 * Copies Rows * ObservationSize floats and queues them.
	 * @return Pending with OutRequest set, Busy if bounds are full or the scheduler is shut down, or Invalid for malformed input.
	 */
	ENPCBackgroundStatus Submit(TConstArrayView<float> Observations, const FNPCBackgroundSubmitOptions& Options, FNPCBackgroundRequest& OutRequest);

	/**
	 * Reserves a pooled buffer for Rows observations and calls Fill to write them in place
	 * (Rows * ObservationSize floats), avoiding an intermediate copy. Fill runs on the calling
	 * thread without the scheduler lock held. Same return values as Submit.
	 */
	ENPCBackgroundStatus SubmitWith(int32 Rows, TFunctionRef<void(TArrayView<float>)> Fill,
		const FNPCBackgroundSubmitOptions& Options, FNPCBackgroundRequest& OutRequest);

	/** Pending, a terminal status, or Invalid for unknown/consumed handles. Does not consume. */
	ENPCBackgroundStatus GetStatus(FNPCBackgroundRequest Request) const;

	/**
	 * If the request is terminal, moves its outcome into OutResult and retires the handle; later calls return Invalid.
	 * If Pending, returns Pending and leaves OutResult unchanged.
	 */
	ENPCBackgroundStatus Consume(FNPCBackgroundRequest Request, FNPCBackgroundResult& OutResult);

	/**
	 * Retires a request without delivering it. Pending rows are skipped if not yet executing; executing
	 * rows finish and are discarded. Returns false for unknown handles.
	 */
	bool Cancel(FNPCBackgroundRequest Request);

	/** Blocks until the request is terminal or the timeout expires (negative = no timeout). Never call on the game thread. */
	ENPCBackgroundStatus Wait(FNPCBackgroundRequest Request, double TimeoutSeconds = -1.0) const;

	/** Appends each newly terminal, unconsumed request once, in completion order. Returns the number appended. */
	int32 DrainCompleted(TArray<FNPCBackgroundRequest>& OutRequests);

	/** True if a submission of Rows would currently be admitted. */
	bool CanAccept(int32 Rows) const;

	FNPCBackgroundStats GetStats() const;
	int64 GetBackendCallCount() const;
	void ResetStats();

	/** Joins workers and cancels every unfinished request. Idempotent. */
	void Shutdown();
	bool IsShuttingDown() const;

	int32 GetObservationSize() const;
	int32 GetActionSize() const;
	int32 GetWorkerCount() const;
	/** Effective rows-per-call cap (settings and backend). */
	int32 GetMaxBatchSize() const;
	const FNPCBackgroundSchedulerSettings& GetSettings() const;

private:
	FNPCBackgroundScheduler() = default;
	TUniquePtr<FNPCBackgroundSchedulerImpl> Impl;
};
