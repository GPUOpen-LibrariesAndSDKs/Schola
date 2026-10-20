// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

/**
 * One mutable execution context (for example, one NNE model instance plus its bindings).
 * A context is owned and used by exactly one Background worker thread; it is never called concurrently.
 */
class IBatchInferenceContext
{
public:
	virtual ~IBatchInferenceContext() = default;

	/**
	 * Runs the model once on a tensor batch.
	 * @param Observations Rows * ObservationSize floats, row-major.
	 * @param Rows Number of rows; 1 <= Rows <= the backend's max batch size (if any).
	 * @param Actions Receives Rows * ActionSize floats, row-major.
	 * @return False if the call failed; the scheduler then fails every request with rows in this batch.
	 */
	virtual bool RunBatch(const float* Observations, int32 Rows, float* Actions) = 0;
};

struct FBatchInferenceBackendStats
{
    int64 ReshapeCalls = 0, PaddingRows = 0;
    double ReshapeSeconds = 0, ServiceSeconds = 0;
    /** Growth of provider-owned tensor buffers; excludes allocations inside the runtime. */
    int64 StagingGrowthEvents = 0;
};

/**
 * Factory and description for one compatible model bucket. Rows submitted to one scheduler must
 * all match this backend's observation/action sizes.
 */
class IBatchInferenceBackend
{
public:
	virtual ~IBatchInferenceBackend() = default;

	/** Floats per observation row. Must be positive. */
	virtual int32 GetObservationSize() const = 0;
	/** Floats per action row. Must be positive. */
	virtual int32 GetActionSize() const = 0;
	/** Largest supported rows per call, or 0 for no backend limit. */
	virtual int32 GetMaxBatchSize() const { return 0; }
	/** Diagnostic name for logs and stats. */
	virtual FString GetName() const = 0;
	virtual FBatchInferenceBackendStats GetBackendStats() const { return {}; }

	/**
	 * Creates one execution context. Called on the thread that creates the scheduler (normally the
	 * game thread), once per worker, before any worker starts.
	 * @return Null on failure; scheduler creation then fails.
	 */
	virtual TUniquePtr<IBatchInferenceContext> CreateContext() = 0;
};
