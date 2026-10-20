// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Inference/BatchInferenceBackend.h"
#include <atomic>

/** Configuration for FDeterministicBatchBackend. Times are busy-wait/sleep durations per backend call. */
struct FDeterministicBatchBackendSettings
{
	int32 ObservationSize = 4;
	int32 ActionSize = 2;
	/** Backend rows-per-call limit, 0 for none. */
	int32 MaxBatchSize = 0;
	/** Simulated service time: FixedSeconds + PerRowSeconds * Rows (busy wait). */
	double FixedSeconds = 0.0;
	double PerRowSeconds = 0.0;
	/** Probability that a call sleeps an extra uniform [0, MaxExtraDelaySeconds], reordering completions. */
	double DelayProbability = 0.0;
	double MaxExtraDelaySeconds = 0.0;
	/** Probability that a call fails (RunBatch returns false). */
	double FailureProbability = 0.0;
	int32 Seed = 1;
};

/**
 * Test backend with a known output function, controllable cost, delays, failures,
 * and a gate that holds every call. Records calls, rows per call, worker concurrency, and any
 * concurrent use of a single context. Not a neural network.
 *
 * Output row: Action[j] = 2 * Observation[j % ObservationSize] + j.
 */
class SCHOLAINFERENCEUTILS_API FDeterministicBatchBackend : public IBatchInferenceBackend, public TSharedFromThis<FDeterministicBatchBackend>
{
public:
	explicit FDeterministicBatchBackend(const FDeterministicBatchBackendSettings& InSettings);
	virtual ~FDeterministicBatchBackend() override;

	virtual int32 GetObservationSize() const override { return Settings.ObservationSize; }
	virtual int32 GetActionSize() const override { return Settings.ActionSize; }
	virtual int32 GetMaxBatchSize() const override { return Settings.MaxBatchSize; }
	virtual FString GetName() const override { return TEXT("Deterministic"); }
	virtual TUniquePtr<IBatchInferenceContext> CreateContext() override;

	/** Reference output for one row. */
	static float ExpectedAction(const float* ObservationRow, int32 ObservationSize, int32 ActionIndex)
	{
		return 2.0f * ObservationRow[ActionIndex % ObservationSize] + static_cast<float>(ActionIndex);
	}

	/** While closed, every RunBatch call blocks until the gate opens. Open by default. */
	void SetGateOpen(bool bOpen);
	/** Waits until at least Count calls are blocked at the closed gate. */
	bool WaitForBlockedCalls(int32 Count, double TimeoutSeconds = 5.0) const;

	int64 GetCalls() const { return Calls.load(); }
	int64 GetRows() const { return Rows.load(); }
	int32 GetBlockedCalls() const { return Blocked.load(); }
	int32 GetPeakConcurrentCalls() const { return PeakConcurrent.load(); }
	int32 GetLargestCallRows() const { return LargestCall.load(); }
	int32 GetContextsCreated() const { return ContextsCreated.load(); }
	/** True if any single context was ever entered by two threads at once. */
	bool HadConcurrentContextUse() const { return bConcurrentContextUse.load(); }

private:
	friend class FDeterministicBatchContext;
	FDeterministicBatchBackendSettings Settings;
	FEvent* Gate = nullptr;
	std::atomic<int64> Calls{0};
	std::atomic<int64> Rows{0};
	std::atomic<int32> Blocked{0};
	std::atomic<int32> Concurrent{0};
	std::atomic<int32> PeakConcurrent{0};
	std::atomic<int32> LargestCall{0};
	std::atomic<int32> ContextsCreated{0};
	std::atomic<bool> bConcurrentContextUse{false};
};
