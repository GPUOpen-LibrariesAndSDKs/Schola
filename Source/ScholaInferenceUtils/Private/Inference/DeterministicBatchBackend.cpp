// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Inference/DeterministicBatchBackend.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"
#include "Math/RandomStream.h"

namespace
{
template <typename T>
void UpdateMax(std::atomic<T>& Max, T Value)
{
	for (T Current = Max.load(); Value > Current && !Max.compare_exchange_weak(Current, Value);)
	{
	}
}

void BusyWait(double Seconds)
{
	if (Seconds <= 0.0)
	{
		return;
	}
	const double End = FPlatformTime::Seconds() + Seconds;
	while (FPlatformTime::Seconds() < End)
	{
	}
}
}

class FDeterministicBatchContext : public IBatchInferenceContext
{
public:
	FDeterministicBatchContext(TSharedRef<FDeterministicBatchBackend> InBackend, int32 Index)
		: Backend(InBackend), Random(InBackend->Settings.Seed * 7919 + Index)
	{
	}

	virtual bool RunBatch(const float* Observations, int32 NumRows, float* Actions) override
	{
		FDeterministicBatchBackend& B = *Backend;
		if (bInUse.exchange(true))
		{
			B.bConcurrentContextUse.store(true);
		}
		UpdateMax(B.PeakConcurrent, B.Concurrent.fetch_add(1) + 1);

		B.Blocked.fetch_add(1);
		B.Gate->Wait();
		B.Blocked.fetch_sub(1);

		const FDeterministicBatchBackendSettings& S = B.Settings;
		BusyWait(S.FixedSeconds + S.PerRowSeconds * NumRows);
		if (S.DelayProbability > 0.0 && Random.FRand() < S.DelayProbability)
		{
			FPlatformProcess::SleepNoStats(static_cast<float>(Random.FRand() * S.MaxExtraDelaySeconds));
		}
		const bool bFail = S.FailureProbability > 0.0 && Random.FRand() < S.FailureProbability;
		if (!bFail)
		{
			for (int32 Row = 0; Row < NumRows; ++Row)
			{
				for (int32 Action = 0; Action < S.ActionSize; ++Action)
				{
					Actions[Row * S.ActionSize + Action] = FDeterministicBatchBackend::ExpectedAction(
						Observations + Row * S.ObservationSize, S.ObservationSize, Action);
				}
			}
		}

		B.Calls.fetch_add(1);
		B.Rows.fetch_add(NumRows);
		UpdateMax(B.LargestCall, NumRows);
		B.Concurrent.fetch_sub(1);
		bInUse.store(false);
		return !bFail;
	}

private:
	TSharedRef<FDeterministicBatchBackend> Backend;
	FRandomStream Random;
	std::atomic<bool> bInUse{false};
};

FDeterministicBatchBackend::FDeterministicBatchBackend(const FDeterministicBatchBackendSettings& InSettings)
	: Settings(InSettings)
{
	Gate = FPlatformProcess::GetSynchEventFromPool(true);
	Gate->Trigger();
}

FDeterministicBatchBackend::~FDeterministicBatchBackend()
{
	FPlatformProcess::ReturnSynchEventToPool(Gate);
}

TUniquePtr<IBatchInferenceContext> FDeterministicBatchBackend::CreateContext()
{
	const int32 Index = ContextsCreated.fetch_add(1);
	return MakeUnique<FDeterministicBatchContext>(AsShared(), Index);
}

void FDeterministicBatchBackend::SetGateOpen(bool bOpen)
{
	if (bOpen)
	{
		Gate->Trigger();
	}
	else
	{
		Gate->Reset();
	}
}

bool FDeterministicBatchBackend::WaitForBlockedCalls(int32 Count, double TimeoutSeconds) const
{
	const double Deadline = FPlatformTime::Seconds() + TimeoutSeconds;
	while (Blocked.load() < Count)
	{
		if (FPlatformTime::Seconds() > Deadline)
		{
			return false;
		}
		FPlatformProcess::SleepNoStats(0.0002f);
	}
	return true;
}
