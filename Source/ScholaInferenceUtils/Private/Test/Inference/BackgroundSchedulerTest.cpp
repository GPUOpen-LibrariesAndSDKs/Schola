// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Inference/BackgroundInferenceScheduler.h"
#include "Inference/DeterministicBatchBackend.h"
#include "Inference/LatencyHistogram.h"
#include "HAL/RunnableThread.h"
#include "HAL/ThreadManager.h"
#include "Math/RandomStream.h"

// Scheduler correctness, admission, and steady-state allocation tests.

namespace
{
using FSchedulerPtr = TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe>;

const ENPCBackgroundDispatch AllStrategies[] = {
	ENPCBackgroundDispatch::SharedQueue, ENPCBackgroundDispatch::Contiguous, ENPCBackgroundDispatch::BatchRoundRobin,
	ENPCBackgroundDispatch::RowRoundRobin, ENPCBackgroundDispatch::KeyAffinity, ENPCBackgroundDispatch::WorkStealing,
	ENPCBackgroundDispatch::LeastLoaded};

FString StrategyName(ENPCBackgroundDispatch Dispatch)
{
	return StaticEnum<ENPCBackgroundDispatch>()->GetNameStringByValue(static_cast<int64>(Dispatch));
}

TSharedRef<FDeterministicBatchBackend> MakeBackend(const FDeterministicBatchBackendSettings& Settings = {})
{
	return MakeShared<FDeterministicBatchBackend>(Settings);
}

FSchedulerPtr MakeScheduler(const FNPCBackgroundSchedulerSettings& Settings, const TSharedRef<FDeterministicBatchBackend>& Backend)
{
	return FNPCBackgroundScheduler::Create(Settings, StaticCastSharedRef<IBatchInferenceBackend>(Backend));
}

/** Observations whose values encode (request, row, column), so misrouted rows produce wrong outputs. */
void MakeObservations(int32 RequestIndex, int32 Rows, int32 ObsSize, TArray<float>& Out)
{
	Out.SetNumUninitialized(Rows * ObsSize);
	for (int32 Row = 0; Row < Rows; ++Row)
	{
		for (int32 Col = 0; Col < ObsSize; ++Col)
		{
			Out[Row * ObsSize + Col] = static_cast<float>((RequestIndex % 100000) * 0.5 + Row * 0.25 + Col);
		}
	}
}

bool OutputsMatch(const TArray<float>& Observations, const FNPCBackgroundResult& Result, int32 ObsSize)
{
	if (Result.Actions.Num() != Result.Rows * Result.ActionSize)
	{
		return false;
	}
	for (int32 Row = 0; Row < Result.Rows; ++Row)
	{
		for (int32 Action = 0; Action < Result.ActionSize; ++Action)
		{
			const float Expected = FDeterministicBatchBackend::ExpectedAction(&Observations[Row * ObsSize], ObsSize, Action);
			if (Result.Actions[Row * Result.ActionSize + Action] != Expected)
			{
				return false;
			}
		}
	}
	return true;
}

bool WaitAllTerminal(const FSchedulerPtr& Scheduler, const TArray<FNPCBackgroundRequest>& Requests, double TimeoutSeconds = 10.0)
{
	for (const FNPCBackgroundRequest& Request : Requests)
	{
		if (Scheduler->Wait(Request, TimeoutSeconds) == ENPCBackgroundStatus::Pending)
		{
			return false;
		}
	}
	return true;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSchedulerStrategiesTest, "Schola.Inference.Background.Scheduler.StrategiesCorrect",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundSchedulerStrategiesTest::RunTest(const FString& Parameters)
{
	for (ENPCBackgroundDispatch Dispatch : AllStrategies)
	{
		const FString Name = StrategyName(Dispatch);
		FDeterministicBatchBackendSettings BackendSettings;
		BackendSettings.ObservationSize = 5;
		BackendSettings.ActionSize = 3;
		BackendSettings.PerRowSeconds = 2e-6;
		BackendSettings.DelayProbability = 0.05;
		BackendSettings.MaxExtraDelaySeconds = 0.0005;
		auto Backend = MakeBackend(BackendSettings);
		FNPCBackgroundSchedulerSettings Settings;
		Settings.WorkerCount = 4;
		Settings.MaxBatchSize = 16;
		Settings.Dispatch = Dispatch;
		// Results are consumed only at the end, and unconsumed rows count against the caps.
		Settings.MaxPendingRows = 1 << 16;
		Settings.MaxPendingBytes = int64(1) << 24;
		FSchedulerPtr Scheduler = MakeScheduler(Settings, Backend);
		if (!TestTrue(Name + TEXT(": created"), Scheduler.IsValid()))
		{
			continue;
		}

		FRandomStream Random(17);
		TArray<TArray<float>> Inputs;
		TArray<TArray<int64>> Keys;
		TArray<FNPCBackgroundRequest> Requests;
		for (int32 Index = 0; Index < 400; ++Index)
		{
			const int32 Rows = Random.RandRange(1, 40);
			TArray<float>& Observations = Inputs.AddDefaulted_GetRef();
			MakeObservations(Index, Rows, 5, Observations);
			TArray<int64>& RowKeys = Keys.AddDefaulted_GetRef();
			for (int32 Row = 0; Row < Rows; ++Row)
			{
				RowKeys.Add(Random.RandRange(0, 1000000));
			}
			FNPCBackgroundSubmitOptions Options;
			Options.Keys = RowKeys;
			Options.UserTag = Index;
			FNPCBackgroundRequest Request;
			ENPCBackgroundStatus Status = Scheduler->Submit(Observations, Options, Request);
			const double RetryDeadline = FPlatformTime::Seconds() + 10.0;
			while (Status == ENPCBackgroundStatus::Busy && FPlatformTime::Seconds() < RetryDeadline)
			{
				FPlatformProcess::SleepNoStats(0.0002f);
				Status = Scheduler->Submit(Observations, Options, Request);
			}
			TestEqual(Name + TEXT(": accepted"), Status, ENPCBackgroundStatus::Pending);
			Requests.Add(Request);
		}
		TestTrue(Name + TEXT(": all terminal"), WaitAllTerminal(Scheduler, Requests));

		bool bAllCorrect = true;
		bool bAffinityHeld = true;
		bool bAllRowsExecuted = true;
		FNPCBackgroundResult Result;
		for (int32 Index = 0; Index < Requests.Num(); ++Index)
		{
			const ENPCBackgroundStatus Status = Scheduler->Consume(Requests[Index], Result);
			bAllCorrect &= Status == ENPCBackgroundStatus::Completed && Result.UserTag == static_cast<uint64>(Index)
				&& OutputsMatch(Inputs[Index], Result, 5) && Result.Keys == Keys[Index];
			for (int32 Row = 0; Row < Result.Rows; ++Row)
			{
				bAllRowsExecuted &= Result.RowWorkers[Row] >= 0 && Result.RowWorkers[Row] < 4;
				if (Dispatch == ENPCBackgroundDispatch::KeyAffinity)
				{
					bAffinityHeld &= Result.RowWorkers[Row] == static_cast<int32>(Keys[Index][Row] % 4);
				}
			}
			TestEqual(Name + TEXT(": consumed once"), Scheduler->Consume(Requests[Index], Result), ENPCBackgroundStatus::Invalid);
		}
		const FNPCBackgroundStats Stats = Scheduler->GetStats();
		TestTrue(Name + TEXT(": outputs, tags and keys match submission order"), bAllCorrect);
		TestTrue(Name + TEXT(": every row executed on a valid worker"), bAllRowsExecuted);
		TestTrue(Name + TEXT(": key affinity routes key mod workers"), bAffinityHeld);
		TestEqual(Name + TEXT(": backend rows equal accepted rows"), Backend->GetRows(), Stats.AcceptedRows);
		TestTrue(Name + TEXT(": rows per call within max batch"), Backend->GetLargestCallRows() <= 16);
		TestTrue(Name + TEXT(": multi-row backend calls"), Backend->GetLargestCallRows() > 1);
		TestTrue(Name + TEXT(": active workers within cap"), Backend->GetPeakConcurrentCalls() <= 4 && Stats.PeakActiveWorkers <= 4);
		TestFalse(Name + TEXT(": no context used concurrently"), Backend->HadConcurrentContextUse());
		TestEqual(Name + TEXT(": one context per worker"), Backend->GetContextsCreated(), 4);
		AddInfo(FString::Printf(TEXT("%s: calls=%lld fill=%.2f stolen=%lld busyRejects=%lld"), *Name, Stats.BackendCalls,
			Stats.MeanBatchFill, Stats.StolenBatches, Stats.BusyRejections));
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSchedulerFaultInjectionTest, "Schola.Inference.Background.Scheduler.RandomizedFaultInjection",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundSchedulerFaultInjectionTest::RunTest(const FString& Parameters)
{
	constexpr int32 RequestsPerStrategy = 100000;
	constexpr int32 ObsSize = 3;
	for (ENPCBackgroundDispatch Dispatch : AllStrategies)
	{
		const FString Name = StrategyName(Dispatch);
		FDeterministicBatchBackendSettings BackendSettings;
		BackendSettings.ObservationSize = ObsSize;
		BackendSettings.ActionSize = 2;
		BackendSettings.FailureProbability = 0.01;
		BackendSettings.DelayProbability = 0.002;
		BackendSettings.MaxExtraDelaySeconds = 0.0003;
		BackendSettings.Seed = static_cast<int32>(Dispatch) + 3;
		auto Backend = MakeBackend(BackendSettings);
		FNPCBackgroundSchedulerSettings Settings;
		Settings.WorkerCount = 4;
		Settings.MaxBatchSize = 32;
		Settings.MaxPendingRows = 512;
		Settings.MaxPendingBytes = 512 * (ObsSize + 2) * sizeof(float);
		Settings.Dispatch = Dispatch;
		FSchedulerPtr Scheduler = MakeScheduler(Settings, Backend);
		if (!TestTrue(Name + TEXT(": created"), Scheduler.IsValid()))
		{
			continue;
		}

		struct FTracked
		{
			TArray<float> Observations;
			int32 Index = 0;
		};
		TMap<int64, FTracked> Live;
		FRandomStream Random(1234 + static_cast<int32>(Dispatch));
		int32 Submitted = 0;
		int64 Duplicates = 0, WrongOutputs = 0, Terminal = 0, Cancelled = 0, Busy = 0;
		int64 StatusCounts[7] = {};
		TArray<FNPCBackgroundRequest> Done;
		FNPCBackgroundResult Result;
		const double Deadline = FPlatformTime::Seconds() + 120.0;

		while ((Submitted < RequestsPerStrategy || Live.Num() > 0) && FPlatformTime::Seconds() < Deadline)
		{
			for (int32 Burst = Random.RandRange(1, 64); Burst > 0 && Submitted < RequestsPerStrategy; --Burst)
			{
				FTracked Tracked;
				Tracked.Index = Submitted;
				const int32 Rows = Random.RandRange(1, 12);
				MakeObservations(Submitted, Rows, ObsSize, Tracked.Observations);
				TArray<int64, TInlineAllocator<12>> RowKeys;
				for (int32 Row = 0; Row < Rows; ++Row)
				{
					RowKeys.Add(Random.RandRange(0, 64));
				}
				FNPCBackgroundSubmitOptions Options;
				Options.Keys = RowKeys;
				Options.MaxAgeMs = Random.FRand() < 0.1f ? 0.05f : 0.0f;
				FNPCBackgroundRequest Request;
				const ENPCBackgroundStatus Status = Scheduler->Submit(Tracked.Observations, Options, Request);
				if (Status == ENPCBackgroundStatus::Busy)
				{
					++Busy;
					break;
				}
				TestEqual(Name + TEXT(": accepted or busy"), Status, ENPCBackgroundStatus::Pending);
				++Submitted;
				if (Random.FRand() < 0.05f)
				{
					// Cancelled requests retire without delivery and must never be drained.
					TestTrue(Name + TEXT(": cancel accepted"), Scheduler->Cancel(Request));
					++Cancelled;
					continue;
				}
				Live.Add(Request.Id, MoveTemp(Tracked));
			}

			Done.Reset();
			Scheduler->DrainCompleted(Done);
			for (const FNPCBackgroundRequest& Request : Done)
			{
				FTracked Tracked;
				if (!Live.RemoveAndCopyValue(Request.Id, Tracked))
				{
					++Duplicates;
					continue;
				}
				const ENPCBackgroundStatus Status = Scheduler->Consume(Request, Result);
				++Terminal;
				++StatusCounts[static_cast<int32>(Status)];
				if (Status == ENPCBackgroundStatus::Completed && !OutputsMatch(Tracked.Observations, Result, ObsSize))
				{
					++WrongOutputs;
				}
				if (Status == ENPCBackgroundStatus::Pending || Status == ENPCBackgroundStatus::Invalid)
				{
					++Duplicates;
				}
			}
			if (Done.Num() == 0)
			{
				FPlatformProcess::YieldThread();
			}
		}

		// Cancel retires the caller's ticket immediately, but executing rows still
		// own capacity until the backend returns. Live only tracks deliverable
		// tickets, so its becoming empty is not a worker-drain barrier.
		while (!Scheduler->CanAccept(Settings.MaxPendingRows) && FPlatformTime::Seconds() < Deadline)
		{
			FPlatformProcess::SleepNoStats(0.0001f);
		}
		const FNPCBackgroundStats Stats = Scheduler->GetStats();
		TestEqual(Name + TEXT(": every request submitted"), Submitted, RequestsPerStrategy);
		TestEqual(Name + TEXT(": every live request reached exactly one terminal status"), Live.Num(), 0);
		TestEqual(Name + TEXT(": no duplicate or unexpected deliveries"), Duplicates, int64(0));
		TestEqual(Name + TEXT(": completed outputs correct"), WrongOutputs, int64(0));
		TestEqual(Name + TEXT(": terminal + cancelled = submitted"), Terminal + Cancelled, int64(Submitted));
		TestEqual(Name + TEXT(": row accounting balances"),
			Stats.CompletedRows + Stats.FailedRows + Stats.CancelledRows + Stats.ExpiredRows, Stats.AcceptedRows);
		TestTrue(Name + TEXT(": pending-row cap held"), Stats.PeakPendingRows <= Settings.MaxPendingRows);
		TestTrue(Name + TEXT(": pending-byte cap held"), Stats.PeakPendingBytes <= Settings.MaxPendingBytes);
		TestTrue(Name + TEXT(": in-flight batch cap held"), Stats.PeakInFlightBatches <= Settings.WorkerCount * Settings.MaxInFlightBatchesPerWorker);
		TestTrue(Name + TEXT(": active worker cap held"), Stats.PeakActiveWorkers <= Settings.WorkerCount && Backend->GetPeakConcurrentCalls() <= Settings.WorkerCount);
		TestFalse(Name + TEXT(": no context used concurrently"), Backend->HadConcurrentContextUse());
		TestTrue(Name + TEXT(": failures were injected"), StatusCounts[static_cast<int32>(ENPCBackgroundStatus::Failed)] > 0);
		TestEqual(Name + TEXT(": nothing left pending"), Stats.PendingRows, 0);
		AddInfo(FString::Printf(TEXT("%s: completed=%lld failed=%lld expired=%lld cancelled=%lld busy=%lld calls=%lld fill=%.2f"),
			*Name, StatusCounts[static_cast<int32>(ENPCBackgroundStatus::Completed)], StatusCounts[static_cast<int32>(ENPCBackgroundStatus::Failed)],
			StatusCounts[static_cast<int32>(ENPCBackgroundStatus::Expired)], Cancelled, Busy, Stats.BackendCalls, Stats.MeanBatchFill));
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSchedulerEagerMergeTest, "Schola.Inference.Background.Scheduler.EagerBatchesGrowUnderLoad",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundSchedulerEagerMergeTest::RunTest(const FString& Parameters)
{
	auto Backend = MakeBackend();
	FNPCBackgroundSchedulerSettings Settings;
	Settings.WorkerCount = 1;
	Settings.MaxInFlightBatchesPerWorker = 1;
	FSchedulerPtr Scheduler = MakeScheduler(Settings, Backend);
	Backend->SetGateOpen(false);

	TArray<FNPCBackgroundRequest> Requests;
	TArray<TArray<float>> Inputs;
	auto SubmitOne = [&](int32 Index)
	{
		MakeObservations(Index, 1, 4, Inputs.AddDefaulted_GetRef());
		Scheduler->Submit(Inputs.Last(), {}, Requests.AddDefaulted_GetRef());
	};
	SubmitOne(0);
	TestTrue(TEXT("First request dispatched immediately (no assembly wait)"), Backend->WaitForBlockedCalls(1));
	for (int32 Index = 1; Index <= 10; ++Index)
	{
		SubmitOne(Index);
	}
	TestEqual(TEXT("Arrivals wait while the only slot is busy"), Backend->GetCalls(), int64(0));
	Backend->SetGateOpen(true);
	TestTrue(TEXT("All complete"), WaitAllTerminal(Scheduler, Requests));
	TestEqual(TEXT("Ten waiting single-row submissions merged into one call"), Backend->GetCalls(), int64(2));
	TestEqual(TEXT("Merged call size"), Backend->GetLargestCallRows(), 10);
	FNPCBackgroundResult Result;
	bool bCorrect = true;
	for (int32 Index = 0; Index < Requests.Num(); ++Index)
	{
		bCorrect &= Scheduler->Consume(Requests[Index], Result) == ENPCBackgroundStatus::Completed && OutputsMatch(Inputs[Index], Result, 4);
	}
	TestTrue(TEXT("Each merged request receives its own rows"), bCorrect);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSchedulerAdmissionTest, "Schola.Inference.Background.Scheduler.AdmissionBounds",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundSchedulerAdmissionTest::RunTest(const FString& Parameters)
{
	auto Backend = MakeBackend();
	FNPCBackgroundSchedulerSettings Settings;
	Settings.WorkerCount = 1;
	Settings.MaxPendingRows = 8;
	FSchedulerPtr Scheduler = MakeScheduler(Settings, Backend);
	Backend->SetGateOpen(false);

	TArray<float> Eight, One, Bad;
	MakeObservations(0, 8, 4, Eight);
	MakeObservations(1, 1, 4, One);
	MakeObservations(2, 9, 4, Bad);
	FNPCBackgroundRequest Accepted, Rejected;
	TestEqual(TEXT("Fills the row cap"), Scheduler->Submit(Eight, {}, Accepted), ENPCBackgroundStatus::Pending);
	TestFalse(TEXT("CanAccept reflects the cap"), Scheduler->CanAccept(1));

	TArray<double> Latencies;
	for (int32 Trial = 0; Trial < 1000; ++Trial)
	{
		const double Start = FPlatformTime::Seconds();
		const ENPCBackgroundStatus Status = Scheduler->Submit(One, {}, Rejected);
		Latencies.Add(FPlatformTime::Seconds() - Start);
		if (Status != ENPCBackgroundStatus::Busy)
		{
			AddError(TEXT("Submission beyond the cap was not rejected as Busy"));
			break;
		}
	}
	Latencies.Sort();
	const double P99Ms = Latencies[FMath::Min(Latencies.Num() - 1, static_cast<int32>(Latencies.Num() * 0.99))] * 1000.0;
	TestTrue(FString::Printf(TEXT("Busy admission p99 %.4f ms <= 0.2 ms"), P99Ms), P99Ms <= 0.2);
	TestFalse(TEXT("Rejected submission returns no handle"), Rejected.IsSet());

	FNPCBackgroundRequest Ignored;
	TestEqual(TEXT("Larger than the cap is invalid"), Scheduler->Submit(Bad, {}, Ignored), ENPCBackgroundStatus::Invalid);
	TestEqual(TEXT("Empty submission is invalid"), Scheduler->Submit(TArray<float>(), {}, Ignored), ENPCBackgroundStatus::Invalid);
	TestEqual(TEXT("Partial row is invalid"), Scheduler->Submit(TArray<float>{1.0f, 2.0f}, {}, Ignored), ENPCBackgroundStatus::Invalid);
	FNPCBackgroundSubmitOptions BadKeys;
	const int64 OneKey[] = {1};
	BadKeys.Keys = OneKey;
	TestEqual(TEXT("Key count must match rows"), Scheduler->Submit(Eight, BadKeys, Ignored), ENPCBackgroundStatus::Invalid);

	FNPCBackgroundResult Result;
	Result.Actions = {42.0f};
	TestEqual(TEXT("Pending consume"), Scheduler->Consume(Accepted, Result), ENPCBackgroundStatus::Pending);
	TestTrue(TEXT("Pending consume leaves result untouched"), Result.Actions == TArray<float>{42.0f});
	TestEqual(TEXT("Unknown handle"), Scheduler->GetStatus(FNPCBackgroundRequest{12345}), ENPCBackgroundStatus::Invalid);

	Backend->SetGateOpen(true);
	TestEqual(TEXT("Completes"), Scheduler->Wait(Accepted, 5.0), ENPCBackgroundStatus::Completed);
	TestEqual(TEXT("Completed rows still count until consumed"), Scheduler->GetStats().PendingRows, 8);
	TestEqual(TEXT("Consume"), Scheduler->Consume(Accepted, Result), ENPCBackgroundStatus::Completed);
	TestTrue(TEXT("Capacity released after consume"), Scheduler->CanAccept(8));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSchedulerStatusTest, "Schola.Inference.Background.Scheduler.TerminalStatuses",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundSchedulerStatusTest::RunTest(const FString& Parameters)
{
	TArray<float> Row;
	MakeObservations(0, 1, 4, Row);
	FNPCBackgroundResult Result;
	{
		auto Backend = MakeBackend();
		FNPCBackgroundSchedulerSettings Settings;
		Settings.WorkerCount = 1;
		Settings.MaxPendingRows = 1;
		auto Scheduler = MakeScheduler(Settings, Backend);
		Backend->SetGateOpen(false);
		FNPCBackgroundRequest Running;
		Scheduler->Submit(Row, {}, Running);
		TestTrue(TEXT("Cancellation regression holds the backend in flight"), Backend->WaitForBlockedCalls(1));
		TestTrue(TEXT("Cancel running work"), Scheduler->Cancel(Running));
		TestEqual(TEXT("Ticket is retired before the backend finishes"), Scheduler->GetStatus(Running), ENPCBackgroundStatus::Invalid);
		TestFalse(TEXT("Cancelled execution still owns capacity"), Scheduler->CanAccept(1));
		TestEqual(TEXT("Cancelled rows are counted at actual retirement"), Scheduler->GetStats().CancelledRows, int64(0));
		Backend->SetGateOpen(true);
		const double Deadline = FPlatformTime::Seconds() + 5;
		while (!Scheduler->CanAccept(1) && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(0.0001f);
		TestTrue(TEXT("Backend completion releases cancelled capacity"), Scheduler->CanAccept(1));
		TestEqual(TEXT("Exactly one cancelled row accounted after drain"), Scheduler->GetStats().CancelledRows, int64(1));
		TArray<FNPCBackgroundRequest> Done;
		Scheduler->DrainCompleted(Done);
		TestTrue(TEXT("Cancelled work is never delivered"), Done.IsEmpty());
	}
	{
		auto Backend = MakeBackend();
		FNPCBackgroundSchedulerSettings Settings;
		Settings.WorkerCount = 1;
		Settings.MaxInFlightBatchesPerWorker = 1;
		FSchedulerPtr Scheduler = MakeScheduler(Settings, Backend);
		Backend->SetGateOpen(false);
		FNPCBackgroundRequest Running, Expiring, Cancelled;
		Scheduler->Submit(Row, {}, Running);
		Backend->WaitForBlockedCalls(1);
		FNPCBackgroundSubmitOptions ShortLived;
		ShortLived.MaxAgeMs = 1.0f;
		Scheduler->Submit(Row, ShortLived, Expiring);
		Scheduler->Submit(Row, {}, Cancelled);
		TestTrue(TEXT("Cancel pending"), Scheduler->Cancel(Cancelled));
		TestEqual(TEXT("Cancelled handle is retired"), Scheduler->GetStatus(Cancelled), ENPCBackgroundStatus::Invalid);
		FPlatformProcess::SleepNoStats(0.005f);
		Backend->SetGateOpen(true);
		TestEqual(TEXT("Running completes"), Scheduler->Wait(Running, 5.0), ENPCBackgroundStatus::Completed);
		TestEqual(TEXT("Over-age request expires"), Scheduler->Wait(Expiring, 5.0), ENPCBackgroundStatus::Expired);
		TArray<FNPCBackgroundRequest> Done;
		Scheduler->DrainCompleted(Done);
		TestTrue(TEXT("Drain reports running and expired only"), Done.Num() == 2 && Done.Contains(Running) && Done.Contains(Expiring));
		Result.Actions = {7.0f};
		TestEqual(TEXT("Consume expired"), Scheduler->Consume(Expiring, Result), ENPCBackgroundStatus::Expired);
		TestTrue(TEXT("Expired delivers no actions"), Result.Actions == TArray<float>{7.0f});
		TestTrue(TEXT("Expired row never executed"), Result.RowWorkers == TArray<int32>{INDEX_NONE});
		TestEqual(TEXT("Only the running row reached the backend"), Backend->GetRows(), int64(1));
		TestEqual(TEXT("Cancelled cannot be consumed"), Scheduler->Consume(Cancelled, Result), ENPCBackgroundStatus::Invalid);
	}
	{
		FDeterministicBatchBackendSettings Failing;
		Failing.FailureProbability = 1.0;
		auto Backend = MakeBackend(Failing);
		FSchedulerPtr Scheduler = MakeScheduler(FNPCBackgroundSchedulerSettings(), Backend);
		FNPCBackgroundRequest Request;
		Scheduler->Submit(Row, {}, Request);
		TestEqual(TEXT("Backend failure"), Scheduler->Wait(Request, 5.0), ENPCBackgroundStatus::Failed);
		Result.Actions = {9.0f};
		TestEqual(TEXT("Consume failed"), Scheduler->Consume(Request, Result), ENPCBackgroundStatus::Failed);
		TestTrue(TEXT("Failure leaves caller actions unchanged"), Result.Actions == TArray<float>{9.0f});
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSchedulerLegacyLanesTest, "Schola.Inference.Background.Scheduler.LegacyLaneMappings",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundSchedulerLegacyLanesTest::RunTest(const FString& Parameters)
{
	// Matches the existing mock policy's documented lanes for 3 rows, batch 2, 2 workers.
	struct FCase
	{
		ENPCBackgroundDispatch Dispatch;
		TArray<int64> Keys;
		TArray<int32> Lanes;
	};
	const FCase Cases[] = {
		{ENPCBackgroundDispatch::Contiguous, {}, {0, 0, 1}},
		{ENPCBackgroundDispatch::RowRoundRobin, {}, {0, 1, 0}},
		{ENPCBackgroundDispatch::KeyAffinity, {7, 8, 9}, {1, 0, 1}},
	};
	TArray<float> Observations;
	MakeObservations(0, 3, 4, Observations);
	for (const FCase& Case : Cases)
	{
		auto Backend = MakeBackend();
		FNPCBackgroundSchedulerSettings Settings;
		Settings.WorkerCount = 2;
		Settings.MaxBatchSize = 2;
		Settings.Dispatch = Case.Dispatch;
		FSchedulerPtr Scheduler = MakeScheduler(Settings, Backend);
		FNPCBackgroundSubmitOptions Options;
		Options.Keys = Case.Keys;
		FNPCBackgroundRequest Request;
		Scheduler->Submit(Observations, Options, Request);
		FNPCBackgroundResult Result;
		Scheduler->Wait(Request, 5.0);
		Scheduler->Consume(Request, Result);
		TestTrue(StrategyName(Case.Dispatch) + TEXT(" lanes"), Result.RowWorkers == Case.Lanes);
		TestEqual(StrategyName(Case.Dispatch) + TEXT(" backend calls"), Backend->GetCalls(), int64(2));
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSchedulerSteadyStateTest, "Schola.Inference.Background.Scheduler.SteadyStateNoAllocation",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundSchedulerSteadyStateTest::RunTest(const FString& Parameters)
{
	for (ENPCBackgroundDispatch Dispatch : AllStrategies)
	{
		auto Backend = MakeBackend();
		FNPCBackgroundSchedulerSettings Settings;
		Settings.WorkerCount = 2;
		Settings.MaxBatchSize = 32;
		Settings.MaxPendingRows = 256;
		Settings.Dispatch = Dispatch;
		FSchedulerPtr Scheduler = MakeScheduler(Settings, Backend);
		const int64 Keys[8] = {1, 2, 3, 4, 5, 6, 7, 8};
		FNPCBackgroundSubmitOptions Options;
		Options.Keys = Keys;
		FNPCBackgroundResult Result;
		TArray<FNPCBackgroundRequest> Done;
		Done.Reserve(1024);
		// Saturating cycles: submit until Busy, then drain everything, with a fixed request shape.
		auto RunCycles = [&](int32 Cycles)
		{
			for (int32 Cycle = 0; Cycle < Cycles; ++Cycle)
			{
				FNPCBackgroundRequest Request;
				while (Scheduler->SubmitWith(8, [](TArrayView<float> Rows) { for (float& V : Rows) { V = 1.0f; } }, Options, Request)
					== ENPCBackgroundStatus::Pending)
				{
				}
				while (Scheduler->GetStats().PendingRows > 0)
				{
					Done.Reset();
					Scheduler->DrainCompleted(Done);
					for (const FNPCBackgroundRequest& Completed : Done)
					{
						Scheduler->Consume(Completed, Result);
					}
					FPlatformProcess::YieldThread();
				}
			}
		};
		RunCycles(200);
		const int64 GrowthAfterWarmup = Scheduler->GetStats().PoolGrowthEvents;
		RunCycles(1000);
		const FNPCBackgroundStats Stats = Scheduler->GetStats();
		TestEqual(StrategyName(Dispatch) + TEXT(": no pool growth after warm-up"), Stats.PoolGrowthEvents, GrowthAfterWarmup);
		AddInfo(FString::Printf(TEXT("%s: growth events during warm-up=%lld rows=%lld"), *StrategyName(Dispatch), GrowthAfterWarmup, Stats.CompletedRows));
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSchedulerShutdownTest, "Schola.Inference.Background.Scheduler.ShutdownWithWorkInFlight",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundSchedulerShutdownTest::RunTest(const FString& Parameters)
{
	FDeterministicBatchBackendSettings Slow;
	Slow.FixedSeconds = 0.001;
	auto Backend = MakeBackend(Slow);
	FNPCBackgroundSchedulerSettings Settings;
	Settings.WorkerCount = 2;
	Settings.MaxBatchSize = 4;
	FSchedulerPtr Scheduler = MakeScheduler(Settings, Backend);
	TArray<float> Rows;
	MakeObservations(0, 4, 4, Rows);
	TArray<FNPCBackgroundRequest> Requests;
	for (int32 Index = 0; Index < 50; ++Index)
	{
		Scheduler->Submit(Rows, {}, Requests.AddDefaulted_GetRef());
	}
	Scheduler->Shutdown();
	int32 Completed = 0, Cancelled = 0, Other = 0;
	FNPCBackgroundResult Result;
	for (const FNPCBackgroundRequest& Request : Requests)
	{
		switch (Scheduler->Consume(Request, Result))
		{
		case ENPCBackgroundStatus::Completed: ++Completed; break;
		case ENPCBackgroundStatus::Cancelled: ++Cancelled; break;
		default: ++Other; break;
		}
	}
	TestEqual(TEXT("Every request is terminal after shutdown"), Other, 0);
	TestTrue(TEXT("Unstarted work was cancelled"), Cancelled > 0);
	FNPCBackgroundRequest Late;
	TestEqual(TEXT("Submissions after shutdown are Busy"), Scheduler->Submit(Rows, {}, Late), ENPCBackgroundStatus::Busy);
	Scheduler->Shutdown();
	AddInfo(FString::Printf(TEXT("completed=%d cancelled=%d"), Completed, Cancelled));
	return true;
}

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundPollingNotificationsTest, "Schola.Inference.Background.Scheduler.PollingDoesNotRetainNotifications",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundPollingNotificationsTest::RunTest(const FString&)
{
    auto Backend = MakeBackend();
    FNPCBackgroundSchedulerSettings Settings;
    Settings.MaxPendingRows = 16;
    auto Scheduler = MakeScheduler(Settings, Backend);
    TArray<float> Rows; MakeObservations(0,1,4,Rows);
    FNPCBackgroundResult Result;
    int64 WarmGrowth = 0;
    for (int32 I = 0; I < 1000; ++I)
    {
        FNPCBackgroundRequest Request;
        if (Scheduler->Submit(Rows,{},Request) != ENPCBackgroundStatus::Pending ||
            Scheduler->Wait(Request,5) != ENPCBackgroundStatus::Completed ||
            Scheduler->Consume(Request,Result) != ENPCBackgroundStatus::Completed) return false;
        if (I == 99) WarmGrowth = Scheduler->GetStats().PoolGrowthEvents;
    }
    TestEqual(TEXT("Poll-only consumer has bounded notification storage"),Scheduler->GetStats().PoolGrowthEvents,WarmGrowth);
    TArray<FNPCBackgroundRequest> Notifications;
    TestEqual(TEXT("Consumed requests have no notifications"),Scheduler->DrainCompleted(Notifications),0);
    return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundResultCapacityTest,"Schola.Inference.Background.Scheduler.ConsumeRetainsLargestWarmBuffers",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundResultCapacityTest::RunTest(const FString&)
{
    FDeterministicBatchBackendSettings BackendSettings;BackendSettings.ObservationSize=BackendSettings.ActionSize=1;
    auto Backend=MakeShared<FDeterministicBatchBackend>(BackendSettings);
    FNPCBackgroundSchedulerSettings Settings;Settings.WorkerCount=1;Settings.MaxBatchSize=64;
    auto Scheduler=FNPCBackgroundScheduler::Create(Settings,Backend);
    TArray<float> Input;Input.Init(1.f,64);
    TArray<int64> Keys;for(int32 I=0;I<64;++I)Keys.Add(I);
    FNPCBackgroundResult Result;
    auto Run=[&](int32 Rows)
    {
        FNPCBackgroundSubmitOptions Options;Options.Keys=MakeArrayView(Keys).Slice(0,Rows);
        FNPCBackgroundRequest Ticket;
        return Scheduler->Submit(MakeArrayView(Input).Slice(0,Rows),Options,Ticket)==ENPCBackgroundStatus::Pending &&
            Scheduler->Wait(Ticket,5)==ENPCBackgroundStatus::Completed &&
            Scheduler->Consume(Ticket,Result)==ENPCBackgroundStatus::Completed;
    };
    TestTrue(TEXT("Warm smaller buffers"),Run(32));
    TestTrue(TEXT("Warm maximum shape and consume it"),Run(64));
    Scheduler->ResetStats();
    TestTrue(TEXT("Same warmed maximum shape completes"),Run(64));
    TestEqual(TEXT("Consuming a result cannot shrink pooled request capacity"),Scheduler->GetStats().PoolGrowthEvents,int64(0));
    TestEqual(TEXT("Actions remain complete"),Result.Actions.Num(),64);
    TestEqual(TEXT("Keys remain ordered"),Result.Keys.Last(),int64(63));
    return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSchedulerWorkerPriorityTest, "Schola.Inference.Background.Scheduler.WorkerPriority",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundSchedulerWorkerPriorityTest::RunTest(const FString& Parameters)
{
	auto WorkerPriorities = []()
	{
		TMap<uint32, EThreadPriority> Out;
		FThreadManager::Get().ForEachThread([&Out](uint32 Id, FRunnableThread* Thread)
		{
			if (Thread->GetThreadName().StartsWith(TEXT("NPCBackgroundWorker"))) Out.Add(Id, Thread->GetThreadPriority());
		});
		return Out;
	};
	for (const ENPCBackgroundWorkerPriority Priority : {ENPCBackgroundWorkerPriority::BelowNormal, ENPCBackgroundWorkerPriority::Normal})
	{
		const TMap<uint32, EThreadPriority> Before = WorkerPriorities();
		FNPCBackgroundSchedulerSettings Settings;
		Settings.WorkerCount = 3;
		Settings.WorkerPriority = Priority;
		FSchedulerPtr Scheduler = MakeScheduler(Settings, MakeBackend());
		int32 Created = 0;
		const EThreadPriority Expected = Priority == ENPCBackgroundWorkerPriority::BelowNormal ? TPri_BelowNormal : TPri_Normal;
		for (const TPair<uint32, EThreadPriority>& Worker : WorkerPriorities())
		{
			if (Before.Contains(Worker.Key)) continue;
			++Created;
			TestEqual(TEXT("Worker thread priority follows WorkerPriority"), int32(Worker.Value), int32(Expected));
		}
		TestEqual(TEXT("One thread per worker"), Created, 3);
		Scheduler->Shutdown();
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSchedulerAssemblyWaitTest, "Schola.Inference.Background.Scheduler.AssemblyWaitFillsPartialBatch",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundSchedulerAssemblyWaitTest::RunTest(const FString& Parameters)
{
	auto Backend = MakeBackend();
	FNPCBackgroundSchedulerSettings Settings;
	Settings.WorkerCount = 1;
	Settings.MaxBatchSize = 4;
	Settings.MaxAssemblyWaitMs = 200.0f;
	FSchedulerPtr Scheduler = MakeScheduler(Settings, Backend);
	TArray<FNPCBackgroundRequest> Requests;
	TArray<TArray<float>> Inputs;
	auto SubmitOne = [&](int32 Index)
	{
		MakeObservations(Index, 1, 4, Inputs.AddDefaulted_GetRef());
		Scheduler->Submit(Inputs.Last(), {}, Requests.AddDefaulted_GetRef());
	};
	SubmitOne(0);
	FPlatformProcess::SleepNoStats(0.02f);
	TestEqual(TEXT("Partial batch waits for more rows while capacity is free"), Backend->GetCalls(), int64(0));
	for (int32 Index = 1; Index < 4; ++Index) SubmitOne(Index);
	TestTrue(TEXT("Full batch completes"), WaitAllTerminal(Scheduler, Requests));
	TestEqual(TEXT("Rows arriving within the wait share one call"), Backend->GetCalls(), int64(1));
	TestEqual(TEXT("Call is full"), Backend->GetLargestCallRows(), 4);

	// A lone row is not held forever: it dispatches once the wait elapses.
	const double Start = FPlatformTime::Seconds();
	SubmitOne(4);
	TestTrue(TEXT("Lone row completes"), Scheduler->Wait(Requests.Last(), 5.0) == ENPCBackgroundStatus::Completed);
	const double Elapsed = FPlatformTime::Seconds() - Start;
	TestTrue(FString::Printf(TEXT("Lone row waited for the assembly window (%.1f ms)"), Elapsed * 1000.0), Elapsed >= 0.15);
	TestEqual(TEXT("Lone row ran by itself"), Backend->GetCalls(), int64(2));
	TestTrue(TEXT("Assembly wait is reported"), Scheduler->GetStats().AssemblyWaitMaxMs >= 150.0);
	FNPCBackgroundResult Result;
	bool bCorrect = true;
	for (int32 Index = 0; Index < Requests.Num(); ++Index)
	{
		bCorrect &= Scheduler->Consume(Requests[Index], Result) == ENPCBackgroundStatus::Completed && OutputsMatch(Inputs[Index], Result, 4);
	}
	TestTrue(TEXT("Each request receives its own rows"), bCorrect);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundLatencyHistogramTest, "Schola.Inference.Background.Telemetry.HistogramQuantiles",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FBackgroundLatencyHistogramTest::RunTest(const FString& Parameters)
{
	FLatencyHistogram Histogram;
	TestEqual(TEXT("Empty quantile"), Histogram.GetQuantile(0.99), 0.0);
	// 1..1000 microseconds, one sample each.
	for (int32 Micros = 1; Micros <= 1000; ++Micros) Histogram.Record(Micros * 1e-6);
	TestEqual(TEXT("Count"), Histogram.GetCount(), int64(1000));
	TestEqual(TEXT("Max"), Histogram.GetMax(), 1000 * 1e-6);
	auto Near = [&](const TCHAR* What, double Actual, double Expected)
	{
		TestTrue(FString::Printf(TEXT("%s: %.3f us vs %.3f us"), What, Actual * 1e6, Expected * 1e6), FMath::Abs(Actual - Expected) <= Expected * 0.011);
	};
	Near(TEXT("p50"), Histogram.GetQuantile(0.5), 500e-6);
	Near(TEXT("p99"), Histogram.GetQuantile(0.99), 990e-6);
	Near(TEXT("p0 is the smallest sample"), Histogram.GetQuantile(0.0), 1e-6);
	TestTrue(TEXT("p100 never exceeds max"), Histogram.GetQuantile(1.0) <= Histogram.GetMax());
	Histogram.Record(500e-6, 9000);
	Near(TEXT("Weighted samples dominate the median"), Histogram.GetQuantile(0.5), 500e-6);
	Histogram.Record(0.0);
	Histogram.Record(1e9);
	TestTrue(TEXT("Out-of-range samples are clamped, not dropped"), Histogram.GetCount() == 10002 && Histogram.GetMax() == 1e9);
	Histogram.Reset();
	TestTrue(TEXT("Reset clears"), Histogram.GetCount() == 0 && Histogram.GetMax() == 0.0 && Histogram.GetQuantile(0.5) == 0.0);
	return true;
}
#endif
