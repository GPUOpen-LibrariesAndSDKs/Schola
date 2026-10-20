// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Inference/BackgroundInferenceScheduler.h"
#include "Inference/LatencyHistogram.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "Math/RandomStream.h"
#include "Misc/ScopeLock.h"
#include "LogScholaInferenceUtils.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

namespace
{
/** Separate wait and hold scopes make scheduler contention visible in Insights. */
class FNPCSchedulerScopeLock
{
public:
    explicit FNPCSchedulerScopeLock(FCriticalSection* InMutex) : Mutex(InMutex)
    {
        { TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola.Bg.Scheduler.LockWait"); Mutex->Lock(); }
#if CPUPROFILERTRACE_ENABLED
        static uint32 Spec = 0;
        Hold.Emplace(Spec, TEXT("Schola.Bg.Scheduler.LockHold"), true, __FILE__, __LINE__);
#endif
    }
    ~FNPCSchedulerScopeLock()
    {
#if CPUPROFILERTRACE_ENABLED
        Hold.Reset();
#endif
        Mutex->Unlock();
    }
private:
    FCriticalSection* Mutex;
#if CPUPROFILERTRACE_ENABLED
    TOptional<FCpuProfilerTrace::FEventScope> Hold;
#endif
};
/** Growable FIFO/LIFO ring. Grow() reports capacity growth so hot-loop allocations are observable. */
template <typename T>
class TRing
{
public:
	int32 Num() const { return Count; }
	bool IsEmpty() const { return Count == 0; }
	T& Front() { return Data[Head]; }
	T& Back() { return Data[(Head + Count - 1) & (Data.Num() - 1)]; }

	/** @return true if storage had to grow. */
	bool PushBack(const T& Value)
	{
		const bool bGrew = EnsureSpace();
		Data[(Head + Count) & (Data.Num() - 1)] = Value;
		++Count;
		return bGrew;
	}

	T PopFront()
	{
		T Value = Data[Head];
		Head = (Head + 1) & (Data.Num() - 1);
		--Count;
		return Value;
	}

	T PopBack()
	{
		--Count;
		return Data[(Head + Count) & (Data.Num() - 1)];
	}

	void Reserve(int32 Capacity)
	{
		while (Data.Num() < Capacity)
		{
			Grow();
		}
	}

private:
	bool EnsureSpace()
	{
		if (Count < Data.Num())
		{
			return false;
		}
		Grow();
		return true;
	}

	void Grow()
	{
		const int32 NewCapacity = FMath::Max(16, Data.Num() * 2);
		TArray<T> NewData;
		NewData.SetNum(NewCapacity);
		for (int32 Index = 0; Index < Count; ++Index)
		{
			NewData[Index] = Data[(Head + Index) & (Data.Num() - 1)];
		}
		Data = MoveTemp(NewData);
		Head = 0;
	}

	TArray<T> Data;
	int32 Head = 0;
	int32 Count = 0;
};

constexpr uint64 SlotBits = 20;
constexpr uint64 SlotMask = (uint64(1) << SlotBits) - 1;

bool IsTerminal(ENPCBackgroundStatus Status)
{
	return Status == ENPCBackgroundStatus::Completed || Status == ENPCBackgroundStatus::Failed
		|| Status == ENPCBackgroundStatus::Cancelled || Status == ENPCBackgroundStatus::Expired;
}

/** Sets Array to Num elements, reporting whether its allocation had to grow. */
template <typename T>
bool SetNumTracked(TArray<T>& Array, int32 Num)
{
	const bool bGrow = Array.Max() < Num;
	Array.SetNumUninitialized(Num, EAllowShrinking::No);
	return bGrow;
}
}

struct FBackgroundRequestState
{
	uint32 Slot = 0;
	uint64 Serial = 0;
	bool bInUse = false;
	bool bCommitted = false;
	int32 Rows = 0;
	int64 Bytes = 0;
	TArray<float> Input;
	TArray<float> Output;
	TArray<int64> Keys;
	TArray<int32> RowWorkers;
    FBackgroundRequestState* CompletedPrev = nullptr;
    FBackgroundRequestState* CompletedNext = nullptr;
    bool bCompletionQueued = false;
	/** Rows not yet executed or skipped; the request is terminal when this reaches zero. */
	int32 RowsOutstanding = 0;
	double SubmitTime = 0.0;
	double ExpireTime = 0.0;
	double CompleteTime = 0.0;
	uint64 UserTag = 0;
	ENPCBackgroundStatus Status = ENPCBackgroundStatus::Pending;
	bool bFailed = false;
	bool bCancelled = false;
	bool bExpired = false;
	bool bReleaseWhenDone = false;

	uint64 GetId() const { return (Serial << SlotBits) | (uint64(Slot) + 1); }
};

struct FBackgroundSegment
{
	FBackgroundRequestState* Request = nullptr;
	int32 Begin = 0;
	int32 Count = 0;
};

struct FBackgroundBatch
{
	TArray<FBackgroundSegment> Segments;
	int32 Rows = 0;
	int32 Lane = INDEX_NONE;
	double FormedTime = 0.0;
	double EstSeconds = 0.0;
};

struct FBackgroundPartition
{
	TRing<FBackgroundSegment> Pending;
	int32 Rows = 0;
};

class FNPCBackgroundSchedulerImpl
{
public:
	struct FWorker : public FRunnable
	{
		FNPCBackgroundSchedulerImpl* Owner = nullptr;
		int32 Index = 0;
		FEvent* Wake = nullptr;
		FRunnableThread* Thread = nullptr;
		TUniquePtr<IBatchInferenceContext> Context;
		TRing<FBackgroundBatch*> Queue;
		double EstQueued = 0.0;
		double EstExecuting = 0.0;
		bool bIdle = false;
		TArray<float> InStage;
		TArray<float> OutStage;
		int64 Batches = 0;
		int64 Rows = 0;
		double BusySeconds = 0.0;

		virtual uint32 Run() override
		{
			Owner->WorkerLoop(*this);
			return 0;
		}
	};

	FNPCBackgroundSchedulerSettings Settings;
	TSharedPtr<IBatchInferenceBackend> Backend;
	int32 ObsSize = 0;
	int32 ActSize = 0;
	int32 MaxBatch = 0;
	int32 MaxInFlight = 0;
	double AssemblyWaitSeconds = 0.0;
	bool bPinnedPartitions = false;

	mutable FCriticalSection Lock;
	bool bStopping = false;
	bool bStopped = false;
	TArray<TUniquePtr<FWorker>> Workers;
	TArray<FBackgroundPartition> Partitions;
	TRing<FBackgroundBatch*> SharedQueue;
	TArray<FBackgroundBatch*> FormedScratch;

	TArray<FBackgroundRequestState*> Slots;
	TArray<int32> FreeSlots;
	TArray<FBackgroundBatch*> FreeBatches;
	FBackgroundRequestState* CompletedHead = nullptr;
    FBackgroundRequestState* CompletedTail = nullptr;
	uint64 NextSerial = 1;

	int32 PendingRows = 0;
	int64 PendingBytes = 0;
	int32 InFlightBatches = 0;
	int32 ActiveWorkers = 0;
	uint32 RowRoundRobin = 0;
	uint32 BatchRoundRobin = 0;
	FRandomStream Random{0x5c401a};

	// Exponentially weighted least-squares estimate of service time = Fixed + PerRow * Rows.
	double EwRows = 0.0, EwTime = 0.0, EwRowsSq = 0.0, EwRowsTime = 0.0;
	bool bHaveServiceEstimate = false;

	FNPCBackgroundStats Stats;
    FBatchInferenceBackendStats BackendStatsBaseline;
	FLatencyHistogram AssemblyWait, QueueWait, Service, Completion;

	~FNPCBackgroundSchedulerImpl()
	{
		Shutdown();
		for (FBackgroundRequestState* Request : Slots)
		{
			delete Request;
		}
		for (FBackgroundBatch* Batch : FreeBatches)
		{
			delete Batch;
		}
	}

	// ---- Setup -------------------------------------------------------------------------------

	bool Start(FString& OutError)
	{
		for (int32 Index = 0; Index < Settings.WorkerCount; ++Index)
		{
			TUniquePtr<FWorker> Worker = MakeUnique<FWorker>();
			Worker->Owner = this;
			Worker->Index = Index;
			Worker->Context = Backend->CreateContext();
			if (!Worker->Context)
			{
				OutError = FString::Printf(TEXT("Backend '%s' failed to create execution context %d"), *Backend->GetName(), Index);
				return false;
			}
			Worker->Wake = FPlatformProcess::GetSynchEventFromPool(false);
			Worker->InStage.SetNumUninitialized(MaxBatch * ObsSize);
			Worker->OutStage.SetNumUninitialized(MaxBatch * ActSize);
			Worker->Queue.Reserve(Settings.MaxInFlightBatchesPerWorker * Settings.WorkerCount);
			Workers.Add(MoveTemp(Worker));
		}
        FreeBatches.Reserve(MaxInFlight + 1);
        for (int32 I = 0; I < MaxInFlight; ++I)
        {
            auto* Batch = new FBackgroundBatch();
            Batch->Segments.Reserve(MaxBatch);
            FreeBatches.Add(Batch);
        }
		Partitions.SetNum(bPinnedPartitions ? Settings.WorkerCount : 1);
        // Every row can be its own segment under key affinity. Reserve the admitted-row bound.
        for (auto& Partition : Partitions) Partition.Pending.Reserve(int32(FMath::Min(int64(Settings.MaxPendingRows), Settings.MaxPendingBytes / ((int64(ObsSize) + ActSize) * int64(sizeof(float))))));
		SharedQueue.Reserve(MaxInFlight);
		FormedScratch.Reserve(MaxInFlight);
		Stats.WorkerBatches.SetNumZeroed(Settings.WorkerCount);
		Stats.WorkerRows.SetNumZeroed(Settings.WorkerCount);
		Stats.WorkerBusySeconds.SetNumZeroed(Settings.WorkerCount);
		// Threads start only after every context exists, so contexts are created on the caller's thread.
		for (const TUniquePtr<FWorker>& Worker : Workers)
		{
			Worker->Thread = FRunnableThread::Create(Worker.Get(),
				*FString::Printf(TEXT("NPCBackgroundWorker%d"), Worker->Index), 0, Settings.WorkerPriority == ENPCBackgroundWorkerPriority::BelowNormal ? TPri_BelowNormal : TPri_Normal);
			if (!Worker->Thread)
			{
				OutError = TEXT("Failed to create a Background worker thread");
				return false;
			}
		}
		return true;
	}

	void Shutdown()
	{
		{
			FNPCSchedulerScopeLock ScopeLock(&Lock);
			if (bStopping)
			{
				return;
			}
			bStopping = true;
			for (const TUniquePtr<FWorker>& Worker : Workers)
			{
				if (Worker->Wake)
				{
					Worker->Wake->Trigger();
				}
			}
		}
		// Workers finish their current batch, then exit; no new batch is taken once stopping.
		for (const TUniquePtr<FWorker>& Worker : Workers)
		{
			if (Worker->Thread)
			{
				Worker->Thread->WaitForCompletion();
				delete Worker->Thread;
				Worker->Thread = nullptr;
			}
		}

		FNPCSchedulerScopeLock ScopeLock(&Lock);
		const double Now = FPlatformTime::Seconds();
		auto AbandonBatch = [this, Now](FBackgroundBatch* Batch)
		{
			for (const FBackgroundSegment& Segment : Batch->Segments)
			{
				Segment.Request->bCancelled = true;
				RetireRowsLocked(Segment.Request, Segment.Count, Now);
			}
			--InFlightBatches;
			ReleaseBatchLocked(Batch);
		};
		while (!SharedQueue.IsEmpty())
		{
			AbandonBatch(SharedQueue.PopFront());
		}
		for (const TUniquePtr<FWorker>& Worker : Workers)
		{
			while (!Worker->Queue.IsEmpty())
			{
				AbandonBatch(Worker->Queue.PopFront());
			}
		}
		for (FBackgroundPartition& Partition : Partitions)
		{
			while (!Partition.Pending.IsEmpty())
			{
				const FBackgroundSegment Segment = Partition.Pending.PopFront();
				Segment.Request->bCancelled = true;
				RetireRowsLocked(Segment.Request, Segment.Count, Now);
			}
			Partition.Rows = 0;
		}
		for (const TUniquePtr<FWorker>& Worker : Workers)
		{
			Worker->Context.Reset();
			if (Worker->Wake)
			{
				FPlatformProcess::ReturnSynchEventToPool(Worker->Wake);
				Worker->Wake = nullptr;
			}
		}
		bStopped = true;
	}

	// ---- Requests ----------------------------------------------------------------------------

	FBackgroundRequestState* FindLocked(FNPCBackgroundRequest Handle) const
	{
		const uint64 Id = static_cast<uint64>(Handle.Id);
		const uint64 SlotPlusOne = Id & SlotMask;
		if (SlotPlusOne == 0 || SlotPlusOne > static_cast<uint64>(Slots.Num()))
		{
			return nullptr;
		}
		FBackgroundRequestState* Request = Slots[SlotPlusOne - 1];
		return Request->bInUse && Request->Serial == (Id >> SlotBits) ? Request : nullptr;
	}

	FBackgroundRequestState* AcquireRequestLocked()
	{
		FBackgroundRequestState* Request = nullptr;
		if (FreeSlots.Num() > 0)
		{
			Request = Slots[FreeSlots.Pop(EAllowShrinking::No)];
		}
		else
		{
			if (static_cast<uint64>(Slots.Num()) >= SlotMask)
			{
				return nullptr;
			}
			Request = new FBackgroundRequestState();
			Request->Slot = Slots.Num();
			Slots.Add(Request);
			FreeSlots.Reserve(Slots.Num());
			++Stats.PoolGrowthEvents;
		}
		Request->Serial = NextSerial++;
		Request->bInUse = true;
		Request->bCommitted = false;
		Request->bFailed = Request->bCancelled = Request->bExpired = Request->bReleaseWhenDone = false;
		Request->Status = ENPCBackgroundStatus::Pending;
		return Request;
	}

    void UnlinkCompletionLocked(FBackgroundRequestState* Request)
    {
        if (!Request->bCompletionQueued) return;
        if (Request->CompletedPrev) Request->CompletedPrev->CompletedNext = Request->CompletedNext;
        else CompletedHead = Request->CompletedNext;
        if (Request->CompletedNext) Request->CompletedNext->CompletedPrev = Request->CompletedPrev;
        else CompletedTail = Request->CompletedPrev;
        Request->CompletedPrev = Request->CompletedNext = nullptr;
        Request->bCompletionQueued = false;
    }
	void ReleaseRequestLocked(FBackgroundRequestState* Request)
	{
        UnlinkCompletionLocked(Request);
		PendingRows -= Request->Rows;
		PendingBytes -= Request->Bytes;
		Request->bInUse = false;
		FreeSlots.Add(Request->Slot);
	}

	/** Rows of Request reached a terminal point (executed, skipped, or abandoned). */
	void RetireRowsLocked(FBackgroundRequestState* Request, int32 Count, double Now)
	{
		Request->RowsOutstanding -= Count;
		check(Request->RowsOutstanding >= 0);
		if (Request->RowsOutstanding == 0 && Request->bCommitted)
		{
			FinishLocked(Request, Now);
		}
	}

	void FinishLocked(FBackgroundRequestState* Request, double Now)
	{
		Request->CompleteTime = Now;
		if (Request->bCancelled)
		{
			Request->Status = ENPCBackgroundStatus::Cancelled;
			Stats.CancelledRows += Request->Rows;
		}
		else if (Request->bExpired || (Request->ExpireTime > 0.0 && Now > Request->ExpireTime))
		{
			Request->Status = ENPCBackgroundStatus::Expired;
			Stats.ExpiredRows += Request->Rows;
		}
		else if (Request->bFailed)
		{
			Request->Status = ENPCBackgroundStatus::Failed;
			Stats.FailedRows += Request->Rows;
		}
		else
		{
			Request->Status = ENPCBackgroundStatus::Completed;
			Stats.CompletedRows += Request->Rows;
		}
		Completion.Record(Now - Request->SubmitTime);
		if (Request->bReleaseWhenDone)
		{
			ReleaseRequestLocked(Request);
		}
        else
        {
            Request->CompletedPrev = CompletedTail;
            Request->CompletedNext = nullptr;
            Request->bCompletionQueued = true;
            if (CompletedTail) CompletedTail->CompletedNext = Request;
            else CompletedHead = Request;
            CompletedTail = Request;
        }
	}

	ENPCBackgroundStatus Submit(int32 Rows, TFunctionRef<void(TArrayView<float>)> Fill,
		const FNPCBackgroundSubmitOptions& Options, FNPCBackgroundRequest& OutRequest)
	{
		const int64 Bytes = static_cast<int64>(Rows) * (ObsSize + ActSize) * sizeof(float);
		FBackgroundRequestState* Request = nullptr;
		{
			FNPCSchedulerScopeLock ScopeLock(&Lock);
			++Stats.SubmittedRequests;
			if (Rows <= 0 || (Options.Keys.Num() != 0 && Options.Keys.Num() != Rows)
				|| Rows > Settings.MaxPendingRows || Bytes > Settings.MaxPendingBytes)
			{
				++Stats.InvalidSubmissions;
				return ENPCBackgroundStatus::Invalid;
			}
			if (bStopping || PendingRows + Rows > Settings.MaxPendingRows || PendingBytes + Bytes > Settings.MaxPendingBytes)
			{
				++Stats.BusyRejections;
				return ENPCBackgroundStatus::Busy;
			}
			Request = AcquireRequestLocked();
			if (!Request)
			{
				++Stats.BusyRejections;
				return ENPCBackgroundStatus::Busy;
			}
			// Reserve capacity now so concurrent submitters cannot overcommit while this one fills.
			PendingRows += Rows;
			PendingBytes += Bytes;
			Stats.PeakPendingRows = FMath::Max(Stats.PeakPendingRows, PendingRows);
			Stats.PeakPendingBytes = FMath::Max(Stats.PeakPendingBytes, PendingBytes);
			Request->Rows = Rows;
			Request->Bytes = Bytes;
			Request->RowsOutstanding = Rows;
			Request->UserTag = Options.UserTag;
			int64 Growth = 0;
			Growth += SetNumTracked(Request->Input, Rows * ObsSize);
			Growth += SetNumTracked(Request->Output, Rows * ActSize);
			Growth += SetNumTracked(Request->Keys, Options.Keys.Num());
			Growth += SetNumTracked(Request->RowWorkers, Rows);
			Stats.PoolGrowthEvents += Growth;
		}

		// The request is invisible to workers until committed, so it can be filled without the lock.
		Fill(TArrayView<float>(Request->Input.GetData(), Rows * ObsSize));
		if (Options.Keys.Num() > 0)
		{
			FMemory::Memcpy(Request->Keys.GetData(), Options.Keys.GetData(), Rows * sizeof(int64));
		}
		for (int32& Worker : Request->RowWorkers)
		{
			Worker = INDEX_NONE;
		}

		FNPCSchedulerScopeLock ScopeLock(&Lock);
		const double Now = FPlatformTime::Seconds();
		Request->SubmitTime = Now;
		const float MaxAgeMs = Options.MaxAgeMs < 0.0f ? Settings.DefaultMaxAgeMs : Options.MaxAgeMs;
		Request->ExpireTime = MaxAgeMs > 0.0f ? Now + MaxAgeMs * 0.001 : 0.0;
		if (bStopping)
		{
			ReleaseRequestLocked(Request);
			++Stats.BusyRejections;
			return ENPCBackgroundStatus::Busy;
		}
		Request->bCommitted = true;
		Stats.AcceptedRows += Rows;
		EnqueueRowsLocked(Request);
		FormBatchesLocked(Now);
		if (AssemblyWaitSeconds > 0.0)
		{
			// Let an idle worker compute its timed wait for the partial batch.
			WakeIdleLocked(Workers.Num());
		}
		OutRequest.Id = static_cast<int64>(Request->GetId());
		return ENPCBackgroundStatus::Pending;
	}

	void PushSegmentLocked(int32 PartitionIndex, FBackgroundRequestState* Request, int32 Row)
	{
		FBackgroundPartition& Partition = Partitions[PartitionIndex];
		if (!Partition.Pending.IsEmpty())
		{
			FBackgroundSegment& Last = Partition.Pending.Back();
			if (Last.Request == Request && Last.Begin + Last.Count == Row)
			{
				++Last.Count;
				++Partition.Rows;
				return;
			}
		}
		if (Partition.Pending.PushBack({Request, Row, 1}))
		{
			++Stats.PoolGrowthEvents;
		}
		++Partition.Rows;
	}

	void EnqueueRowsLocked(FBackgroundRequestState* Request)
	{
		const int32 NumWorkers = Workers.Num();
		const bool bKeyed = Settings.Dispatch == ENPCBackgroundDispatch::KeyAffinity && Request->Keys.Num() == Request->Rows;
		if (!bPinnedPartitions)
		{
			if (Partitions[0].Pending.PushBack({Request, 0, Request->Rows}))
			{
				++Stats.PoolGrowthEvents;
			}
			Partitions[0].Rows += Request->Rows;
			return;
		}
		for (int32 Row = 0; Row < Request->Rows; ++Row)
		{
			// Keyless rows under KeyAffinity fall back to row round-robin.
			const int32 Partition = bKeyed
				? static_cast<int32>(static_cast<uint64>(Request->Keys[Row]) % static_cast<uint64>(NumWorkers))
				: static_cast<int32>(RowRoundRobin++ % static_cast<uint32>(NumWorkers));
			PushSegmentLocked(Partition, Request, Row);
		}
	}

	// ---- Batch formation and routing ---------------------------------------------------------

	FBackgroundBatch* AcquireBatchLocked()
	{
		if (FreeBatches.Num() > 0)
		{
			return FreeBatches.Pop(EAllowShrinking::No);
		}
		++Stats.PoolGrowthEvents;
		FBackgroundBatch* Batch = new FBackgroundBatch();
		Batch->Segments.Reserve(MaxBatch);
		FreeBatches.Reserve(MaxInFlight + 1);
		return Batch;
	}

	void ReleaseBatchLocked(FBackgroundBatch* Batch)
	{
		Batch->Segments.Reset();
		Batch->Rows = 0;
		Batch->Lane = INDEX_NONE;
		FreeBatches.Add(Batch);
	}

	/** True if the request's queued rows should be skipped instead of executed. */
	bool ShouldSkipLocked(FBackgroundRequestState* Request, double Now)
	{
		if (!Request->bCancelled && !Request->bExpired && Request->ExpireTime > 0.0 && Now > Request->ExpireTime)
		{
			Request->bExpired = true;
		}
		return Request->bCancelled || Request->bExpired;
	}

	void FormBatchesLocked(double Now)
	{
		if (bStopping)
		{
			return;
		}
		for (int32 PartitionIndex = 0; PartitionIndex < Partitions.Num(); ++PartitionIndex)
		{
			FBackgroundPartition& Partition = Partitions[PartitionIndex];
			FormedScratch.Reset();
			while (Partition.Rows > 0 && InFlightBatches < MaxInFlight)
			{
				// Drop skipped segments at the front so the assembly-wait check sees a live row.
				FBackgroundSegment& Front = Partition.Pending.Front();
				if (ShouldSkipLocked(Front.Request, Now))
				{
					const FBackgroundSegment Skipped = Partition.Pending.PopFront();
					Partition.Rows -= Skipped.Count;
					RetireRowsLocked(Skipped.Request, Skipped.Count, Now);
					continue;
				}
				if (Partition.Rows < MaxBatch && AssemblyWaitSeconds > 0.0 && Now - Front.Request->SubmitTime < AssemblyWaitSeconds)
				{
					break;
				}

				FBackgroundBatch* Batch = AcquireBatchLocked();
				while (Batch->Rows < MaxBatch && !Partition.Pending.IsEmpty())
				{
					FBackgroundSegment& Segment = Partition.Pending.Front();
					if (ShouldSkipLocked(Segment.Request, Now))
					{
						const FBackgroundSegment Skipped = Partition.Pending.PopFront();
						Partition.Rows -= Skipped.Count;
						RetireRowsLocked(Skipped.Request, Skipped.Count, Now);
						continue;
					}
					const int32 Take = FMath::Min(Segment.Count, MaxBatch - Batch->Rows);
					Batch->Segments.Add({Segment.Request, Segment.Begin, Take});
					AssemblyWait.Record(Now - Segment.Request->SubmitTime, Take);
					Batch->Rows += Take;
					Partition.Rows -= Take;
					Segment.Begin += Take;
					Segment.Count -= Take;
					if (Segment.Count == 0)
					{
						Partition.Pending.PopFront();
					}
				}
				if (Batch->Rows == 0)
				{
					ReleaseBatchLocked(Batch);
					continue;
				}
				Batch->FormedTime = Now;
				Batch->EstSeconds = EstimateSeconds(Batch->Rows);
				++InFlightBatches;
				Stats.PeakInFlightBatches = FMath::Max(Stats.PeakInFlightBatches, InFlightBatches);
				if (bPinnedPartitions)
				{
					RouteToLaneLocked(Batch, PartitionIndex);
				}
				else
				{
					FormedScratch.Add(Batch);
				}
			}
			for (int32 Index = 0; Index < FormedScratch.Num(); ++Index)
			{
				RouteLocked(FormedScratch[Index], Index, FormedScratch.Num());
			}
		}
	}

	double EstimateSeconds(int32 Rows) const
	{
		if (!bHaveServiceEstimate)
		{
			return static_cast<double>(Rows);
		}
		const double Variance = EwRowsSq - EwRows * EwRows;
		double PerRow, Fixed;
		if (Variance > 1e-9)
		{
			PerRow = FMath::Max(0.0, (EwRowsTime - EwRows * EwTime) / Variance);
			Fixed = FMath::Max(0.0, EwTime - PerRow * EwRows);
		}
		else
		{
			PerRow = EwRows > 0.0 ? EwTime / EwRows : 0.0;
			Fixed = 0.0;
		}
		return Fixed + PerRow * Rows;
	}

	void UpdateServiceEstimate(int32 Rows, double Seconds)
	{
		const double Alpha = bHaveServiceEstimate ? 0.05 : 1.0;
		EwRows += Alpha * (Rows - EwRows);
		EwTime += Alpha * (Seconds - EwTime);
		EwRowsSq += Alpha * (double(Rows) * Rows - EwRowsSq);
		EwRowsTime += Alpha * (Rows * Seconds - EwRowsTime);
		bHaveServiceEstimate = true;
	}

	double LaneLoad(int32 Lane) const
	{
		return Workers[Lane]->EstQueued + Workers[Lane]->EstExecuting;
	}

	void RouteLocked(FBackgroundBatch* Batch, int32 IndexInFlush, int32 FlushCount)
	{
		const int32 NumWorkers = Workers.Num();
		switch (Settings.Dispatch)
		{
		case ENPCBackgroundDispatch::Contiguous:
			RouteToLaneLocked(Batch, static_cast<int32>(static_cast<int64>(IndexInFlush) * NumWorkers / FlushCount));
			return;
		case ENPCBackgroundDispatch::BatchRoundRobin:
		case ENPCBackgroundDispatch::WorkStealing:
			RouteToLaneLocked(Batch, static_cast<int32>(BatchRoundRobin++ % static_cast<uint32>(NumWorkers)));
			return;
		case ENPCBackgroundDispatch::LeastLoaded:
		{
			int32 Best = 0;
			if (NumWorkers <= 2)
			{
				Best = NumWorkers == 2 && LaneLoad(1) < LaneLoad(0) ? 1 : 0;
			}
			else
			{
				// Power of two choices on estimated queued service time.
				const int32 First = Random.RandHelper(NumWorkers);
				int32 Second = Random.RandHelper(NumWorkers - 1);
				Second += Second >= First ? 1 : 0;
				Best = LaneLoad(Second) < LaneLoad(First) ? Second : First;
			}
			RouteToLaneLocked(Batch, Best);
			return;
		}
		default:
			if (SharedQueue.PushBack(Batch))
			{
				++Stats.PoolGrowthEvents;
			}
			WakeIdleLocked(1);
			return;
		}
	}

	void RouteToLaneLocked(FBackgroundBatch* Batch, int32 Lane)
	{
		FWorker& Worker = *Workers[Lane];
		Batch->Lane = Lane;
		if (Worker.Queue.PushBack(Batch))
		{
			++Stats.PoolGrowthEvents;
		}
		Worker.EstQueued += Batch->EstSeconds;
		TriggerLocked(Worker);
		if (Settings.Dispatch == ENPCBackgroundDispatch::WorkStealing)
		{
			WakeIdleLocked(1);
		}
	}

	void TriggerLocked(FWorker& Worker)
	{
		if (Worker.bIdle)
		{
			Worker.bIdle = false;
			Worker.Wake->Trigger();
		}
	}

	void WakeIdleLocked(int32 Count)
	{
		for (const TUniquePtr<FWorker>& Worker : Workers)
		{
			if (Count <= 0)
			{
				return;
			}
			if (Worker->bIdle)
			{
				TriggerLocked(*Worker);
				--Count;
			}
		}
	}

	// ---- Workers -----------------------------------------------------------------------------

	/** Returns the next batch for Worker, or null with OutWaitSeconds (negative = until woken). */
	FBackgroundBatch* TakeLocked(FWorker& Worker, double Now, double& OutWaitSeconds)
	{
		FormBatchesLocked(Now);
		FBackgroundBatch* Batch = nullptr;
		if (!SharedQueue.IsEmpty())
		{
			Batch = SharedQueue.PopFront();
		}
		else if (!Worker.Queue.IsEmpty())
		{
			Batch = Worker.Queue.PopFront();
			Worker.EstQueued -= Batch->EstSeconds;
		}
		else if (Settings.Dispatch == ENPCBackgroundDispatch::WorkStealing)
		{
			FWorker* Victim = nullptr;
			for (const TUniquePtr<FWorker>& Other : Workers)
			{
				if (Other->Queue.Num() > 0 && (!Victim || Other->EstQueued > Victim->EstQueued))
				{
					Victim = Other.Get();
				}
			}
			if (Victim)
			{
				// Steal the newest batch; the owner keeps its oldest (FIFO) work.
				Batch = Victim->Queue.PopBack();
				Victim->EstQueued -= Batch->EstSeconds;
				++Stats.StolenBatches;
			}
		}

		if (Batch)
		{
			Worker.EstExecuting += Batch->EstSeconds;
			++ActiveWorkers;
			Stats.PeakActiveWorkers = FMath::Max(Stats.PeakActiveWorkers, ActiveWorkers);
			return Batch;
		}

		OutWaitSeconds = -1.0;
		if (AssemblyWaitSeconds > 0.0 && InFlightBatches < MaxInFlight)
		{
			for (FBackgroundPartition& Partition : Partitions)
			{
				if (Partition.Rows > 0)
				{
					const double Due = Partition.Pending.Front().Request->SubmitTime + AssemblyWaitSeconds - Now;
					OutWaitSeconds = OutWaitSeconds < 0.0 ? FMath::Max(0.0, Due) : FMath::Min(OutWaitSeconds, FMath::Max(0.0, Due));
				}
			}
		}
		Worker.bIdle = true;
		return nullptr;
	}

	void WorkerLoop(FWorker& Worker)
	{
		for (;;)
		{
			FBackgroundBatch* Batch = nullptr;
			double WaitSeconds = -1.0;
			{
				FNPCSchedulerScopeLock ScopeLock(&Lock);
				if (bStopping)
				{
					return;
				}
				Batch = TakeLocked(Worker, FPlatformTime::Seconds(), WaitSeconds);
			}
			if (!Batch)
			{
				if (WaitSeconds < 0.0)
				{
					Worker.Wake->Wait();
				}
				else
				{
					Worker.Wake->Wait(FMath::Max<uint32>(1, static_cast<uint32>(FMath::CeilToDouble(WaitSeconds * 1000.0))));
				}
				continue;
			}
			Execute(Worker, Batch);
		}
	}

	void Execute(FWorker& Worker, FBackgroundBatch* Batch)
	{
		// Request buffers are stable while rows are outstanding, and each row belongs to one batch,
		// so reading inputs and writing outputs here needs no lock.
		const double Start = FPlatformTime::Seconds();
		const bool bDirect = Batch->Segments.Num() == 1;
		bool bSuccess;
		if (bDirect)
		{
			const FBackgroundSegment& Segment = Batch->Segments[0];
			bSuccess = Worker.Context->RunBatch(Segment.Request->Input.GetData() + Segment.Begin * ObsSize, Batch->Rows,
				Segment.Request->Output.GetData() + Segment.Begin * ActSize);
		}
		else
		{
			float* In = Worker.InStage.GetData();
			for (const FBackgroundSegment& Segment : Batch->Segments)
			{
				FMemory::Memcpy(In, Segment.Request->Input.GetData() + Segment.Begin * ObsSize, Segment.Count * ObsSize * sizeof(float));
				In += Segment.Count * ObsSize;
			}
			bSuccess = Worker.Context->RunBatch(Worker.InStage.GetData(), Batch->Rows, Worker.OutStage.GetData());
			if (bSuccess)
			{
				const float* Out = Worker.OutStage.GetData();
				for (const FBackgroundSegment& Segment : Batch->Segments)
				{
					FMemory::Memcpy(Segment.Request->Output.GetData() + Segment.Begin * ActSize, Out, Segment.Count * ActSize * sizeof(float));
					Out += Segment.Count * ActSize;
				}
			}
		}
		const double End = FPlatformTime::Seconds();

		FNPCSchedulerScopeLock ScopeLock(&Lock);
		--ActiveWorkers;
		--InFlightBatches;
		Worker.EstExecuting = FMath::Max(0.0, Worker.EstExecuting - Batch->EstSeconds);
		++Stats.BackendCalls;
		Stats.BackendRows += Batch->Rows;
		Stats.FailedBackendCalls += bSuccess ? 0 : 1;
		Stats.WorkerBatches[Worker.Index] += 1;
		Stats.WorkerRows[Worker.Index] += Batch->Rows;
		Stats.WorkerBusySeconds[Worker.Index] += static_cast<float>(End - Start);
		QueueWait.Record(Start - Batch->FormedTime, Batch->Rows);
		Service.Record(End - Start);
		if (bSuccess)
		{
			UpdateServiceEstimate(Batch->Rows, End - Start);
		}
		for (const FBackgroundSegment& Segment : Batch->Segments)
		{
			FBackgroundRequestState* Request = Segment.Request;
			for (int32 Row = Segment.Begin; Row < Segment.Begin + Segment.Count; ++Row)
			{
				Request->RowWorkers[Row] = Worker.Index;
			}
			Request->bFailed |= !bSuccess;
			RetireRowsLocked(Request, Segment.Count, End);
		}
		ReleaseBatchLocked(Batch);
	}

	// ---- Stats -------------------------------------------------------------------------------

	FNPCBackgroundStats GetStats() const
	{
		FNPCSchedulerScopeLock ScopeLock(&Lock);
		FNPCBackgroundStats Out = Stats;
        const auto Counters = Backend->GetBackendStats();
        Out.BackendStagingGrowthEvents = Counters.StagingGrowthEvents - BackendStatsBaseline.StagingGrowthEvents;
        Out.BackendReshapeCalls = Counters.ReshapeCalls - BackendStatsBaseline.ReshapeCalls;
        Out.PaddingRows = Counters.PaddingRows - BackendStatsBaseline.PaddingRows;
        Out.BackendReshapeSeconds = Counters.ReshapeSeconds - BackendStatsBaseline.ReshapeSeconds;
        Out.BackendServiceSeconds = Counters.ServiceSeconds - BackendStatsBaseline.ServiceSeconds;
		Out.PendingRows = PendingRows;
		Out.PendingBytes = PendingBytes;
		Out.MeanBatchFill = Stats.BackendCalls > 0
			? static_cast<float>(static_cast<double>(Stats.BackendRows) / (static_cast<double>(Stats.BackendCalls) * MaxBatch)) : 0.0f;
		Out.AssemblyWaitP99Ms = static_cast<float>(AssemblyWait.GetQuantile(0.99) * 1000.0);
		Out.AssemblyWaitMaxMs = AssemblyWait.GetMax() * 1000.0;
		Out.QueueWaitP99Ms = static_cast<float>(QueueWait.GetQuantile(0.99) * 1000.0);
		Out.ServiceP50Ms = static_cast<float>(Service.GetQuantile(0.50) * 1000.0);
		Out.ServiceP99Ms = static_cast<float>(Service.GetQuantile(0.99) * 1000.0);
		Out.CompletionP50Ms = static_cast<float>(Completion.GetQuantile(0.50) * 1000.0);
		Out.CompletionP99Ms = static_cast<float>(Completion.GetQuantile(0.99) * 1000.0);
		return Out;
	}

	void ResetStats()
	{
		FNPCSchedulerScopeLock ScopeLock(&Lock);
		const int32 NumWorkers = Workers.Num();
		Stats = FNPCBackgroundStats();
        BackendStatsBaseline = Backend->GetBackendStats();
		Stats.WorkerBatches.SetNumZeroed(NumWorkers);
		Stats.WorkerRows.SetNumZeroed(NumWorkers);
		Stats.WorkerBusySeconds.SetNumZeroed(NumWorkers);
		Stats.PeakPendingRows = PendingRows;
		Stats.PeakPendingBytes = PendingBytes;
		Stats.PeakInFlightBatches = InFlightBatches;
		Stats.PeakActiveWorkers = ActiveWorkers;
		AssemblyWait.Reset();
		QueueWait.Reset();
		Service.Reset();
		Completion.Reset();
	}
};

// ---- Public facade -------------------------------------------------------------------------------

TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> FNPCBackgroundScheduler::Create(
	const FNPCBackgroundSchedulerSettings& Settings, const TSharedRef<IBatchInferenceBackend>& Backend, FString* OutError)
{
	FString Error;
	const int32 ObsSize = Backend->GetObservationSize();
	const int32 ActSize = Backend->GetActionSize();
	const int32 BackendMax = Backend->GetMaxBatchSize();
	if (Settings.WorkerCount < 1 || Settings.WorkerCount > 64)
	{
		Error = TEXT("WorkerCount must be in [1, 64]");
	}
	else if (Settings.MaxBatchSize < 1 || Settings.MaxPendingRows < 1 || Settings.MaxPendingBytes < 1
		|| Settings.MaxInFlightBatchesPerWorker < 1 || Settings.MaxAssemblyWaitMs < 0.0f || Settings.DefaultMaxAgeMs < 0.0f)
	{
		Error = TEXT("Batch size, caps, and in-flight limit must be positive; waits and ages nonnegative");
	}
	else if (!StaticEnum<ENPCBackgroundDispatch>()->IsValidEnumValue(static_cast<int64>(Settings.Dispatch)) ||
        !StaticEnum<ENPCBackgroundWorkerPriority>()->IsValidEnumValue(static_cast<int64>(Settings.WorkerPriority)))
	{
		Error = TEXT("Unknown dispatch strategy");
	}
	else if (ObsSize <= 0 || ActSize <= 0 || BackendMax < 0 ||
        int64(FMath::Max(ObsSize, ActSize)) * Settings.MaxBatchSize > MAX_int32 / 8)
	{
		Error = FString::Printf(TEXT("Backend '%s' reports invalid row sizes"), *Backend->GetName());
	}

	TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Scheduler;
	if (Error.IsEmpty())
	{
		Scheduler = MakeShareable(new FNPCBackgroundScheduler());
		Scheduler->Impl = MakeUnique<FNPCBackgroundSchedulerImpl>();
		FNPCBackgroundSchedulerImpl& Impl = *Scheduler->Impl;
		Impl.Settings = Settings;
		Impl.Backend = Backend;
		Impl.ObsSize = ObsSize;
		Impl.ActSize = ActSize;
		Impl.MaxBatch = BackendMax > 0 ? FMath::Min(Settings.MaxBatchSize, BackendMax) : Settings.MaxBatchSize;
		Impl.MaxInFlight = Settings.WorkerCount * Settings.MaxInFlightBatchesPerWorker;
		Impl.AssemblyWaitSeconds = Settings.MaxAssemblyWaitMs * 0.001;
		Impl.bPinnedPartitions = Settings.Dispatch == ENPCBackgroundDispatch::RowRoundRobin
			|| Settings.Dispatch == ENPCBackgroundDispatch::KeyAffinity;
		if (!Impl.Start(Error))
		{
			Scheduler.Reset();
		}
	}
	if (!Error.IsEmpty())
	{
		UE_LOGFMT(LogScholaInferenceUtils, Error, "FNPCBackgroundScheduler::Create(): {0}", Error);
		if (OutError)
		{
			*OutError = Error;
		}
	}
	return Scheduler;
}

FNPCBackgroundScheduler::~FNPCBackgroundScheduler() = default;

ENPCBackgroundStatus FNPCBackgroundScheduler::Submit(TConstArrayView<float> Observations,
	const FNPCBackgroundSubmitOptions& Options, FNPCBackgroundRequest& OutRequest)
{
	const int32 ObsSize = Impl->ObsSize;
	if (Observations.Num() == 0 || Observations.Num() % ObsSize != 0)
	{
		FNPCSchedulerScopeLock ScopeLock(&Impl->Lock);
		++Impl->Stats.SubmittedRequests;
		++Impl->Stats.InvalidSubmissions;
		return ENPCBackgroundStatus::Invalid;
	}
	return Impl->Submit(Observations.Num() / ObsSize,
		[&Observations](TArrayView<float> Rows) { FMemory::Memcpy(Rows.GetData(), Observations.GetData(), Rows.Num() * sizeof(float)); },
		Options, OutRequest);
}

ENPCBackgroundStatus FNPCBackgroundScheduler::SubmitWith(int32 Rows, TFunctionRef<void(TArrayView<float>)> Fill,
	const FNPCBackgroundSubmitOptions& Options, FNPCBackgroundRequest& OutRequest)
{
	return Impl->Submit(Rows, Fill, Options, OutRequest);
}

ENPCBackgroundStatus FNPCBackgroundScheduler::GetStatus(FNPCBackgroundRequest Request) const
{
	FNPCSchedulerScopeLock ScopeLock(&Impl->Lock);
	const FBackgroundRequestState* State = Impl->FindLocked(Request);
	// A cancelled request is retired for the caller even while its executing rows drain.
	return State && !State->bReleaseWhenDone ? State->Status : ENPCBackgroundStatus::Invalid;
}

ENPCBackgroundStatus FNPCBackgroundScheduler::Consume(FNPCBackgroundRequest Request, FNPCBackgroundResult& OutResult)
{
	FNPCSchedulerScopeLock ScopeLock(&Impl->Lock);
	FBackgroundRequestState* State = Impl->FindLocked(Request);
	if (!State || State->bReleaseWhenDone)
	{
		return ENPCBackgroundStatus::Invalid;
	}
	if (!IsTerminal(State->Status))
	{
		return State->Status;
	}
	OutResult.Status = State->Status;
	OutResult.Rows = State->Rows;
	OutResult.ActionSize = Impl->ActSize;
	OutResult.UserTag = State->UserTag;
	OutResult.SubmitSeconds = State->SubmitTime;
	OutResult.CompleteSeconds = State->CompleteTime;
    // Keep both request and consumer high-water capacities. Swapping a large
    // request with a smaller previous result made already-warmed pools regrow.
    auto CopyBuffer=[&](auto& To,const auto& From)
    {
        Impl->Stats.PoolGrowthEvents+=SetNumTracked(To,From.Num());
        if(!From.IsEmpty())FMemory::Memcpy(To.GetData(),From.GetData(),From.Num()*sizeof(*From.GetData()));
    };
    CopyBuffer(OutResult.Keys,State->Keys);
    CopyBuffer(OutResult.RowWorkers,State->RowWorkers);
    if(State->Status==ENPCBackgroundStatus::Completed)CopyBuffer(OutResult.Actions,State->Output);
	Impl->ReleaseRequestLocked(State);
	return OutResult.Status;
}

bool FNPCBackgroundScheduler::Cancel(FNPCBackgroundRequest Request)
{
	FNPCSchedulerScopeLock ScopeLock(&Impl->Lock);
	FBackgroundRequestState* State = Impl->FindLocked(Request);
	if (!State || State->bReleaseWhenDone)
	{
		return false;
	}
	if (IsTerminal(State->Status))
	{
		Impl->ReleaseRequestLocked(State);
	}
	else
	{
		State->bCancelled = true;
		State->bReleaseWhenDone = true;
	}
	return true;
}

ENPCBackgroundStatus FNPCBackgroundScheduler::Wait(FNPCBackgroundRequest Request, double TimeoutSeconds) const
{
	// Polling keeps the hot completion path free of per-request events; this path is for
	// synchronous adapters off the game thread, where sub-millisecond wake latency suffices.
	const double Deadline = TimeoutSeconds < 0.0 ? 0.0 : FPlatformTime::Seconds() + TimeoutSeconds;
	for (int32 Iteration = 0;; ++Iteration)
	{
		const ENPCBackgroundStatus Status = GetStatus(Request);
		if (Status != ENPCBackgroundStatus::Pending)
		{
			return Status;
		}
		if (TimeoutSeconds >= 0.0 && FPlatformTime::Seconds() >= Deadline)
		{
			return Status;
		}
		if (Iteration < 64)
		{
			FPlatformProcess::YieldThread();
		}
		else
		{
			FPlatformProcess::SleepNoStats(0.0001f);
		}
	}
}

int32 FNPCBackgroundScheduler::DrainCompleted(TArray<FNPCBackgroundRequest>& OutRequests)
{
	FNPCSchedulerScopeLock ScopeLock(&Impl->Lock);
	int32 Added = 0;
    while (Impl->CompletedHead)
    {
        auto* State = Impl->CompletedHead;
        FNPCBackgroundRequest Request;
        Request.Id = int64(State->GetId());
        Impl->UnlinkCompletionLocked(State);
        OutRequests.Add(Request);
        ++Added;
    }
	return Added;
}

bool FNPCBackgroundScheduler::CanAccept(int32 Rows) const
{
	FNPCSchedulerScopeLock ScopeLock(&Impl->Lock);
	const int64 Bytes = static_cast<int64>(Rows) * (Impl->ObsSize + Impl->ActSize) * sizeof(float);
	return !Impl->bStopping && Rows > 0 && Impl->PendingRows + Rows <= Impl->Settings.MaxPendingRows
		&& Impl->PendingBytes + Bytes <= Impl->Settings.MaxPendingBytes;
}

FNPCBackgroundStats FNPCBackgroundScheduler::GetStats() const { return Impl->GetStats(); }
int64 FNPCBackgroundScheduler::GetBackendCallCount() const { FNPCSchedulerScopeLock ScopeLock(&Impl->Lock); return Impl->Stats.BackendCalls; }
void FNPCBackgroundScheduler::ResetStats() { Impl->ResetStats(); }
void FNPCBackgroundScheduler::Shutdown() { Impl->Shutdown(); }
bool FNPCBackgroundScheduler::IsShuttingDown() const { FNPCSchedulerScopeLock ScopeLock(&Impl->Lock); return Impl->bStopping; }
int32 FNPCBackgroundScheduler::GetObservationSize() const { return Impl->ObsSize; }
int32 FNPCBackgroundScheduler::GetActionSize() const { return Impl->ActSize; }
int32 FNPCBackgroundScheduler::GetWorkerCount() const { return Impl->Workers.Num(); }
int32 FNPCBackgroundScheduler::GetMaxBatchSize() const { return Impl->MaxBatch; }
const FNPCBackgroundSchedulerSettings& FNPCBackgroundScheduler::GetSettings() const { return Impl->Settings; }
