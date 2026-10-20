// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

/**
 * Preallocated logarithmic histogram for durations from 0.1 µs to ~100 s with ~1% relative bucket width
 * (HdrHistogram-style). Recording never allocates. Not thread-safe; guard externally.
 */
class FLatencyHistogram
{
public:
	FLatencyHistogram() { Counts.SetNumZeroed(NumBuckets); }

	void Record(double Seconds, int64 Count = 1)
	{
		Counts[BucketFor(Seconds)] += Count;
		Total += Count;
		Max = FMath::Max(Max, Seconds);
	}

	void Reset()
	{
		FMemory::Memzero(Counts.GetData(), Counts.Num() * sizeof(int64));
		Total = 0;
		Max = 0.0;
	}

	int64 GetCount() const { return Total; }
	double GetMax() const { return Max; }

	/** Value at quantile Q in [0, 1], in seconds (bucket midpoint; ~0.5% error). 0 if empty. */
	double GetQuantile(double Q) const
	{
		if (Total == 0)
		{
			return 0.0;
		}
		const int64 Rank = FMath::Clamp<int64>(static_cast<int64>(FMath::CeilToDouble(Q * Total)), 1, Total);
		int64 Seen = 0;
		for (int32 Bucket = 0; Bucket < NumBuckets; ++Bucket)
		{
			Seen += Counts[Bucket];
			if (Seen >= Rank)
			{
				return FMath::Min(BucketMidpoint(Bucket), Max);
			}
		}
		return Max;
	}

private:
	static constexpr double MinSeconds = 1e-7;
	static constexpr double Growth = 1.01;
	static constexpr int32 NumBuckets = 2100;

	static int32 BucketFor(double Seconds)
	{
		if (Seconds <= MinSeconds)
		{
			return 0;
		}
		const int32 Bucket = 1 + static_cast<int32>(FMath::Loge(Seconds / MinSeconds) / FMath::Loge(Growth));
		return FMath::Min(Bucket, NumBuckets - 1);
	}

	static double BucketMidpoint(int32 Bucket)
	{
		if (Bucket == 0)
		{
			return MinSeconds;
		}
		const double Low = MinSeconds * FMath::Pow(Growth, static_cast<double>(Bucket - 1));
		return Low * (1.0 + Growth) * 0.5;
	}

	TArray<int64> Counts;
	int64 Total = 0;
	double Max = 0.0;
};
