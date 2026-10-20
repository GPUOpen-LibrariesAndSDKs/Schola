// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "CohortScheduler.generated.h"

UENUM(BlueprintType)
enum class ENPCPhaseLayout : uint8 { ModelAware, PerEntityUniform };

/** Game-thread phase allocator shared by all adapters to a model bucket. */
class SCHOLAINFERENCEUTILS_API FNPCCohortScheduler
{
public:
    explicit FNPCCohortScheduler(uint32 InOffset = 0) : Offset(InOffset) {}
    int32 Acquire(float Period, ENPCPhaseLayout Layout, uint32 Key);
    void Release(float Period, int32 Slot);
    double NextDue(float Period, int32 Slot, double Now);
    uint32 NextRequestSerial();
    static int32 NumSlots(float Period) { return FMath::IsFinite(Period) ? FMath::RoundToInt(FMath::Clamp(double(Period) * 60.0, 1.0, 65535.0)) : 1; }
private:
    FCriticalSection Mutex;
    TMap<int32, TArray<int32>> Loads;
    uint32 Offset = 0, Serial = 0;
    double Epoch = 0;
    bool bHaveEpoch = false;
};
