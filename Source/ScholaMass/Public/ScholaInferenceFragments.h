// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "Mass/EntityFragments.h"
#include "ScholaInferenceFragments.generated.h"

USTRUCT()
struct SCHOLAMASS_API FScholaInferenceScheduleFragment : public FMassFragment
{
    GENERATED_BODY()
    float PeriodSeconds = 1.f;
    double NextDueTime = 0;
    uint16 PhaseSlot = 0;
    bool bUrgent = false;
};
USTRUCT()
struct SCHOLAMASS_API FScholaInferenceStateFragment : public FMassFragment
{
    GENERATED_BODY()
    int32 Slot = INDEX_NONE;
    uint32 SlotSerial = 0, SubmittedSerial = 0, AppliedSerial = 0;
    double ObsTime = 0;
};
USTRUCT()
struct SCHOLAMASS_API FScholaInferenceBucketFragment : public FMassConstSharedFragment
{
    GENERATED_BODY()
    UPROPERTY() int32 BucketId = INDEX_NONE;
};
static_assert(std::is_trivially_copyable_v<FScholaInferenceScheduleFragment>);
static_assert(std::is_trivially_copyable_v<FScholaInferenceStateFragment>);
