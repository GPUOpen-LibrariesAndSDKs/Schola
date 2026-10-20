// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Inference/CohortScheduler.h"
#include "Misc/ScopeLock.h"
int32 FNPCCohortScheduler::Acquire(float Period, ENPCPhaseLayout Layout, uint32 Key)
{
    FScopeLock Lock(&Mutex);
    const int32 Count = NumSlots(Period);
    auto& Counts = Loads.FindOrAdd(Count);
    if (Counts.IsEmpty()) Counts.Init(0, Count);
    int32 Best = Layout == ENPCPhaseLayout::PerEntityUniform ? GetTypeHash(Key) % Count : Offset % Count;
    if (Layout == ENPCPhaseLayout::ModelAware)
        for (int32 I = 1; I < Count; ++I)
        {
            const int32 Candidate = (I + Offset) % Count;
            if (Counts[Candidate] < Counts[Best]) Best = Candidate;
        }
    ++Counts[Best];
    return Best;
}
void FNPCCohortScheduler::Release(float Period, int32 Slot)
{
    FScopeLock Lock(&Mutex);
    if (auto* Counts = Loads.Find(NumSlots(Period)); Counts && Counts->IsValidIndex(Slot))
    { check((*Counts)[Slot] > 0); --(*Counts)[Slot]; }
}

uint32 FNPCCohortScheduler::NextRequestSerial()
{ FScopeLock Lock(&Mutex); check(Serial != MAX_uint32); return ++Serial; }

double FNPCCohortScheduler::NextDue(float Period,int32 Slot,double Now)
{
    FScopeLock Lock(&Mutex);
    if(!bHaveEpoch){Epoch=Now;bHaveEpoch=true;}
    const double Base=Epoch+double(Slot)*Period/NumSlots(Period);
    return Base+FMath::Max(0.0,FMath::CeilToDouble((Now-Base-1e-6)/Period))*Period;
}
