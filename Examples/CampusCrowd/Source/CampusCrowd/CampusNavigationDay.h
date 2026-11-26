#pragma once

#include "CoreMinimal.h"
#include "CampusLandmarkPlanner.h"

class FCampusNavigationDay
{
public:
    bool Initialize(int32 Count, const TCHAR* Command, bool bBackground, FString& Error);
    void TickPlanner();
    bool UsesRoutePlanner() const { return Planner.IsValid(); }
    bool UsesLivePlanner() const { return UsesRoutePlanner() && !bPrecomputed; }
    FCampusPlanningStats GetPlanningStats() const { return Planner->GetStats(); }
    void Step(float Dt, TArray<FTransform>& Transforms);
    CampusNav::Vec GetGate(int32 Index) const { return Stops[FMath::Clamp(Index, 0, GateCount - 1)]; }
    bool IsWalking(int32 Index) const { return Students[Index].bWalking; }
    TConstArrayView<CampusNav::Vec> GetActiveRoute(int32 Index) const;
    const CampusNav::Grid& GetGrid() const { return Map; }
    CampusNav::Vec GetPosition(int32 Index) const { return Students[Index].Motion.Position; }
    int32 Generation(int32 Index) const { return Students[Index].Leg; }
    bool Finished() const { return Completed + TimedOut == Students.Num(); }
    int32 Active = 0, Completed = 0, TimedOut = 0, LegsCompleted = 0;

private:
    struct FStudent
    {
        int32 Stops[6] = {};
        int32 Leg = 0;
        bool bWalking = false;
        bool bTimedOut = false;
        bool bWaitingForRoute = false;
        TSharedPtr<const FCampusPlannedRoute, ESPMode::ThreadSafe> Route;
        float Wake = 0.f, Deadline = 0.f;
        CampusNav::State Motion;
        CampusNav::ShortcutCursor Shortcut;
    };

    bool LoadMap(FString& Error);
    bool LoadRoutes(const FString& Path, FString& Error);
    struct FStepTotals
    {
        int32 Active = 0, Completed = 0, TimedOut = 0, LegsCompleted = 0;
        TArray<int32> Departing;
    };
    static constexpr int32 StudentsPerBatch = 128;
    TArray<FStepTotals> BatchTotals;

    TArray<uint8> GridBytes;
    CampusNav::Grid Map;
    TArray<CampusNav::Vec> Stops;
    TArray<FCampusPlannedRoute> Routes;
    TUniquePtr<FCampusLandmarkPlanner> Planner;
    TArray<TArray<int32>> WaitingByPair, SubmittedByPair;
    TBitArray<> PendingGoals;
    TArray<FStudent> Students;
    bool bPrecomputed = false;
    int32 GateCount = 0;
    TArray<TArray<int32>> EntrancesByBuilding;
    float Time = 0.f;
    float Dwell = 18.f, LegSeconds = 1800.f;
};
