#pragma once

#include "CoreMinimal.h"
#include "CampusNavigationDay.h"
#include "CampusTurnDay.h"

namespace CampusMigration
{
constexpr int32 LegCount = 5;
constexpr int32 ObservationSize = 64; // Same batch ABI as the benchmark.
constexpr int32 NeighborStart = 8;
constexpr int32 NeighborStride = 5;
constexpr int32 NeighborCount = 4;
}

struct FCampusMigrationStudent
{
    int32 Stops[CampusMigration::LegCount + 1] = {};
    bool bWalking = false;
    bool bArrived = false;
    int32 Segment = 0;
    float Progress = 0;
    float Offset = 0;
    float Pace = 140;
    float Lane = 0;
    float ReleaseDelay = 0;
    FVector2f Position = FVector2f::ZeroVector;
    FVector2f Velocity = FVector2f::ZeroVector;
    FVector2f LocalVelocity = FVector2f::ZeroVector;
    FVector2f Action = FVector2f::ZeroVector;
    FVector2f Facing = FVector2f(1, 0);
    float Observation[CampusMigration::ObservationSize] = {};
};

/** Five schedule-driven outdoor trips. Positions and observations are indexed by Mass instance. */
class FCampusMigrationDay
{
public:
    bool Initialize(int32 StudentCount, bool bNavigation, bool bTurns, const TCHAR* Command, FString& Error, bool bBackground, bool bFourDemo = false);
    void Step(float DeltaSeconds, TArray<FTransform>& Transforms);
    bool IsWalking(int32 Student) const { return Turns ? Turns->IsWalking(Student) : Navigation ? Navigation->IsWalking(Student) : Students[Student].bWalking; }
    bool WantsDecision(int32 Student) const { return Turns ? Turns->WantsDecision(Student) : IsWalking(Student); }
    const FCampusTurnDay* GetTurns() const { return Turns.Get(); }
    void RetryOverdueTurns() { Turns->RetryOverdueDecisions(); }
    void WriteObservation(int32 Student, TArrayView<float> Output) const;
    void ApplyAction(int32 Student, TConstArrayView<float> Input);
    int32 GetActive() const { return Turns ? Turns->Active : Navigation ? Navigation->Active : Active; }
    int32 GetArrived() const { return Turns ? Turns->Completed : Navigation ? Navigation->Completed : Arrived; }
    CampusNav::Vec GetNavigationGate(int32 Index) const { return Navigation->GetGate(Index); }
    TConstArrayView<CampusNav::Vec> GetNavigationRoute(int32 Student) const { return Navigation->GetActiveRoute(Student); }
    const CampusNav::Grid& GetNavigationGrid() const { return Navigation->GetGrid(); }
    CampusNav::Vec GetNavigationPosition(int32 Student) const { return Navigation->GetPosition(Student); }
    bool UsesRoutePlanner() const { return Navigation && Navigation->UsesRoutePlanner(); }
    bool UsesLivePlanner() const { return Navigation && Navigation->UsesLivePlanner(); }
    void TickPlanner() { Navigation->TickPlanner(); }
    FCampusPlanningStats GetPlanningStats() const { return Navigation->GetPlanningStats(); }
    bool IsNavigation() const { return Navigation.IsValid(); }
    int32 GetObservationSize() const { return Turns ? CampusTurn::ObservationSize : CampusMigration::ObservationSize; }
    int32 GetActionSize() const { return Turns ? CampusTurn::ExitCount : 2; }
    int32 GetGeneration(int32 Student) const { return Turns ? Turns->Generation(Student) : Navigation ? Navigation->Generation(Student) : Leg; }
    int32 GetTimedOut() const { return Navigation ? Navigation->TimedOut : 0; }
    int32 GetLegsCompleted() const { return Turns ? Turns->LegsCompleted : Navigation ? Navigation->LegsCompleted : 0; }
    int32 GetLeg() const { return Leg; }
    int64 GetModelDecisions() const { return Turns ? Turns->Stats.Applied : ModelDecisions; }
    double GetTime() const { return Turns ? Turns->Time : Time; }
    bool IsFinished() const { return Turns ? Turns->Finished() : Navigation ? Navigation->Finished() : Leg == CampusMigration::LegCount; }
    bool UsesModel() const { return Turns || (!Navigation && bLearned); }
    bool HasTrace() const { return Trace.IsValid(); }
    bool HasTraceError() const { return Turns ? Turns->HasTraceError() : Trace && Trace->IsError(); }

private:
    struct FStop
    {
        FString Name;
        int32 Node = INDEX_NONE;
    };
    struct FRoute
    {
        TArray<FVector2f> Points;
        TArray<float> Distances;
        TArray<float> Widths;
    };

    bool LoadRoutes(FString& Error);
    const FRoute& RouteFor(const FCampusMigrationStudent& Student) const;
    void StartLeg();
    void StartStudent(FCampusMigrationStudent& Student);
    void RecordFrame();
    void WriteTrace(const FString& Line);
    void BuildGrid();
    void Observe(int32 Index);
    void Integrate(FCampusMigrationStudent& Student, float DeltaSeconds);
    float CorridorWidth(const FRoute& Route, int32 Segment, float Progress) const;
    bool EntranceClear(FVector2f Position, int32 Stop) const;
    FVector2f Rules(const float* Row) const;

    TUniquePtr<FCampusNavigationDay> Navigation;
    TUniquePtr<FCampusTurnDay> Turns;
    TArray<FStop> Stops;
    TArray<FRoute> Routes;
    TArray<FCampusMigrationStudent> Students;
    TMap<FIntPoint, TArray<int32>> Grid;
    TArray<bool> ReleasedAtStop;
    TUniquePtr<FArchive> Trace;
    int32 GateCount = 0;
    int32 Leg = 0;
    int32 Arrived = 0;
    int32 Active = 0;
    int32 Seed = 301;
    float Yield = 1;
    float LaneWidth = 72;
    float ReleaseSeconds = 30;
    float DwellSeconds = 18;
    double Time = 0;
    double WaveStart = 0;
    double NextWave = -1;
    double NextFrame = 0;
    double ClosePersonSeconds = 0;
    double WalkingPersonSeconds = 0;
    int32 CloseStudents = 0;
    int64 ModelDecisions = 0;
    bool bLearned = true;
};
