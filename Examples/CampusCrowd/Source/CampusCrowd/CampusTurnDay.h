#pragma once

#include "CoreMinimal.h"

namespace CampusTurn
{
constexpr int32 ExitCount = 8;
constexpr int32 FeaturesPerExit = 12;
constexpr int32 ObservationSize = ExitCount * FeaturesPerExit;
constexpr int32 LastStop = 5;
constexpr int32 LatencyBinCount = 1001;
}

struct FCampusTurnStats
{
    int64 Requested = 0, Resolved = 0, Applied = 0, RetracedEdges = 0, Retries = 0;
    int64 LatencyBins[CampusTurn::LatencyBinCount] = {};
    double P99LatencyMsSince(const FCampusTurnStats& Start) const;
};

/** Learned choices at mapped path junctions. Movement interpolates along exported edge geometry. */
class FCampusTurnDay
{
public:
    bool Initialize(int32 Count, const TCHAR* Command, FString& Error, bool bFourDemo = false);
    void Step(float DeltaSeconds, TArray<FTransform>& Transforms);
    int32 GetStudentCount() const { return Students.Num(); }
    FVector2f GetStudentPosition(int32 Index) const { return Students[Index].Position; }
    int32 GetTargetStop(int32 Index) const;
    int32 GetStopColorKey(int32 Stop) const { return StopBuildings[Stop]; }
    bool IsFirstClassOnly() const { return bFirstClassOnly; }
    bool IsStudentVisible(int32 Index) const { return bShowReference || Students[Index].bActive; }
    const TMap<FString, int32>& GetBuildingKeys() const { return BuildingKeys; }
    TConstArrayView<FVector2f> GetReferencePath(int32 Index) const;
    const TCHAR* GetDecision(int32 Index) const;
    static FLinearColor DestinationColor(int32 Key);
    static FRotator OverviewRotation() { return FRotator(-65, -90, 0); }
    bool WantsDecision(int32 Student) const { return Students[Student].PendingIndex != INDEX_NONE; }
    TConstArrayView<int32> GetPendingStudents() const { return PendingStudents; }
    bool IsWalking(int32 Student) const { return Students[Student].bActive; }
    int32 Generation(int32 Student) const { return Students[Student].Generation; }
    bool Finished() const { return Completed == Students.Num(); }
    void WriteObservation(int32 Student, TArrayView<float> Row) const;
    void ApplyAction(int32 Student, TConstArrayView<float> Scores);
    void RetryOverdueDecisions();
    int32 Active = 0, LegsCompleted = 0, Completed = 0;
    double Time = 0.;
    FCampusTurnStats Stats;
    bool HasTraceError() const { return Trace && Trace->IsError(); }

private:
    struct FEdge
    {
        int32 To = INDEX_NONE;
        TArray<FVector2f> Points;
        TArray<float> Distance;
    };
    struct FTraceDecision
    {
        double Time;
        bool bLearned, bRetracing;
    };
    struct FStudent
    {
        int32 Stops[CampusTurn::LastStop + 1] = {};
        int32 Leg = 0, Node = INDEX_NONE, Edge = INDEX_NONE, Segment = 0, Generation = 0, RetracedEdges = 0;
        float Pace = 1.4f, Progress = 0.f;
        double Wake = 0., RequestedAt = 0., AttemptAt = 0.;
        int32 PendingIndex = INDEX_NONE;
        bool bActive = false;
        const TCHAR* Decision = TEXT("At entrance");
        FVector2f Position = FVector2f::ZeroVector, Incoming = FVector2f::ZeroVector, Facing = FVector2f(1.f, 0.f);
        TArray<int32, TInlineAllocator<32>> Visited, Trail;
        TArray<int32> TracePath;
        TArray<FTraceDecision> TraceDecisions;
        double TraceStartTime = 0.;
    };
    bool LoadGraph(FString& Error);
    bool IsLegal(const FStudent& Student, int32 Exit) const;
    void AtNode(FStudent& Student, int32 Index, double Now);
    void SetEdge(FStudent& Student, int32 Edge, bool bRetracing = false);
    void RetryDecision(FStudent& Student, double Now);
    void FinishLeg(FStudent& Student, int32 Index);
    void BuildReferencePath(const FStudent& Student, int32 Index);

    TArray<FVector2f> Points;
    TArray<TArray<FEdge>> Edges;
    TArray<int32> Stops, StopBuildings;
    TMap<FString, int32> BuildingKeys;
    TArray<TArray<FVector2f>> ReferencePaths;
    bool bShowReference = false, bFirstClassOnly = false;
    TArray<TArray<float>> LowerBounds;
    TArray<TArray<uint8>> AllowedExits;
    TArray<TArray<int32>> Buildings;
    TArray<FStudent> Students;
    TArray<int32> PendingStudents;
    TUniquePtr<FArchive> Trace;
    int32 Gates = 0, LegCount = CampusTurn::LastStop;
    float Dwell = 18.f;
    double RetryMs = 250.;
};
