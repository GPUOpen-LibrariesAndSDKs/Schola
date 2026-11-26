#pragma once

#include "CampusShortcutRoute.h"
#include "Inference/BackgroundInferenceScheduler.h"

class FJsonObject;

struct FCampusPlannedRoute
{
    TArray<CampusNav::Vec> Points;
    TArray<float> Seconds;
};

struct FCampusPlanningStats
{
    int64 Routes = 0, StudentLegs = 0, ExpandedNodes = 0, FallbackRoutes = 0;
    int64 ModelRows = 0, ModelCalls = 0, InferenceGroups = 0;
    double InferenceLatencyMs = 0;
};

class FCampusLandmarkPlanner
{
public:
    bool Initialize(TConstArrayView<CampusNav::Vec> Stops, const CampusNav::Grid& Grid,
        const TCHAR* Command, bool bBackground, FString& Error);
    bool InitializePrecomputed(TConstArrayView<CampusNav::Vec> Stops, const CampusNav::Grid& Grid,
        const FJsonObject& Table, FString& Error);
    void Tick(const TBitArray<>& Demand, TFunctionRef<void(int32)> OnStarted,
        TFunctionRef<void(int32, TConstArrayView<float>)> OnReady);
    FCampusPlannedRoute Plan(int32 Start, int32 End, TConstArrayView<float> Eta, const CampusNav::Grid& Grid);
    FCampusPlanningStats GetStats() const;
    FCampusPlanningStats Stats;

private:
    static constexpr int32 ObservationSize = 9;
    struct FLink
    {
        int32 To;
        float Seconds;
        TArray<CampusNav::Vec> Interior;
    };
    FCampusPlannedRoute BuildRoute(int32 From, TConstArrayView<const FLink*> Path, const CampusNav::Grid& Grid) const;
    void BuildObservations(int32 Goal, TArrayView<float> Output) const;
    bool LoadGraph(TConstArrayView<CampusNav::Vec> Stops, const CampusNav::Grid& Grid);
    void Deliver(TConstArrayView<float> Actions, double LatencyMs,
        TFunctionRef<void(int32, TConstArrayView<float>)> OnReady);

    TArray<CampusNav::Vec> Points;
    TArray<int32> StopNodes;
    TArray<TArray<FLink>> Links;
    float FeatureScale = 0.f;
    TArray<TArray<float>> Observations;
    TArray<TArray<float>> CachedEta;
    TArray<TArray<int32>> NextHopLinks;
    TArray<int32> Goals;
    TArray<float> SyncActions;
    TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Scheduler;
    TUniquePtr<IBatchInferenceContext> Sync;
    FNPCBackgroundRequest Request;
    FNPCBackgroundResult Result;
};
