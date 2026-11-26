#include "CampusLandmarkPlanner.h"

#include "Algo/AllOf.h"
#include "Algo/Reverse.h"

#include "Inference/NNEBatchBackendProvider.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Spaces/BoxSpace.h"

namespace
{
constexpr int32 GoalBatchRows = 16384;

bool ReadPoint(const TSharedPtr<FJsonValue>& Value, CampusNav::Vec& Point)
{
    const TArray<TSharedPtr<FJsonValue>>* XY;
    double X, Y;
    if (!Value->TryGetArray(XY) || XY->Num() != 2 || !(*XY)[0]->TryGetNumber(X) || !(*XY)[1]->TryGetNumber(Y))
        return false;
    Point = {float(X), float(Y)};
    return FMath::IsFinite(Point.X) && FMath::IsFinite(Point.Y);
}

void RefineRoute(const CampusNav::Grid& Grid, TArray<CampusNav::Vec>& Points)
{
    // Move each bend toward its adjacent chord while preserving grid clearance.
    const int32 Last = Points.Num() - 1;
    for (int32 Direction : {1, -1})
    {
        for (int32 I = Direction == 1 ? 1 : Last - 1; I > 0 && I < Last; I += Direction)
        {
            const auto Previous = Points[I - 1], Bend = Points[I], Following = Points[I + 1];
            const auto Chord = Following - Previous;
            const float Squared = Chord.X * Chord.X + Chord.Y * Chord.Y;
            if (Squared <= 0.f) continue;
            const auto Offset = Bend - Previous;
            const auto Delta = Previous + Chord * FMath::Clamp(
                (Offset.X * Chord.X + Offset.Y * Chord.Y) / Squared, 0.f, 1.f) - Bend;
            const float Travel = Delta.Length();
            float Low = 0.f, High = 1.f;
            while ((High - Low) * Travel > Grid.Cell)
            {
                const float T = (Low + High) * .5f;
                const auto Candidate = Bend + Delta * T;
                if (CampusNav::SegmentClear(Grid, Previous, Candidate) &&
                    CampusNav::SegmentClear(Grid, Bend, Candidate) &&
                    CampusNav::SegmentClear(Grid, Candidate, Following))
                    Low = T;
                else
                    High = T;
            }
            Points[I] = Bend + Delta * Low;
        }
    }
}
}

bool FCampusLandmarkPlanner::LoadGraph(TConstArrayView<CampusNav::Vec> Stops, const CampusNav::Grid& Grid)
{
    FString Json;
    TSharedPtr<FJsonObject> Root;
    const TArray<TSharedPtr<FJsonValue>> *Positions, *StopIds, *Edges, *Features;
    if (!FFileHelper::LoadFileToString(Json, *(FPaths::ProjectContentDir() / TEXT("Campus/navigation_landmarks.json"))) ||
        !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Root) || !Root ||
        !Root->TryGetArrayField(TEXT("positions"), Positions) || Positions->Num() < Stops.Num() ||
        !Root->TryGetArrayField(TEXT("stops"), StopIds) || StopIds->Num() != Stops.Num() ||
        !Root->TryGetArrayField(TEXT("links"), Edges))
        return false;
    Points.SetNum(Positions->Num());
    Links.SetNum(Points.Num());
    for (int32 I = 0; I < Points.Num(); ++I)
        if (!ReadPoint((*Positions)[I], Points[I])) return false;
    for (int32 I = 0; I < Stops.Num(); ++I)
    {
        int32 Node;
        if (!(*StopIds)[I]->TryGetNumber(Node) || !Points.IsValidIndex(Node) ||
            (Points[Node] - Stops[I]).Length() > .001f || StopNodes.Contains(Node))
            return false;
        Points[Node] = Stops[I];
        StopNodes.Add(Node);
    }
    for (const auto& Edge : *Edges)
    {
        const TArray<TSharedPtr<FJsonValue>> *Row, *Interior;
        int32 A, B;
        double Seconds;
        if (!Edge->TryGetArray(Row) || Row->Num() != 4 || !(*Row)[0]->TryGetNumber(A) || !(*Row)[1]->TryGetNumber(B) ||
            !Points.IsValidIndex(A) || !Points.IsValidIndex(B) || A >= B || !(*Row)[2]->TryGetNumber(Seconds) ||
            !FMath::IsFinite(float(Seconds)) || Seconds <= 0 || !(*Row)[3]->TryGetArray(Interior))
            return false;
        FLink Forward{B, float(Seconds), {}}, Reverse{A, float(Seconds), {}};
        auto Previous = Points[A];
        for (const auto& Value : *Interior)
        {
            CampusNav::Vec Point;
            if (!ReadPoint(Value, Point) || !CampusNav::SegmentClear(Grid, Previous, Point)) return false;
            Forward.Interior.Add(Point);
            Previous = Point;
        }
        if (!CampusNav::SegmentClear(Grid, Previous, Points[B])) return false;
        for (int32 I = Forward.Interior.Num() - 1; I >= 0; --I) Reverse.Interior.Add(Forward.Interior[I]);
        Links[A].Add(MoveTemp(Forward));
        Links[B].Add(MoveTemp(Reverse));
    }
    TBitArray<> Connected(false, Points.Num());
    TArray<int32> Visit{StopNodes[0]};
    Connected[StopNodes[0]] = true;
    for (int32 I = 0; I < Visit.Num(); ++I)
        for (const auto& Link : Links[Visit[I]])
            if (!Connected[Link.To])
            {
                Connected[Link.To] = true;
                Visit.Add(Link.To);
            }
    if (Visit.Num() != Points.Num()) return false;
    if (Root->TryGetNumberField(TEXT("scale"), FeatureScale))
        return FMath::IsFinite(FeatureScale) && FeatureScale > 0.f;
    if (!Root->TryGetArrayField(TEXT("observations_by_stop"), Features) || Features->Num() != Stops.Num())
        return false;

    for (const auto& Goal : *Features)
    {
        const TArray<TSharedPtr<FJsonValue>>* Rows;
        if (!Goal->TryGetArray(Rows) || Rows->Num() != Points.Num()) return false;
        auto& Flat = Observations.AddDefaulted_GetRef();
        Flat.Reserve(Points.Num() * ObservationSize);
        for (const auto& Row : *Rows)
        {
            const TArray<TSharedPtr<FJsonValue>>* Values;
            if (!Row->TryGetArray(Values) || Values->Num() != ObservationSize) return false;
            for (const auto& Value : *Values)
            {
                double Number;
                if (!Value->TryGetNumber(Number) || !FMath::IsFinite(float(Number))) return false;
                Flat.Add(float(Number));
            }
        }
    }
    return true;
}

void FCampusLandmarkPlanner::BuildObservations(int32 Goal, TArrayView<float> Output) const
{
    if (!Observations.IsEmpty())
    {
        FMemory::Memcpy(Output.GetData(), Observations[Goal].GetData(), Output.Num() * sizeof(float));
        return;
    }
    const auto Destination = Points[StopNodes[Goal]] * (1.f / FeatureScale);
    TArray<float> Straight;
    Straight.SetNumUninitialized(Points.Num());
    for (int32 Node = 0; Node < Points.Num(); ++Node)
        Straight[Node] = (Points[Node] * (1.f / FeatureScale) - Destination).Length();
    for (int32 Node = 0; Node < Points.Num(); ++Node)
    {
        const auto From = Points[Node] * (1.f / FeatureScale);
        float Best[] = {MAX_flt, MAX_flt, MAX_flt};
        for (const FLink& Link : Links[Node])
        {
            float Candidate = Link.Seconds * 1.4f / FeatureScale + Straight[Link.To];
            for (float& Value : Best)
                if (Candidate < Value) Swap(Candidate, Value);
        }
        float* Row = Output.GetData() + Node * ObservationSize;
        Row[0] = From.X;
        Row[1] = From.Y;
        Row[2] = Destination.X;
        Row[3] = Destination.Y;
        Row[4] = Straight[Node];
        for (int32 I = 0; I < 3; ++I) Row[5 + I] = Best[FMath::Min(I, Links[Node].Num() - 1)];
        Row[8] = float(Links[Node].Num()) / (Points.Num() - 1);
    }
}

bool FCampusLandmarkPlanner::InitializePrecomputed(TConstArrayView<CampusNav::Vec> Stops, const CampusNav::Grid& Grid,
    const FJsonObject& Table, FString& Error)
{
    const TArray<TSharedPtr<FJsonValue>> *Rows, *Ids;
    if (!LoadGraph(Stops, Grid) || !Table.TryGetArrayField(TEXT("next_hop_by_stop"), Rows) ||
        !Table.TryGetArrayField(TEXT("stop_nodes"), Ids) || Rows->Num() != Stops.Num() || Ids->Num() != Stops.Num())
        return false;
    for (int32 Goal = 0; Goal < Stops.Num(); ++Goal)
    {
        int32 GoalNode;
        const TArray<TSharedPtr<FJsonValue>>* Row;
        if (!(*Ids)[Goal]->TryGetNumber(GoalNode) || GoalNode != StopNodes[Goal] ||
            !(*Rows)[Goal]->TryGetArray(Row) || Row->Num() != Points.Num())
            return false;
        auto& Indices = NextHopLinks.AddDefaulted_GetRef();
        Indices.Init(INDEX_NONE, Points.Num());
        for (int32 Node = 0; Node < Points.Num(); ++Node)
        {
            int32 Next;
            if (!(*Row)[Node]->TryGetNumber(Next)) return false;
            if (Node == GoalNode)
            {
                if (Next != INDEX_NONE) return false;
                continue;
            }
            Indices[Node] = Links[Node].IndexOfByPredicate([Next](const FLink& Link) { return Link.To == Next; });
            if (Indices[Node] == INDEX_NONE) return false;
        }
        // Every parent chain must reach the destination, which also rules out loops.
        TBitArray<> ReachesGoal(false, Points.Num()), Visiting(false, Points.Num());
        ReachesGoal[GoalNode] = true;
        TArray<int32> Chain;
        for (int32 Start = 0; Start < Points.Num(); ++Start)
        {
            Chain.Reset();
            for (int32 Node = Start; !ReachesGoal[Node]; Node = Links[Node][Indices[Node]].To)
            {
                if (Visiting[Node]) return false;
                Visiting[Node] = true;
                Chain.Add(Node);
            }
            for (int32 Node : Chain) ReachesGoal[Node] = true;
        }
    }
    Error.Reset();
    return true;
}

bool FCampusLandmarkPlanner::Initialize(TConstArrayView<CampusNav::Vec> Stops, const CampusNav::Grid& Grid,
    const TCHAR* Command, bool bBackground, FString& Error)
{
    if (!LoadGraph(Stops, Grid))
    {
        Error = TEXT("Invalid campus landmark graph. Regenerate the graph and ETA model with --export-live Content.");
        return false;
    }
    const FString Path = FPaths::ProjectContentDir() / TEXT("Models/landmark_eta.onnx");
    TArray<uint8> Bytes;
    if (!FFileHelper::LoadFileToArray(Bytes, *Path))
    {
        Error = TEXT("Missing campus ETA model at ") + Path;
        return false;
    }
    auto* Provider = NewObject<UNNEBatchBackendProvider>();
    Provider->ModelData = NewObject<UNNEModelData>(Provider);
    Provider->ModelData->Init(TEXT("onnx"), Bytes);
    FBoxSpace ObservationsSpace, ActionSpace;
    for (int32 I = 0; I < ObservationSize; ++I) ObservationsSpace.Add(-1e6f, 1e6f);
    ActionSpace.Add(0.f, 1e6f);
    const FInteractionDefinition Definition(TInstancedStruct<FSpace>::Make<FBoxSpace>(ObservationsSpace),
        TInstancedStruct<FSpace>::Make<FBoxSpace>(ActionSpace));
    FNPCRowLayout Layout;
    const auto Backend = Provider->CreateBackend(Definition, Layout, Error);
    if (!Backend || Layout.ObsSize != ObservationSize || Layout.ActSize != 1)
    {
        Error = TEXT("Could not initialize the campus ETA model: ") + Error;
        return false;
    }
    CachedEta.SetNum(Stops.Num());
    if (bBackground)
    {
        FNPCBackgroundSchedulerSettings Settings;
        FParse::Value(Command, TEXT("Workers="), Settings.WorkerCount);
        FParse::Value(Command, TEXT("MaxBatch="), Settings.MaxBatchSize);
        Settings.WorkerCount = FMath::Clamp(Settings.WorkerCount, 1, 64);
        Settings.MaxBatchSize = FMath::Max(1, Settings.MaxBatchSize);
        Settings.MaxPendingRows = FMath::Max(GoalBatchRows, Points.Num());
        Settings.MaxPendingBytes = FMath::Max(Settings.MaxPendingBytes,
            int64(Settings.MaxPendingRows) * (ObservationSize + 1) * int64(sizeof(float)));
        Settings.DefaultMaxAgeMs = 250.f;
        Scheduler = FNPCBackgroundScheduler::Create(Settings, Backend.ToSharedRef(), &Error);
        return Scheduler.IsValid();
    }
    Sync = Backend->CreateContext();
    if (!Sync) Error = TEXT("Could not create the synchronous campus ETA context");
    return Sync.IsValid();
}

void FCampusLandmarkPlanner::Deliver(TConstArrayView<float> Actions, double LatencyMs,
    TFunctionRef<void(int32, TConstArrayView<float>)> OnReady)
{
    ++Stats.InferenceGroups;
    Stats.InferenceLatencyMs += LatencyMs;
    const bool bValid = Actions.Num() == Goals.Num() * Points.Num();
    for (int32 I = 0; I < Goals.Num(); ++I)
    {
        const int32 Goal = Goals[I];
        if (bValid)
        {
            const auto Eta = Actions.Slice(I * Points.Num(), Points.Num());
            if (Algo::AllOf(Eta, [](float Value) { return FMath::IsFinite(Value) && Value >= 0.f; }))
                CachedEta[Goal].Append(Eta.GetData(), Eta.Num());
        }
        OnReady(Goal, CachedEta[Goal]);
    }
    Goals.Reset();
}

void FCampusLandmarkPlanner::Tick(const TBitArray<>& Demand, TFunctionRef<void(int32)> OnStarted,
    TFunctionRef<void(int32, TConstArrayView<float>)> OnReady)
{
    if (!NextHopLinks.IsEmpty())
    {
        for (TConstSetBitIterator<> It(Demand); It; ++It)
        {
            OnStarted(It.GetIndex());
            OnReady(It.GetIndex(), {});
        }
        return;
    }
    if (Request.IsSet())
    {
        const auto Status = Scheduler->Consume(Request, Result);
        if (Status == ENPCBackgroundStatus::Pending) return;
        Request = {};
        const TConstArrayView<float> Actions = Status == ENPCBackgroundStatus::Completed
            ? TConstArrayView<float>(Result.Actions) : TConstArrayView<float>{};
        const double LatencyMs = Status == ENPCBackgroundStatus::Invalid
            ? 0. : (Result.CompleteSeconds - Result.SubmitSeconds) * 1000.;
        Deliver(Actions, LatencyMs, OnReady);
    }
    const int32 GoalLimit = Scheduler ? FMath::Max(1, GoalBatchRows / Points.Num()) : 1;
    for (TConstSetBitIterator<> It(Demand); It; ++It)
    {
        const int32 Goal = It.GetIndex();
        if (!CachedEta[Goal].IsEmpty())
        {
            OnStarted(Goal);
            OnReady(Goal, CachedEta[Goal]);
        }
        else if (Goals.Num() < GoalLimit)
        {
            Goals.Add(Goal);
        }
    }
    if (Goals.IsEmpty()) return;

    const int32 Rows = Goals.Num() * Points.Num();
    if (Scheduler)
    {
        const auto Status = Scheduler->SubmitWith(Rows, [this](TArrayView<float> Input)
        {
            int32 Offset = 0;
            for (int32 Goal : Goals)
            {
                const int32 Size = Points.Num() * ObservationSize;
                BuildObservations(Goal, Input.Slice(Offset, Size));
                Offset += Size;
            }
        }, {}, Request);
        if (Status == ENPCBackgroundStatus::Busy)
        {
            Goals.Reset();
            return;
        }
        for (int32 Goal : Goals) OnStarted(Goal);
        if (Status != ENPCBackgroundStatus::Pending) Deliver({}, 0., OnReady);
        return;
    }

    for (int32 Goal : Goals) OnStarted(Goal);
    const double Started = FPlatformTime::Seconds();
    SyncActions.SetNumUninitialized(Rows, EAllowShrinking::No);
    int32 Output = 0;
    bool bSuccess = true;
    for (int32 Goal : Goals)
    {
        TArray<float> Features;
        Features.SetNumUninitialized(Points.Num() * ObservationSize);
        BuildObservations(Goal, Features);
        for (int32 I = 0; I < Points.Num(); ++I)
        {
            ++Stats.ModelCalls;
            ++Stats.ModelRows;
            bSuccess &= Sync->RunBatch(Features.GetData() + I * ObservationSize, 1, &SyncActions[Output++]);
        }
    }
    Deliver(bSuccess ? TConstArrayView<float>(SyncActions) : TConstArrayView<float>{},
        (FPlatformTime::Seconds() - Started) * 1000., OnReady);
}

FCampusPlanningStats FCampusLandmarkPlanner::GetStats() const
{
    auto Snapshot = Stats;
    if (Scheduler)
    {
        const auto Background = Scheduler->GetStats();
        Snapshot.ModelRows = Background.BackendRows;
        Snapshot.ModelCalls = Background.BackendCalls;
    }
    return Snapshot;
}

FCampusPlannedRoute FCampusLandmarkPlanner::Plan(int32 Start, int32 End, TConstArrayView<float> Eta,
    const CampusNav::Grid& Grid)
{
    ++Stats.Routes;
    const int32 From = StopNodes[Start], Goal = StopNodes[End];
    TArray<const FLink*, TInlineAllocator<32>> Path;
    if (!NextHopLinks.IsEmpty())
    {
        for (int32 Node = From; Node != Goal;)
        {
            const auto& Link = Links[Node][NextHopLinks[End][Node]];
            Path.Add(&Link);
            Node = Link.To;
        }
        return BuildRoute(From, Path, Grid);
    }
    if (Eta.IsEmpty()) ++Stats.FallbackRoutes;
    const auto Remaining = [&](int32 Node) { return Eta.IsEmpty() || Node == Goal ? 0.f : Eta[Node]; };
    struct FVisit { int32 Node; float Cost, Priority; };
    const auto Earlier = [](const FVisit& A, const FVisit& B)
    {
        if (A.Priority != B.Priority) return A.Priority < B.Priority;
        return A.Cost != B.Cost ? A.Cost < B.Cost : A.Node < B.Node;
    };
    TArray<float, TInlineAllocator<128>> Best;
    TArray<const FLink*, TInlineAllocator<128>> Incoming;
    TArray<int32, TInlineAllocator<128>> Parent;
    Best.Init(TNumericLimits<float>::Max(), Points.Num());
    Incoming.SetNumZeroed(Points.Num());
    Parent.SetNumUninitialized(Points.Num());
    TArray<FVisit, TInlineAllocator<128>> Queue;
    Best[From] = 0;
    Queue.HeapPush({From, 0, Remaining(From)}, Earlier);
    while (!Queue.IsEmpty())
    {
        FVisit Visit;
        Queue.HeapPop(Visit, Earlier, EAllowShrinking::No);
        if (Visit.Cost != Best[Visit.Node]) continue;
        ++Stats.ExpandedNodes;
        if (Visit.Node == Goal) break;
        for (const auto& Link : Links[Visit.Node])
        {
            const float Cost = Visit.Cost + Link.Seconds;
            if (Cost >= Best[Link.To]) continue;
            Best[Link.To] = Cost;
            Incoming[Link.To] = &Link;
            Parent[Link.To] = Visit.Node;
            Queue.HeapPush({Link.To, Cost, Cost + Remaining(Link.To)}, Earlier);
        }
    }
    for (int32 Node = Goal; Node != From; Node = Parent[Node]) Path.Add(Incoming[Node]);
    Algo::Reverse(Path);
    return BuildRoute(From, Path, Grid);
}

FCampusPlannedRoute FCampusLandmarkPlanner::BuildRoute(int32 From, TConstArrayView<const FLink*> Path,
    const CampusNav::Grid& Grid) const
{
    FCampusPlannedRoute Route;
    Route.Points.Add(Points[From]);
    for (const FLink* Link : Path)
    {
        Route.Points.Append(Link->Interior);
        Route.Points.Add(Points[Link->To]);
    }
    RefineRoute(Grid, Route.Points);
    Route.Seconds.SetNumUninitialized(Route.Points.Num());
    CampusNav::ShortcutTimes(Grid, Route.Points.GetData(), Route.Points.Num(), Route.Seconds.GetData());
    return Route;
}
