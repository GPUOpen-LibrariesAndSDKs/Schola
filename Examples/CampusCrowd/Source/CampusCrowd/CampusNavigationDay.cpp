#include "CampusNavigationDay.h"
#include "Async/ParallelFor.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"

namespace
{
constexpr int32 LastStop = 5;

struct FGridHeader
{
    char Magic[4];
    int32 Width, Height;
    float Cell, OriginX, OriginY, Scale;
};
static_assert(sizeof(FGridHeader) == 28);
}

bool FCampusNavigationDay::LoadMap(FString& Error)
{
    const FString Directory = FPaths::ProjectContentDir() / TEXT("Campus");
    Error = TEXT("Invalid navigation grid. Regenerate it with Tools/build_navigation_map.py.");
    if (!FFileHelper::LoadFileToArray(GridBytes, *(Directory / TEXT("navigation_grid.bin"))) ||
        GridBytes.Num() < sizeof(FGridHeader))
        return false;
    FGridHeader Header;
    FMemory::Memcpy(&Header, GridBytes.GetData(), sizeof(Header));
    if (FMemory::Memcmp(Header.Magic, "CNV1", 4) || Header.Width <= 0 || Header.Height <= 0 ||
        !FMath::IsFinite(Header.Cell) || Header.Cell <= 0 || !FMath::IsFinite(Header.Scale) || Header.Scale <= 0 ||
        !FMath::IsFinite(Header.OriginX) || !FMath::IsFinite(Header.OriginY) ||
        GridBytes.Num() != sizeof(Header) + (int64(Header.Width) * Header.Height + 7) / 8)
        return false;
    Map = {GridBytes.GetData() + sizeof(Header), Header.Width, Header.Height, Header.Cell, {Header.OriginX, Header.OriginY}};

    Error = TEXT("Missing or invalid navigation stops. Regenerate them with Tools/build_navigation_map.py.");
    FString Json;
    TSharedPtr<FJsonObject> Root;
    const TArray<TSharedPtr<FJsonValue>>* Rows;
    if (!FFileHelper::LoadFileToString(Json, *(Directory / TEXT("navigation_stops.json"))) ||
        !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Root) || !Root ||
        !Root->TryGetNumberField(TEXT("gate_count"), GateCount) || GateCount != 4 ||
        !Root->TryGetArrayField(TEXT("stops"), Rows) || Rows->Num() < GateCount + 2)
        return false;
    TMap<FString, int32> BuildingIndices;
    for (const auto& Row : *Rows)
    {
        const TSharedPtr<FJsonObject>* Object;
        const TArray<TSharedPtr<FJsonValue>>* Position;
        double X, Y;
        if (!Row->TryGetObject(Object) || !(*Object)->TryGetArrayField(TEXT("position"), Position) ||
            Position->Num() != 2 || !(*Position)[0]->TryGetNumber(X) || !(*Position)[1]->TryGetNumber(Y) ||
            !FMath::IsFinite(X) || !FMath::IsFinite(Y))
            return false;
        const CampusNav::Vec Stop{float(X), float(Y)};
        if (Map.Clearance(Stop, {1, 0}, .01f) <= 0) return false;
        const int32 StopIndex = Stops.Add(Stop);
        if (StopIndex >= GateCount)
        {
            FString BuildingId;
            if (!(*Object)->TryGetStringField(TEXT("building_id"), BuildingId) || BuildingId.IsEmpty())
                BuildingId = FString::Printf(TEXT("legacy:%d"), StopIndex);
            int32* BuildingIndex = BuildingIndices.Find(BuildingId);
            if (!BuildingIndex)
                BuildingIndex = &BuildingIndices.Add(BuildingId, EntrancesByBuilding.AddDefaulted());
            EntrancesByBuilding[*BuildingIndex].Add(StopIndex);
        }
    }
    if (EntrancesByBuilding.Num() < 2) return false;
    Error.Reset();
    return true;
}

bool FCampusNavigationDay::LoadRoutes(const FString& Path, FString& Error)
{
    Error = FString::Printf(TEXT("Missing or invalid navigation routes at %s. Regenerate them with Tools/train_waypoint_navigation.py --export-routes."), *Path);
    FString Json;
    TSharedPtr<FJsonObject> Root;
    const TArray<TSharedPtr<FJsonValue>>* Rows;
    if (!FFileHelper::LoadFileToString(Json, *Path) ||
        !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Root) || !Root)
        return false;
    if (Root->HasField(TEXT("next_hop_by_stop")))
    {
        Planner = MakeUnique<FCampusLandmarkPlanner>();
        if (!Planner->InitializePrecomputed(Stops, Map, *Root, Error)) return false;
        bPrecomputed = true;
        UE_LOG(LogTemp, Display, TEXT("Loaded exact next-hop routes from %s"), *Path);
        return true;
    }
    if (!Root->TryGetArrayField(TEXT("routes"), Rows) || Rows->Num() != Stops.Num() * (Stops.Num() - 1))
        return false;

    Routes.SetNum(Stops.Num() * Stops.Num());
    for (const auto& Row : *Rows)
    {
        const TSharedPtr<FJsonObject>* Object;
        const TArray<TSharedPtr<FJsonValue>>* Points;
        int32 Start, End;
        if (!Row->TryGetObject(Object) || !(*Object)->TryGetNumberField(TEXT("start"), Start) ||
            !(*Object)->TryGetNumberField(TEXT("end"), End) || Start == End ||
            !Stops.IsValidIndex(Start) || !Stops.IsValidIndex(End) ||
            !(*Object)->TryGetArrayField(TEXT("points"), Points) || Points->Num() < 2)
            return false;
        auto& Route = Routes[Start * Stops.Num() + End];
        auto& Vertices = Route.Points;
        if (!Vertices.IsEmpty()) return false;
        Vertices.Reserve(Points->Num());
        for (int32 Index = 0; Index < Points->Num(); ++Index)
        {
            const TArray<TSharedPtr<FJsonValue>>* Pair;
            double X, Y;
            if (!(*Points)[Index]->TryGetArray(Pair) || Pair->Num() != 2 ||
                !(*Pair)[0]->TryGetNumber(X) || !(*Pair)[1]->TryGetNumber(Y) ||
                !FMath::IsFinite(X) || !FMath::IsFinite(Y))
                return false;
            CampusNav::Vec Position{float(X), float(Y)};
            if (Index == 0 || Index == Points->Num() - 1)
            {
                const CampusNav::Vec Stop = Stops[Index == 0 ? Start : End];
                if ((Position - Stop).Length() > .001f) return false;
                Position = Stop;
            }
            if (!Vertices.IsEmpty() &&
                ((Position - Vertices.Last()).Length() <= 0.f || !CampusNav::SegmentClear(Map, Vertices.Last(), Position)))
                return false;
            Vertices.Add(Position);
        }
        Route.Seconds.SetNumUninitialized(Vertices.Num());
        CampusNav::ShortcutTimes(Map, Vertices.GetData(), Vertices.Num(), Route.Seconds.GetData());
    }
    Error.Reset();
    UE_LOG(LogTemp, Display, TEXT("Loaded %d navigation routes from %s"), Rows->Num(), *Path);
    return true;
}

bool FCampusNavigationDay::Initialize(int32 Count, const TCHAR* Command, bool bBackground, FString& Error)
{
    if (!LoadMap(Error)) return false;
    FString RoutePath;
    if (FParse::Value(Command, TEXT("NavigationRoutes="), RoutePath))
    {
        if (!LoadRoutes(RoutePath, Error)) return false;
    }
    else
    {
        Planner = MakeUnique<FCampusLandmarkPlanner>();
        if (!Planner->Initialize(Stops, Map, Command, bBackground, Error)) return false;
        UE_LOG(LogTemp, Display, TEXT("Live landmark routing: %s"),
            bBackground ? TEXT("Schola background") : TEXT("synchronous"));
    }
    if (Planner)
    {
        WaitingByPair.SetNum(Stops.Num() * Stops.Num());
        SubmittedByPair.SetNum(WaitingByPair.Num());
        PendingGoals.Init(false, Stops.Num());
    }
    int32 Seed = 301;
    float Release = 5;
    FParse::Value(Command, TEXT("MigrationSeed="), Seed);
    FParse::Value(Command, TEXT("MigrationRelease="), Release);
    FParse::Value(Command, TEXT("MigrationDwell="), Dwell);
    FParse::Value(Command, TEXT("NavigationLegSeconds="), LegSeconds);
    Release = FMath::Clamp(Release, 0.f, 600.f);
    Dwell = FMath::Clamp(Dwell, 0.f, 600.f);
    LegSeconds = FMath::Clamp(LegSeconds, 1.f, 10000.f);
    FRandomStream Random(Seed);
    Students.SetNum(Count);
    BatchTotals.SetNum(FMath::DivideAndRoundUp(Count, StudentsPerBatch));
    for (auto& Student : Students)
    {
        Student.Stops[0] = Random.RandHelper(GateCount);
        Student.Stops[LastStop] = Random.RandHelper(GateCount);
        int32 PreviousBuilding = INDEX_NONE;
        for (int32 Leg = 1; Leg < LastStop; ++Leg)
        {
            int32 Building = Random.RandHelper(EntrancesByBuilding.Num() - int32(PreviousBuilding != INDEX_NONE));
            if (PreviousBuilding != INDEX_NONE && Building >= PreviousBuilding) ++Building;
            const auto& Entrances = EntrancesByBuilding[Building];
            Student.Stops[Leg] = Entrances.Num() == 1 ? Entrances[0] : Entrances[Random.RandHelper(Entrances.Num())];
            PreviousBuilding = Building;
        }
        Student.Wake = Random.FRandRange(0.f, Release);
        Student.Motion.Position = Stops[Student.Stops[0]];
    }
    return true;
}

TConstArrayView<CampusNav::Vec> FCampusNavigationDay::GetActiveRoute(int32 Index) const
{
    const auto& Student = Students[Index];
    if (!Student.bWalking) return {};
    if (Student.Route) return Student.Route->Points;
    return Routes[Student.Stops[Student.Leg] * Stops.Num() + Student.Stops[Student.Leg + 1]].Points;
}

void FCampusNavigationDay::TickPlanner()
{
    const auto StartGroup = [this](int32 Goal)
    {
        for (int32 Start = 0; Start < Stops.Num(); ++Start)
        {
            const int32 Pair = Start * Stops.Num() + Goal;
            Swap(WaitingByPair[Pair], SubmittedByPair[Pair]);
        }
        PendingGoals[Goal] = false;
    };
    const auto CompleteGroup = [this](int32 Goal, TConstArrayView<float> Eta)
    {
        for (int32 Start = 0; Start < Stops.Num(); ++Start)
        {
            auto& Waiting = SubmittedByPair[Start * Stops.Num() + Goal];
            if (Waiting.IsEmpty()) continue;
            const auto Route = MakeShared<FCampusPlannedRoute, ESPMode::ThreadSafe>(Planner->Plan(Start, Goal, Eta, Map));
            for (int32 Index : Waiting) Students[Index].Route = Route;
            Planner->Stats.StudentLegs += Waiting.Num();
            Waiting.Reset();
        }
    };
    Planner->Tick(PendingGoals, StartGroup, CompleteGroup);
}

void FCampusNavigationDay::Step(float Dt, TArray<FTransform>& Transforms)
{
    if (Finished()) return;
    Time += Dt;
    ParallelFor(TEXT("CampusNavigation"), BatchTotals.Num(), 1, [&](int32 Batch)
    {
        auto& Totals = BatchTotals[Batch];
        Totals.Active = Totals.Completed = Totals.TimedOut = Totals.LegsCompleted = 0;
        Totals.Departing.Reset();
        const int32 First = Batch * StudentsPerBatch;
        const int32 EndIndex = FMath::Min(First + StudentsPerBatch, Students.Num());
        for (int32 Index = First; Index < EndIndex; ++Index)
        {
            auto& Student = Students[Index];
            if (!Student.bWalking && Student.Leg < LastStop && !Student.bTimedOut && Student.Wake <= Time)
            {
                if (Planner && !Student.Route)
                {
                    if (!Student.bWaitingForRoute)
                    {
                        Student.bWaitingForRoute = true;
                        Totals.Departing.Add(Index);
                    }
                    continue;
                }
                Student.bWaitingForRoute = false;
                Student.bWalking = true;
                Student.Deadline = Time + LegSeconds;
                Student.Motion.Velocity = {};
                Student.Shortcut = {};
                ++Totals.Active;
            }
            if (Student.bWalking)
            {
                const int32 Start = Student.Stops[Student.Leg], End = Student.Stops[Student.Leg + 1];
                const auto& Route = Student.Route ? *Student.Route : Routes[Start * Stops.Num() + End];
                CampusNav::WalkShortcut(Map, Student.Motion, Route.Points.GetData(), Route.Seconds.GetData(),
                    Route.Points.Num(), Student.Shortcut, Dt);
                const bool bReached = CampusNav::Arrived(Map, Student.Motion, Stops[End]);
                if (bReached || Time >= Student.Deadline)
                {
                    Student.bWalking = false;
                    Student.Route.Reset();
                    --Totals.Active;
                    if (bReached)
                    {
                        ++Totals.LegsCompleted;
                        if (++Student.Leg == LastStop) ++Totals.Completed;
                        else Student.Wake = Time + Dwell;
                    }
                    else
                    {
                        Student.bTimedOut = true;
                        ++Totals.TimedOut;
                    }
                }
            }
            const auto P = Student.Motion.Position;
            const auto V = Student.Motion.Velocity;
            Transforms[Index] = Student.bWalking
                ? FTransform(FRotator(0, FMath::RadiansToDegrees(FMath::Atan2(V.Y, V.X)), 0),
                    FVector(100 * P.X, 100 * P.Y, 94), FVector(.6f, .6f, 1.8f))
                : FTransform(FRotator::ZeroRotator, FVector(0, 0, -1000), FVector::ZeroVector);
        }
    }, BatchTotals.Num() < 4 ? EParallelForFlags::ForceSingleThread : EParallelForFlags::Unbalanced);
    for (const auto& Totals : BatchTotals)
    {
        Active += Totals.Active;
        Completed += Totals.Completed;
        TimedOut += Totals.TimedOut;
        LegsCompleted += Totals.LegsCompleted;
        for (int32 Index : Totals.Departing)
        {
            const auto& Student = Students[Index];
            const int32 Start = Student.Stops[Student.Leg], End = Student.Stops[Student.Leg + 1];
            WaitingByPair[Start * Stops.Num() + End].Add(Index);
            PendingGoals[End] = true;
        }
    }
    if (Finished())
    {
        UE_LOG(LogTemp, Display, TEXT("Campus navigation finished: %d schedules complete, %d timed out, %d legs reached"),
            Completed, TimedOut, LegsCompleted);
        if (Planner)
        {
            const auto Stats = Planner->GetStats();
            UE_LOG(LogTemp, Display, TEXT("Campus planning: %lld student legs, %lld routes, %lld fallback routes, %lld rows across %lld model calls"),
                Stats.StudentLegs, Stats.Routes, Stats.FallbackRoutes, Stats.ModelRows, Stats.ModelCalls);
        }
    }
}
