#include "CampusTurnDay.h"

#include "Algo/Reverse.h"

#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"

namespace
{
bool ReadPoint(const TSharedPtr<FJsonValue>& Value, FVector2f& Point)
{
    const TArray<TSharedPtr<FJsonValue>>* Row;
    double X, Y;
    if (!Value->TryGetArray(Row) || Row->Num() != 2 || !(*Row)[0]->TryGetNumber(X) || !(*Row)[1]->TryGetNumber(Y) ||
        !FMath::IsFinite(float(X)) || !FMath::IsFinite(float(Y)))
        return false;
    Point = FVector2f(X, Y);
    return true;
}
}

double FCampusTurnStats::P99LatencyMsSince(const FCampusTurnStats& Start) const
{
    const int64 Target = FMath::CeilToInt64((Applied - Start.Applied) * .99);
    if (Target <= 0) return 0.;
    int64 Samples = 0;
    for (int32 Bin = 0; Bin < CampusTurn::LatencyBinCount; ++Bin)
    {
        Samples += LatencyBins[Bin] - Start.LatencyBins[Bin];
        if (Samples >= Target) return Bin;
    }
    return CampusTurn::LatencyBinCount - 1;
}

bool FCampusTurnDay::LoadGraph(FString& Error)
{
    FString Text;
    TSharedPtr<FJsonObject> Root;
    const TArray<TSharedPtr<FJsonValue>> *Positions, *Links, *StopRows, *AnchorRows, *AllowedRows;
    int32 ExitCount, FeatureCount;
    Error = TEXT("Invalid Campus/turn_paths.json. Regenerate it with Tools/train_turn_policy.py --export Content.");
    if (!FFileHelper::LoadFileToString(Text, *(FPaths::ProjectContentDir() / TEXT("Campus/turn_paths.json"))) ||
        !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root) || !Root ||
        !Root->TryGetArrayField(TEXT("positions"), Positions) || Positions->IsEmpty() ||
        !Root->TryGetArrayField(TEXT("links"), Links) ||
        !Root->TryGetArrayField(TEXT("stops"), StopRows) ||
        !Root->TryGetNumberField(TEXT("gate_count"), Gates) || Gates <= 0 || StopRows->Num() < Gates + 2 ||
        !Root->TryGetNumberField(TEXT("exits"), ExitCount) || ExitCount != CampusTurn::ExitCount ||
        !Root->TryGetNumberField(TEXT("features"), FeatureCount) || FeatureCount != CampusTurn::FeaturesPerExit ||
        !Root->TryGetArrayField(TEXT("anchor_distance_cm"), AnchorRows) || AnchorRows->IsEmpty() ||
        !Root->TryGetArrayField(TEXT("allowed_exit_bits_by_stop"), AllowedRows) || AllowedRows->Num() != StopRows->Num())
        return false;
    for (const auto& Value : *Positions)
        if (!ReadPoint(Value, Points.AddDefaulted_GetRef())) return false;
    Edges.SetNum(Points.Num());
    for (const auto& Value : *Links)
    {
        const TArray<TSharedPtr<FJsonValue>> *Row, *Interior;
        int32 A, B;
        if (!Value->TryGetArray(Row) || Row->Num() != 3 || !(*Row)[0]->TryGetNumber(A) ||
            !(*Row)[1]->TryGetNumber(B) || !Points.IsValidIndex(A) || !Points.IsValidIndex(B) || A >= B ||
            !(*Row)[2]->TryGetArray(Interior))
            return false;
        auto& Forward = Edges[A].AddDefaulted_GetRef();
        auto& Reverse = Edges[B].AddDefaulted_GetRef();
        Forward.To = B;
        Reverse.To = A;
        Forward.Points.Add(Points[A]);
        for (const auto& Point : *Interior)
            if (!ReadPoint(Point, Forward.Points.AddDefaulted_GetRef())) return false;
        Forward.Points.Add(Points[B]);
        for (int32 I = Forward.Points.Num() - 1; I >= 0; --I) Reverse.Points.Add(Forward.Points[I]);
        for (FEdge* Edge : {&Forward, &Reverse})
        {
            Edge->Distance.Add(0.f);
            for (int32 I = 1; I < Edge->Points.Num(); ++I)
            {
                const float Length = FVector2f::Distance(Edge->Points[I], Edge->Points[I - 1]);
                if (Length <= 0.f || !FMath::IsFinite(Length)) return false;
                Edge->Distance.Add(Edge->Distance.Last() + Length);
            }
        }
    }
    for (auto& Node : Edges)
    {
        if (Node.IsEmpty() || Node.Num() > CampusTurn::ExitCount) return false;
        Node.Sort([](const FEdge& A, const FEdge& B) { return A.To < B.To; });
        for (int32 I = 1; I < Node.Num(); ++I)
            if (Node[I - 1].To == Node[I].To) return false;
    }
    for (const auto& Value : *StopRows)
    {
        const TSharedPtr<FJsonObject>* Row;
        int32 Node;
        if (!Value->TryGetObject(Row) || !(*Row)->TryGetNumberField(TEXT("node"), Node) ||
            !Points.IsValidIndex(Node) || Stops.Contains(Node))
            return false;
        const int32 Index = Stops.Add(Node);
        if (Index < Gates)
        {
            StopBuildings.Add(-Index - 1);
            continue;
        }
        FString Id;
        if (!(*Row)->TryGetStringField(TEXT("building_id"), Id) || Id.IsEmpty())
            Id = FString::Printf(TEXT("legacy:%d"), Index);
        int32* Group = BuildingKeys.Find(Id);
        if (!Group) Group = &BuildingKeys.Add(Id, Buildings.AddDefaulted());
        StopBuildings.Add(*Group);
        Buildings[*Group].Add(Index);
    }
    if (Buildings.Num() < 2) return false;

    TArray<TArray<int32>> Anchors;
    for (const auto& Value : *AnchorRows)
    {
        const TArray<TSharedPtr<FJsonValue>>* Row;
        if (!Value->TryGetArray(Row) || Row->Num() != Points.Num()) return false;
        auto& Distances = Anchors.AddDefaulted_GetRef();
        for (const auto& Cell : *Row)
        {
            int32 Distance;
            if (!Cell->TryGetNumber(Distance) || Distance < 0) return false;
            Distances.Add(Distance);
        }
    }
    for (int32 Goal = 0; Goal < Stops.Num(); ++Goal)
    {
        auto& Bounds = LowerBounds.AddDefaulted_GetRef();
        Bounds.Init(0.f, Points.Num());
        for (const auto& Anchor : Anchors)
            for (int32 Node = 0; Node < Points.Num(); ++Node)
                Bounds[Node] = FMath::Max(Bounds[Node], FMath::Abs(Anchor[Node] - Anchor[Stops[Goal]]) / 100.f);
        const TArray<TSharedPtr<FJsonValue>>* Masks;
        if (!(*AllowedRows)[Goal]->TryGetArray(Masks) || Masks->Num() != Points.Num()) return false;
        auto& Allowed = AllowedExits.AddDefaulted_GetRef();
        for (int32 Node = 0; Node < Points.Num(); ++Node)
        {
            int32 Mask;
            if (!(*Masks)[Node]->TryGetNumber(Mask) || Mask < 0 || Mask >= (1 << Edges[Node].Num()) ||
                (Node == Stops[Goal] ? Mask != 0 : Mask == 0)) return false;
            Allowed.Add(uint8(Mask));
        }
        // Walk the allowed graph in reverse. Every node must have a route to the goal.
        TBitArray<> Reaches(false, Points.Num());
        TArray<int32> Queue{Stops[Goal]};
        Reaches[Stops[Goal]] = true;
        for (int32 Index = 0; Index < Queue.Num(); ++Index)
        {
            const int32 Node = Queue[Index];
            for (const FEdge& Edge : Edges[Node])
            {
                if (Reaches[Edge.To]) continue;
                const int32 Reverse = Edges[Edge.To].IndexOfByPredicate([Node](const FEdge& Link) { return Link.To == Node; });
                if (Reverse == INDEX_NONE || (Allowed[Edge.To] & (1 << Reverse)) == 0) continue;
                Reaches[Edge.To] = true;
                Queue.Add(Edge.To);
            }
        }
        if (Queue.Num() != Points.Num()) return false;
    }
    Error.Reset();
    return true;
}

bool FCampusTurnDay::Initialize(int32 Count, const TCHAR* Command, FString& Error, bool bFourDemo)
{
    if (!LoadGraph(Error)) return false;
    int32 Seed = 301, Start = INDEX_NONE, Goal = INDEX_NONE;
    float Release = 30.f;
    bShowReference = bFourDemo;
    const bool bStart = !bFourDemo && FParse::Value(Command, TEXT("TurnStart="), Start);
    const bool bGoal = !bFourDemo && FParse::Value(Command, TEXT("TurnGoal="), Goal);
    if (bStart || bGoal)
    {
        if (!Stops.IsValidIndex(Start) || !Stops.IsValidIndex(Goal) || Start == Goal)
        {
            Error = TEXT("TurnStart and TurnGoal must be different valid stop indices.");
            return false;
        }
        LegCount = 1;
    }
    bFirstClassOnly = !bFourDemo && !bStart && !bGoal && FParse::Param(Command, TEXT("FirstClassOnly"));
    FParse::Value(Command, TEXT("MigrationSeed="), Seed);
    FParse::Value(Command, TEXT("MigrationRelease="), Release);
    FParse::Value(Command, TEXT("MigrationDwell="), Dwell);
    FParse::Value(Command, TEXT("TurnRetryMs="), RetryMs);
    RetryMs = FMath::Clamp(RetryMs, 100., 10000.);
    Release = FMath::Clamp(Release, 0.f, 600.f);
    Dwell = FMath::Clamp(Dwell, 0.f, 600.f);
    FString TracePath;
    if (FParse::Value(Command, TEXT("TurnTrace="), TracePath))
    {
        Trace.Reset(IFileManager::Get().CreateFileWriter(*TracePath));
        if (!Trace)
        {
            Error = TEXT("Could not open turn trace ") + TracePath;
            return false;
        }
    }

    FRandomStream Random(bFourDemo ? 2429 : Seed);
    Students.SetNum(Count);
    if (bShowReference) ReferencePaths.SetNum(Count);
    for (int32 Index = 0; Index < Count; ++Index)
    {
        auto& Student = Students[Index];
        if (LegCount == 1)
        {
            Student.Stops[0] = Start;
            Student.Stops[1] = Goal;
        }
        else
        {
            Student.Stops[0] = Random.RandHelper(Gates);
            Student.Stops[LegCount] = Random.RandHelper(Gates);
            int32 PreviousBuilding = INDEX_NONE;
            for (int32 I = 1; I < LegCount; ++I)
            {
                int32 Building = Random.RandHelper(Buildings.Num() - int32(PreviousBuilding != INDEX_NONE));
                if (PreviousBuilding != INDEX_NONE && Building >= PreviousBuilding) ++Building;
                const auto& Entrances = Buildings[Building];
                Student.Stops[I] = Entrances[Random.RandHelper(Entrances.Num())];
                PreviousBuilding = Building;
            }
        }
        Student.Wake = Random.FRandRange(0.f, Release);
        Student.Pace = Random.FRandRange(1.15f, 1.8f);
        Student.Position = Points[Stops[Student.Stops[0]]];
        if (bShowReference) BuildReferencePath(Student, Index);
    }
    // Preserve the normal seeded schedule and stop after the first classroom arrival.
    if (bFirstClassOnly) LegCount = 1;
    return true;
}

bool FCampusTurnDay::IsLegal(const FStudent& Student, int32 Exit) const
{
    const uint8 Mask = AllowedExits[Student.Stops[Student.Leg + 1]][Student.Node];
    return (Mask & (1 << Exit)) != 0 && !Student.Visited.Contains(Edges[Student.Node][Exit].To);
}

void FCampusTurnDay::SetEdge(FStudent& Student, int32 Edge, bool bRetracing)
{
    const bool bLearned = Student.PendingIndex != INDEX_NONE;
    if (bLearned)
    {
        const int32 Index = Student.PendingIndex;
        PendingStudents.RemoveAtSwap(Index, 1, EAllowShrinking::No);
        if (PendingStudents.IsValidIndex(Index)) Students[PendingStudents[Index]].PendingIndex = Index;
        Student.PendingIndex = INDEX_NONE;
        ++Stats.Resolved;
    }
    if (!bRetracing) Student.Trail.Add(Student.Node);
    if (bShowReference)
    {
        if (bRetracing) Student.Decision = TEXT("Retrace edge");
        else if (!bLearned) Student.Decision = TEXT("Follow path");
        else if (Student.Incoming.IsNearlyZero()) Student.Decision = TEXT("Choose exit");
        else
        {
            const auto& Path = Edges[Student.Node][Edge].Points;
            const FVector2f Heading = (Path[1] - Path[0]).GetSafeNormal();
            const float Cross = Student.Incoming.X * Heading.Y - Student.Incoming.Y * Heading.X;
            const float Angle = FMath::RadiansToDegrees(FMath::Atan2(Cross, FVector2f::DotProduct(Student.Incoming, Heading)));
            constexpr float StraightDegrees = 23.f, UTurnDegrees = 155.f;
            Student.Decision = FMath::Abs(Angle) >= UTurnDegrees ? TEXT("Model: U-turn") :
                FMath::Abs(Angle) <= StraightDegrees ? TEXT("Model: straight") : Angle > 0 ? TEXT("Model: right") : TEXT("Model: left");
        }
    }
    Student.Edge = Edge;
    Student.Segment = 0;
    Student.Progress = 0.f;
    if (Trace)
    {
        Student.TracePath.Add(Edges[Student.Node][Edge].To);
        Student.TraceDecisions.Add({Time, bLearned, bRetracing});
    }
}

void FCampusTurnDay::RetryDecision(FStudent& Student, double Now)
{
    Student.AttemptAt = Now;
    ++Student.Generation;
    ++Stats.Retries;
}

void FCampusTurnDay::RetryOverdueDecisions()
{
    const double Now = FPlatformTime::Seconds();
    for (int32 Index : PendingStudents)
    {
        auto& Student = Students[Index];
        if ((Now - Student.AttemptAt) * 1000. >= RetryMs) RetryDecision(Student, Now);
    }
}

void FCampusTurnDay::AtNode(FStudent& Student, int32 Index, double Now)
{
    Student.Edge = INDEX_NONE;
    int32 Count = 0, Forced = INDEX_NONE;
    const auto& Options = Edges[Student.Node];
    for (int32 I = 0; I < Options.Num(); ++I)
        if (IsLegal(Student, I)) { ++Count; Forced = I; }
    if (Count == 0)
    {
        // Unwind the walked route until an unvisited exit becomes available.
        check(!Student.Trail.IsEmpty()); // LoadGraph verifies reachability after masking.
        const int32 Previous = Student.Trail.Pop(EAllowShrinking::No);
        const int32 Reverse = Options.IndexOfByPredicate([Previous](const FEdge& Edge) { return Edge.To == Previous; });
        check(Reverse != INDEX_NONE);
        ++Student.RetracedEdges;
        ++Stats.RetracedEdges;
        SetEdge(Student, Reverse, true);
    }
    else if (Count == 1) SetEdge(Student, Forced);
    else
    {
        Student.PendingIndex = PendingStudents.Add(Index);
        Student.RequestedAt = Student.AttemptAt = Now;
        ++Student.Generation;
        ++Stats.Requested;
    }
}

void FCampusTurnDay::WriteObservation(int32 Index, TArrayView<float> Row) const
{
    check(Row.Num() == CampusTurn::ObservationSize);
    FMemory::Memzero(Row.GetData(), Row.Num() * sizeof(float));
    const FStudent& Student = Students[Index];
    if (Student.PendingIndex == INDEX_NONE) return;
    const int32 Stop = Student.Stops[Student.Leg + 1];
    const int32 Goal = Stops[Stop];
    const FVector2f At = Points[Student.Node], Delta = Points[Goal] - At;
    const float Distance = Delta.Size();
    const auto& Options = Edges[Student.Node];
    for (int32 I = 0; I < Options.Num(); ++I)
    {
        const auto& Edge = Options[I];
        const FVector2f Heading = (Edge.Points[1] - At).GetSafeNormal();
        const float Left = FVector2f::Distance(Points[Goal], Points[Edge.To]);
        float* Out = Row.GetData() + I * CampusTurn::FeaturesPerExit;
        Out[0] = LowerBounds[Stop][Edge.To] / 100;
        Out[1] = (LowerBounds[Stop][Student.Node] - LowerBounds[Stop][Edge.To]) / 100;
        Out[2] = Delta.X / 100; Out[3] = Delta.Y / 100;
        Out[4] = Heading.X; Out[5] = Heading.Y;
        Out[6] = FVector2f::DotProduct(Student.Incoming, Heading);
        Out[7] = Student.Incoming.X * Heading.Y - Student.Incoming.Y * Heading.X;
        Out[8] = Edge.Distance.Last() / 100;
        Out[9] = Left / 100; Out[10] = (Distance - Left) / 100;
        Out[11] = Edge.To == Goal ? 1.f : 0.f;
    }
}

void FCampusTurnDay::ApplyAction(int32 Index, TConstArrayView<float> Scores)
{
    FStudent& Student = Students[Index];
    if (Student.PendingIndex == INDEX_NONE) return;
    const double Now = FPlatformTime::Seconds();
    check(Scores.Num() == CampusTurn::ExitCount);
    const auto& Options = Edges[Student.Node];
    int32 Best = INDEX_NONE;
    for (int32 I = 0; I < Options.Num(); ++I)
        if (IsLegal(Student, I) && FMath::IsFinite(Scores[I]) && (Best == INDEX_NONE || Scores[I] > Scores[Best]))
            Best = I;
    if (Best == INDEX_NONE) RetryDecision(Student, Now);
    else
    {
        ++Stats.Applied;
        const double AgeMs = (Now - Student.RequestedAt) * 1000.;
        ++Stats.LatencyBins[FMath::Clamp(FMath::CeilToInt32(AgeMs), 0, CampusTurn::LatencyBinCount - 1)];
        SetEdge(Student, Best);
    }
}

void FCampusTurnDay::FinishLeg(FStudent& Student, int32 Index)
{
    if (Trace)
    {
        FString Line = FString::Printf(
            TEXT("{\"student\":%d,\"leg\":%d,\"retraced_edges\":%d,\"start_stop\":%d,\"goal_stop\":%d,")
            TEXT("\"start_time\":%.3f,\"arrival_time\":%.3f,\"pace_mps\":%.6f,\"landmarks\":["),
            Index, Student.Leg + 1, Student.RetracedEdges, Student.Stops[Student.Leg], Student.Stops[Student.Leg + 1],
            Student.TraceStartTime, Time, Student.Pace);
        for (int32 I = 0; I < Student.TracePath.Num(); ++I)
        {
            if (I > 0) Line.AppendChar(',');
            Line.AppendInt(Student.TracePath[I]);
        }
        Line += TEXT("],\"decisions\":[");
        for (int32 I = 0; I < Student.TraceDecisions.Num(); ++I)
        {
            if (I > 0) Line.AppendChar(',');
            const auto& Decision = Student.TraceDecisions[I];
            Line.Appendf(TEXT("[%.3f,\"%s\"]"), Decision.Time, Decision.bRetracing ? TEXT("retrace") :
                Decision.bLearned ? TEXT("model") : TEXT("forced"));
        }
        Line += TEXT("]}\n");
        FTCHARToUTF8 Utf8(*Line);
        Trace->Serialize(const_cast<ANSICHAR*>(Utf8.Get()), Utf8.Length());
    }
    Student.bActive = false;
    Student.Wake = Time + Dwell;
    ++LegsCompleted;
    if (++Student.Leg == LegCount && ++Completed == Students.Num())
        UE_LOG(LogTemp, Display, TEXT("Campus turns finished: %d schedules, %d legs, %lld learned turns, %lld retraced edges, %lld inference retries"),
            Completed, LegsCompleted, Stats.Applied, Stats.RetracedEdges, Stats.Retries);
}

void FCampusTurnDay::Step(float DeltaSeconds, TArray<FTransform>& Transforms)
{
    Time += DeltaSeconds;
    const double Now = FPlatformTime::Seconds();
    Active = 0;
    for (int32 I = 0; I < Students.Num(); ++I)
    {
        FStudent& Student = Students[I];
        if (!Student.bActive && Student.Leg < LegCount && Time >= Student.Wake)
        {
            Student.bActive = true;
            Student.RetracedEdges = 0;
            Student.Trail.Reset();
            Student.Node = Stops[Student.Stops[Student.Leg]];
            Student.Position = Points[Student.Node];
            if (bShowReference && Student.Leg > 0) BuildReferencePath(Student, I);
            Student.Incoming = FVector2f::ZeroVector;
            Student.Visited.Reset();
            Student.Visited.Add(Student.Node);
            if (Trace)
            {
                Student.TraceStartTime = Time;
                Student.TracePath.Reset();
                Student.TracePath.Add(Student.Node);
                Student.TraceDecisions.Reset();
            }
            AtNode(Student, I, Now);
        }
        if (Student.bActive && Student.Edge != INDEX_NONE)
        {
            const FEdge& Edge = Edges[Student.Node][Student.Edge];
            Student.Progress = FMath::Min(Student.Progress + Student.Pace * DeltaSeconds, Edge.Distance.Last());
            while (Student.Segment + 2 < Edge.Points.Num() && Student.Progress >= Edge.Distance[Student.Segment + 1])
                ++Student.Segment;
            const int32 A = Student.Segment, B = A + 1;
            const float Alpha = (Student.Progress - Edge.Distance[A]) / (Edge.Distance[B] - Edge.Distance[A]);
            Student.Position = FMath::Lerp(Edge.Points[A], Edge.Points[B], FMath::Clamp(Alpha, 0.f, 1.f));
            Student.Facing = (Edge.Points[B] - Edge.Points[A]).GetSafeNormal();
            if (Student.Progress >= Edge.Distance.Last())
            {
                Student.Node = Edge.To;
                Student.Position = Points[Student.Node];
                Student.Incoming = Student.Facing;
                Student.Visited.AddUnique(Student.Node);
                if (Student.Node == Stops[Student.Stops[Student.Leg + 1]]) FinishLeg(Student, I);
                else AtNode(Student, I, Now);
            }
        }
        Active += Student.bActive;
        Transforms[I] = Student.bActive
            ? FTransform(FRotator(0, FMath::RadiansToDegrees(FMath::Atan2(Student.Facing.Y, Student.Facing.X)), 0),
                FVector(100 * Student.Position.X, 100 * Student.Position.Y, 94), FVector(.6f, .6f, 1.8f))
            : FTransform(FRotator::ZeroRotator, FVector(0, 0, -1000), FVector::ZeroVector);
    }
}

int32 FCampusTurnDay::GetTargetStop(int32 Index) const
{
    const auto& Student = Students[Index];
    return Student.Stops[FMath::Min(LegCount, Student.Leg + int32(Student.bActive || Student.Leg == 0))];
}

const TCHAR* FCampusTurnDay::GetDecision(int32 Index) const
{
    const auto& Student = Students[Index];
    if (Student.Leg == LegCount) return TEXT("Day complete");
    if (!Student.bActive) return Student.Leg ? TEXT("At class") : TEXT("At entrance");
    return Student.PendingIndex != INDEX_NONE ? TEXT("Choosing turn") : Student.Decision;
}

FLinearColor FCampusTurnDay::DestinationColor(int32 Key)
{
    const FColor Gates[] = {FColor(121, 222, 219), FColor(245, 199, 100), FColor(190, 164, 246), FColor(119, 184, 248)};
    if (Key < 0 && -Key <= UE_ARRAY_COUNT(Gates)) return FLinearColor::FromSRGBColor(Gates[-Key - 1]);
    return FLinearColor::MakeFromHSV8(uint8(Key * 137), 130 + (Key % 3) * 33, 245 - (Key % 2) * 35);
}

TConstArrayView<FVector2f> FCampusTurnDay::GetReferencePath(int32 Index) const
{
    return ReferencePaths.IsValidIndex(Index) ? TConstArrayView<FVector2f>(ReferencePaths[Index]) : TConstArrayView<FVector2f>();
}

void FCampusTurnDay::BuildReferencePath(const FStudent& Student, int32 Index)
{
    const int32 Start = Stops[Student.Stops[Student.Leg]], Goal = Stops[Student.Stops[Student.Leg + 1]];
    struct FVisit { float Cost; int32 Node; };
    const auto Earlier = [](const FVisit& A, const FVisit& B) { return A.Cost < B.Cost; };
    TArray<FVisit> Queue;
    TArray<float> Best;
    TArray<int32> Parent, ParentEdge;
    Best.Init(MAX_flt, Points.Num());
    Parent.Init(INDEX_NONE, Points.Num());
    ParentEdge.SetNumUninitialized(Points.Num());
    Best[Start] = 0;
    Queue.HeapPush({0, Start}, Earlier);
    while (!Queue.IsEmpty())
    {
        FVisit Top;
        Queue.HeapPop(Top, Earlier, EAllowShrinking::No);
        if (Top.Cost != Best[Top.Node]) continue;
        if (Top.Node == Goal) break;
        const auto& Links = Edges[Top.Node];
        for (int32 I = 0; I < Links.Num(); ++I)
        {
            const auto& Link = Links[I];
            const float Cost = Top.Cost + Link.Distance.Last();
            if (Cost >= Best[Link.To]) continue;
            Best[Link.To] = Cost;
            Parent[Link.To] = Top.Node;
            ParentEdge[Link.To] = I;
            Queue.HeapPush({Cost, Link.To}, Earlier);
        }
    }
    // This path goes only to the map overlay. Turn inference never reads it.
    TArray<const FEdge*> Links;
    for (int32 Node = Goal; Parent[Node] != INDEX_NONE; Node = Parent[Node])
        Links.Add(&Edges[Parent[Node]][ParentEdge[Node]]);
    Algo::Reverse(Links);
    auto& Path = ReferencePaths[Index];
    Path.Reset();
    Path.Add(Points[Start]);
    for (const FEdge* Link : Links)
        Path.Append(MakeArrayView(Link->Points).RightChop(1));
}
