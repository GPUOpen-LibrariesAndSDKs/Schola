#include "CampusMigrationDay.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"

namespace
{
constexpr float CorridorHalfWidth = 115.f;
constexpr float NeighborRange = 450.f;
constexpr float VelocityScale = 220.f;
constexpr float EntranceGap = 70.f;
constexpr float CloseDistance = 45.f;

FIntPoint CellFor(FVector2f Point)
{
    return FIntPoint(FMath::FloorToInt32(Point.X / NeighborRange), FMath::FloorToInt32(Point.Y / NeighborRange));
}

FVector2f RightOf(FVector2f Direction)
{
    return FVector2f(-Direction.Y, Direction.X);
}

FVector2f Toward(FVector2f Current, FVector2f Target, float MaxChange)
{
    return Current + (Target - Current).GetClampedToMaxSize(MaxChange);
}
}

bool FCampusMigrationDay::LoadRoutes(FString& Error)
{
    FString Text;
    TSharedPtr<FJsonObject> Root;
    const FString Path = FPaths::ProjectContentDir() / TEXT("Campus/migration_routes.json");
    Error = TEXT("Missing or invalid Campus/migration_routes.json");
    if (!FFileHelper::LoadFileToString(Text, *Path) ||
        !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root) || !Root)
        return false;

    const TArray<TSharedPtr<FJsonValue>> *NodeRows, *StopRows, *PathRows, *CorridorRows;
    if (!Root->TryGetNumberField(TEXT("gate_count"), GateCount) || GateCount != 4 ||
        !Root->TryGetArrayField(TEXT("nodes"), NodeRows) ||
        !Root->TryGetArrayField(TEXT("stops"), StopRows) || StopRows->Num() < GateCount + 2 ||
        !Root->TryGetArrayField(TEXT("paths"), PathRows) || PathRows->Num() != FMath::Square(StopRows->Num()) ||
        !Root->TryGetArrayField(TEXT("corridors"), CorridorRows))
        return false;

    TArray<FVector2f> Points;
    for (const auto& Value : *NodeRows)
    {
        const TArray<TSharedPtr<FJsonValue>>* Pair;
        double X, Y;
        if (!Value->TryGetArray(Pair) || Pair->Num() != 2 ||
            !(*Pair)[0]->TryGetNumber(X) || !(*Pair)[1]->TryGetNumber(Y) ||
            !FMath::IsFinite(X) || !FMath::IsFinite(Y) || FMath::Abs(X) > 100000 || FMath::Abs(Y) > 100000)
            return false;
        Points.Emplace(X, Y);
    }
    TMap<FIntPoint, float> Widths;
    for (const auto& Value : *CorridorRows)
    {
        const TArray<TSharedPtr<FJsonValue>>* Row;
        double A, B, Width;
        if (!Value->TryGetArray(Row) || Row->Num() != 3 ||
            !(*Row)[0]->TryGetNumber(A) || !(*Row)[1]->TryGetNumber(B) || !(*Row)[2]->TryGetNumber(Width) ||
            A < 0 || A >= B || B >= Points.Num() || FMath::FloorToDouble(A) != A || FMath::FloorToDouble(B) != B ||
            !FMath::IsFinite(Width) || Width < 0 || Width > CorridorHalfWidth)
            return false;
        Widths.Add(FIntPoint(int32(A), int32(B)), float(Width));
    }
    for (const auto& Value : *StopRows)
    {
        const TSharedPtr<FJsonObject>* Object;
        FStop Stop;
        if (!Value->TryGetObject(Object) || !(*Object)->TryGetStringField(TEXT("name"), Stop.Name) ||
            !(*Object)->TryGetNumberField(TEXT("node"), Stop.Node) || !Points.IsValidIndex(Stop.Node))
            return false;
        Stops.Add(MoveTemp(Stop));
    }
    for (int32 I = 0; I < PathRows->Num(); ++I)
    {
        const int32 Start = I / Stops.Num(), End = I % Stops.Num();
        const TArray<TSharedPtr<FJsonValue>>* Row;
        if (!(*PathRows)[I]->TryGetArray(Row) || (Start == End ? !Row->IsEmpty() : Row->Num() < 2))
            return false;

        FRoute& Route = Routes.AddDefaulted_GetRef();
        int32 Previous = INDEX_NONE;
        for (const auto& Value : *Row)
        {
            double Number;
            if (!Value->TryGetNumber(Number) || Number < 0 || Number >= Points.Num() ||
                FMath::FloorToDouble(Number) != Number)
                return false;
            const int32 Node = int32(Number);
            const FVector2f Point = Points[Node];
            const float Length = Previous == INDEX_NONE ? 0 : FVector2f::Distance(Route.Points.Last(), Point);
            if (Previous != INDEX_NONE)
            {
                const float* Width = Widths.Find(FIntPoint(FMath::Min(Previous, Node), FMath::Max(Previous, Node)));
                if (Length < 1.f || !Width) return false;
                Route.Widths.Add(*Width);
            }
            Previous = Node;
            Route.Distances.Add(Route.Distances.IsEmpty() ? 0.f : Route.Distances.Last() + Length);
            Route.Points.Add(Point);
        }
        if (Start != End && (Route.Points[0] != Points[Stops[Start].Node] || Route.Points.Last() != Points[Stops[End].Node]))
            return false;
    }
    Error.Reset();
    return true;
}

bool FCampusMigrationDay::Initialize(int32 StudentCount, bool bNavigation, bool bTurns, const TCHAR* Command, FString& Error, bool bBackground, bool bFourDemo)
{
    if (bTurns)
    {
        Turns = MakeUnique<FCampusTurnDay>();
        return Turns->Initialize(StudentCount, Command, Error, bFourDemo);
    }
    if (bNavigation)
    {
        Navigation = MakeUnique<FCampusNavigationDay>();
        return Navigation->Initialize(StudentCount, Command, bBackground, Error);
    }
    if (!LoadRoutes(Error)) return false;
    FString Policy = TEXT("Learned");
    FParse::Value(Command, TEXT("MigrationPolicy="), Policy);
    if (!Policy.Equals(TEXT("Rules"), ESearchCase::IgnoreCase) && !Policy.Equals(TEXT("Learned"), ESearchCase::IgnoreCase))
    {
        Error = TEXT("Use -MigrationPolicy=Rules or -MigrationPolicy=Learned");
        return false;
    }
    bLearned = Policy.Equals(TEXT("Learned"), ESearchCase::IgnoreCase);
    FParse::Value(Command, TEXT("MigrationSeed="), Seed);
    FParse::Value(Command, TEXT("MigrationYield="), Yield);
    FParse::Value(Command, TEXT("MigrationLane="), LaneWidth);
    FParse::Value(Command, TEXT("MigrationRelease="), ReleaseSeconds);
    FParse::Value(Command, TEXT("MigrationDwell="), DwellSeconds);
    Yield = FMath::Clamp(Yield, 0.f, 1.5f);
    LaneWidth = FMath::Clamp(LaneWidth, 0.f, CorridorHalfWidth);
    ReleaseSeconds = FMath::Clamp(ReleaseSeconds, 0.f, 600.f);
    DwellSeconds = FMath::Clamp(DwellSeconds, 0.f, 600.f);

    FRandomStream Random(Seed);
    Students.SetNum(StudentCount);
    for (auto& Student : Students)
    {
        Student.Stops[0] = Random.RandHelper(GateCount);
        Student.Stops[CampusMigration::LegCount] = Random.RandHelper(GateCount);
        for (int32 I = 1; I < CampusMigration::LegCount; ++I)
        {
            // Uniformly draw from the other buildings on a class-to-class leg.
            const int32 Previous = Student.Stops[I - 1];
            const bool bPreviousBuilding = Previous >= GateCount;
            int32 Destination = GateCount + Random.RandHelper(Stops.Num() - GateCount - int32(bPreviousBuilding));
            if (bPreviousBuilding && Destination >= Previous) ++Destination;
            Student.Stops[I] = Destination;
        }
        Student.Pace = Random.FRandRange(115.f, 180.f);
        Student.Lane = Random.FRandRange(.55f, 1.f) * LaneWidth / CorridorHalfWidth;
        Student.ReleaseDelay = Random.FRandRange(0.f, ReleaseSeconds);
    }
    ReleasedAtStop.Init(false, Stops.Num());

    FString TracePath;
    if (FParse::Value(Command, TEXT("MigrationTrace="), TracePath))
    {
        Trace.Reset(IFileManager::Get().CreateFileWriter(*TracePath));
        if (!Trace)
        {
            Error = TEXT("Could not open migration trace ") + TracePath;
            return false;
        }
        auto Header = MakeShared<FJsonObject>();
        Header->SetStringField(TEXT("policy"), bLearned ? TEXT("Learned") : TEXT("Rules"));
        Header->SetNumberField(TEXT("yield"), Yield);
        Header->SetNumberField(TEXT("lane_cm"), LaneWidth);
        Header->SetNumberField(TEXT("release_seconds"), ReleaseSeconds);
        Header->SetNumberField(TEXT("students"), Students.Num());
        Header->SetNumberField(TEXT("seed"), Seed);
        Header->SetNumberField(TEXT("gates"), GateCount);
        TArray<TSharedPtr<FJsonValue>> Locations;
        for (int32 I = 0; I < Stops.Num(); ++I)
        {
            const auto& Position = Routes[I * Stops.Num() + (I + 1) % Stops.Num()].Points[0];
            auto Location = MakeShared<FJsonObject>();
            Location->SetStringField(TEXT("name"), Stops[I].Name);
            Location->SetNumberField(TEXT("x"), Position.X);
            Location->SetNumberField(TEXT("y"), Position.Y);
            Locations.Add(MakeShared<FJsonValueObject>(Location));
        }
        Header->SetArrayField(TEXT("stops"), Locations);
        FString Line;
        FJsonSerializer::Serialize(Header, TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Line));
        WriteTrace(Line + LINE_TERMINATOR);
    }
    StartLeg();
    return true;
}

void FCampusMigrationDay::StartLeg()
{
    WaveStart = Time;
    NextWave = -1;
    Arrived = 0;
    for (auto& Student : Students)
    {
        Student.bArrived = false;
        Student.ReleaseDelay = FMath::Fmod(Student.ReleaseDelay + ReleaseSeconds * .381966f * int32(Leg > 0),
            FMath::Max(.01f, ReleaseSeconds));
    }
    UE_LOG(LogTemp, Display, TEXT("Campus migration %d/%d started"), Leg + 1, CampusMigration::LegCount);
}

const FCampusMigrationDay::FRoute& FCampusMigrationDay::RouteFor(const FCampusMigrationStudent& Student) const
{
    return Routes[Student.Stops[Leg] * Stops.Num() + Student.Stops[Leg + 1]];
}

void FCampusMigrationDay::StartStudent(FCampusMigrationStudent& Student)
{
    const auto& Route = RouteFor(Student);
    Student.bWalking = true;
    Student.Segment = 0;
    Student.Progress = Student.Offset = 0;
    Student.Position = Route.Points[0];
    Student.Velocity = Student.LocalVelocity = FVector2f::ZeroVector;
    Student.Facing = (Route.Points[1] - Route.Points[0]).GetSafeNormal();
    Student.Action = FVector2f(-1, 0); // Use Rules until the first model decision.
    FMemory::Memzero(Student.Observation);
    ++Active;
}

void FCampusMigrationDay::BuildGrid()
{
    Grid.Reset();
    for (int32 I = 0; I < Students.Num(); ++I)
        if (Students[I].bWalking) Grid.FindOrAdd(CellFor(Students[I].Position)).Add(I);
}

bool FCampusMigrationDay::EntranceClear(FVector2f Position, int32 Stop) const
{
    if (ReleasedAtStop[Stop]) return false;
    const auto Cell = CellFor(Position);
    for (int32 Y = -1; Y <= 1; ++Y)
        for (int32 X = -1; X <= 1; ++X)
            if (const auto* Bucket = Grid.Find(Cell + FIntPoint(X, Y)))
                for (const int32 Index : *Bucket)
                    if (FVector2f::DistSquared(Students[Index].Position, Position) < FMath::Square(EntranceGap))
                        return false;
    return true;
}

void FCampusMigrationDay::Observe(int32 Index)
{
    auto& Student = Students[Index];
    const auto& Route = RouteFor(Student);
    const int32 Segment = Student.Segment;
    const FVector2f Forward = (Route.Points[Segment + 1] - Route.Points[Segment]).GetSafeNormal();
    const FVector2f Right = RightOf(Forward);
    float* Row = Student.Observation;
    Row[0] = Student.Offset / CorridorHalfWidth;
    Row[1] = Student.LocalVelocity.X / Student.Pace;
    Row[2] = Student.LocalVelocity.Y / Student.Pace;
    Row[3] = FMath::Clamp((Route.Distances.Last() - Student.Progress) / 500.f, 0.f, 1.f);
    Row[4] = Route.Points.IsValidIndex(Segment + 2)
        ? FVector2f::DotProduct(Right, (Route.Points[Segment + 2] - Route.Points[Segment + 1]).GetSafeNormal()) : 0;
    Row[5] = 1.f - FMath::Clamp((Route.Distances[Segment + 1] - Student.Progress) / 400.f, 0.f, 1.f);
    Row[6] = FMath::Min(Student.Lane, CorridorWidth(Route, Segment, Student.Progress) / CorridorHalfWidth);

    float Distances[CampusMigration::NeighborCount];
    int32 Neighbors[CampusMigration::NeighborCount];
    for (int32 K = 0; K < CampusMigration::NeighborCount; ++K)
    {
        Distances[K] = FMath::Square(NeighborRange);
        Neighbors[K] = INDEX_NONE;
    }
    const auto Cell = CellFor(Student.Position);
    for (int32 Y = -1; Y <= 1; ++Y)
        for (int32 X = -1; X <= 1; ++X)
            if (const auto* Bucket = Grid.Find(Cell + FIntPoint(X, Y)))
                for (const int32 Other : *Bucket)
                {
                    if (Other == Index) continue;
                    const float Distance = FVector2f::DistSquared(Student.Position, Students[Other].Position);
                    for (int32 K = 0; K < CampusMigration::NeighborCount && Distance < Distances[CampusMigration::NeighborCount - 1]; ++K)
                    {
                        if (Distance >= Distances[K]) continue;
                        for (int32 J = CampusMigration::NeighborCount - 1; J > K; --J)
                        {
                            Distances[J] = Distances[J - 1];
                            Neighbors[J] = Neighbors[J - 1];
                        }
                        Distances[K] = Distance;
                        Neighbors[K] = Other;
                        break;
                    }
                }
    CloseStudents += Distances[0] < FMath::Square(CloseDistance);
    for (int32 K = 0; K < CampusMigration::NeighborCount; ++K)
    {
        float* Slot = Row + CampusMigration::NeighborStart + K * CampusMigration::NeighborStride;
        if (Neighbors[K] == INDEX_NONE)
        {
            FMemory::Memzero(Slot, CampusMigration::NeighborStride * sizeof(float));
            continue;
        }
        const auto& Other = Students[Neighbors[K]];
        const FVector2f Position = (Other.Position - Student.Position) / NeighborRange;
        const FVector2f Velocity = (Other.Velocity - Student.Velocity) / VelocityScale;
        Slot[0] = FVector2f::DotProduct(Position, Forward);
        Slot[1] = FVector2f::DotProduct(Position, Right);
        Slot[2] = FMath::Clamp(FVector2f::DotProduct(Velocity, Forward), -2.f, 2.f);
        Slot[3] = FMath::Clamp(FVector2f::DotProduct(Velocity, Right), -2.f, 2.f);
        Slot[4] = 1;
    }
}

FVector2f FCampusMigrationDay::Rules(const float* Row) const
{
    const float Turn = FMath::Abs(Row[4]) * Row[5];
    const float BasePace = FMath::Clamp(Row[3] * 2.5f, .30f, 1.f) * (1.f - .22f * Turn);
    float Slowdown = 0, Dodge = 0;
    for (int32 K = 0; K < CampusMigration::NeighborCount; ++K)
    {
        const float* Slot = Row + CampusMigration::NeighborStart + K * CampusMigration::NeighborStride;
        const float Front = Slot[0], Side = Slot[1], Closing = -Slot[2];
        const float Ahead = FMath::Clamp((.60f - Front) * 2.5f, 0.f, 1.f);
        const float Aligned = FMath::Clamp(1.f - FMath::Abs(Side) * 5.f, 0.f, 1.f);
        if (Slot[4] == 0 || Front <= 0) continue;
        Slowdown = FMath::Max(Slowdown, Ahead * Aligned * FMath::Clamp(.55f + Closing, 0.f, 1.f));
        Dodge += .32f * Ahead * Aligned * (Side <= .03f ? 1.f : -1.f);
    }
    return FVector2f(FMath::Max(.13f, BasePace * (1.f - .75f * Yield * Slowdown)),
        FMath::Clamp((Row[6] - Row[0]) * .8f + Yield * Dodge, -.65f, .65f));
}

float FCampusMigrationDay::CorridorWidth(const FRoute& Route, int32 Segment, float Progress) const
{
    const float Here = Route.Widths[Segment];
    const float Past = Segment == 0 ? 0.f : FMath::Min(Here, Route.Widths[Segment - 1]);
    const float Ahead = Segment + 1 == Route.Widths.Num() ? 0.f : FMath::Min(Here, Route.Widths[Segment + 1]);
    const float FromStart = FMath::Clamp((Progress - Route.Distances[Segment]) / 140.f, 0.f, 1.f);
    const float FromEnd = FMath::Clamp((Route.Distances[Segment + 1] - Progress) / 140.f, 0.f, 1.f);
    return FMath::Min(FMath::Lerp(Past, Here, FromStart), FMath::Lerp(Ahead, Here, FromEnd));
}

void FCampusMigrationDay::Integrate(FCampusMigrationStudent& Student, float Dt)
{
    const auto& Route = RouteFor(Student);
    const FVector2f Previous = Student.Position;
    const FVector2f Rule = Rules(Student.Observation);
    const FVector2f Desired = bLearned && Student.Action.X >= 0 ? Student.Action : Rule;
    // Keep a small forward speed in congested opposing flows where both sides yield.
    const FVector2f Speed(FMath::Clamp(Desired.X, .13f, 1.f), FMath::Clamp(Desired.Y, -.65f, .65f));
    Student.LocalVelocity = Toward(Student.LocalVelocity, Speed * Student.Pace, 185.f * Dt);
    Student.Progress = FMath::Min(Route.Distances.Last(), Student.Progress + Student.LocalVelocity.X * Dt);
    if (Student.Progress >= Route.Distances.Last())
    {
        Student.Position = Route.Points.Last();
        Student.Velocity = FVector2f::ZeroVector;
        Student.bWalking = false;
        Student.bArrived = true;
        --Active;
        ++Arrived;
        return;
    }
    while (Student.Progress >= Route.Distances[Student.Segment + 1]) ++Student.Segment;
    const int32 S = Student.Segment;
    const float Along = Student.Progress - Route.Distances[S];
    const float Left = Route.Distances[S + 1] - Student.Progress;
    const FVector2f Forward = (Route.Points[S + 1] - Route.Points[S]).GetSafeNormal();
    FVector2f Basis = Forward;
    if (S > 0 && Along < 100.f)
        Basis += (Route.Points[S] - Route.Points[S - 1]).GetSafeNormal() * (1.f - Along / 100.f);
    if (Route.Points.IsValidIndex(S + 2) && Left < 100.f)
        Basis += (Route.Points[S + 2] - Route.Points[S + 1]).GetSafeNormal() * (1.f - Left / 100.f);
    const float Width = CorridorWidth(Route, S, Student.Progress);
    Student.Offset = FMath::Clamp(Student.Offset + Student.LocalVelocity.Y * Dt, -Width, Width);
    Student.Position = Route.Points[S] + Forward * Along + RightOf(Basis.GetSafeNormal()) * Student.Offset;
    Student.Velocity = (Student.Position - Previous) / Dt;
    if (Student.Velocity.SizeSquared() > 36.f)
        Student.Facing = FMath::Lerp(Student.Facing, Student.Velocity.GetSafeNormal(), FMath::Min(1.f, Dt * 6.f)).GetSafeNormal();
}

void FCampusMigrationDay::Step(float DeltaSeconds, TArray<FTransform>& Transforms)
{
    if (Turns) { Turns->Step(DeltaSeconds, Transforms); return; }
    if (IsFinished() || DeltaSeconds <= 0) return;
    Time += DeltaSeconds;
    if (Navigation)
    {
        Navigation->Step(DeltaSeconds, Transforms);
        return;
    }
    if (NextWave >= 0 && Time >= NextWave)
    {
        ++Leg;
        if (IsFinished())
        {
            RecordFrame();
            if (Trace) Trace->Flush();
            UE_LOG(LogTemp, Display, TEXT("Campus migration day complete: %d students, %.1f simulated seconds"), Students.Num(), Time);
            return;
        }
        StartLeg();
    }

    BuildGrid();
    for (int32 Stop = 0; Stop < Stops.Num(); ++Stop) ReleasedAtStop[Stop] = false;
    for (auto& Student : Students)
        if (!Student.bWalking && !Student.bArrived && Time - WaveStart >= Student.ReleaseDelay)
        {
            const int32 Stop = Student.Stops[Leg];
            if (EntranceClear(RouteFor(Student).Points[0], Stop))
            {
                StartStudent(Student);
                Grid.FindOrAdd(CellFor(Student.Position)).Add(int32(&Student - Students.GetData()));
                ReleasedAtStop[Stop] = true;
            }
        }

    CloseStudents = 0;
    for (int32 I = 0; I < Students.Num(); ++I)
        if (Students[I].bWalking) Observe(I);
    ClosePersonSeconds += CloseStudents * DeltaSeconds;
    WalkingPersonSeconds += Active * DeltaSeconds;

    for (int32 I = 0; I < Students.Num(); ++I)
    {
        auto& Student = Students[I];
        if (Student.bWalking) Integrate(Student, DeltaSeconds);
        Transforms[I] = Student.bWalking
            ? FTransform(FRotator(0, FMath::RadiansToDegrees(FMath::Atan2(Student.Facing.Y, Student.Facing.X)), 0),
                FVector(Student.Position.X, Student.Position.Y, 94), FVector(.6f, .6f, 1.8f))
            : FTransform(FRotator::ZeroRotator, FVector(0, 0, -1000), FVector::ZeroVector);
    }
    const bool bWaveComplete = Arrived == Students.Num() && NextWave < 0;
    if (bWaveComplete)
    {
        UE_LOG(LogTemp, Display, TEXT("Campus migration %d/%d complete at %.1fs (%d arrivals)"),
            Leg + 1, CampusMigration::LegCount, Time, Arrived);
        NextWave = Time + (Leg + 1 == CampusMigration::LegCount ? 0.f : DwellSeconds);
    }
    if (Trace && (Time >= NextFrame || bWaveComplete))
    {
        RecordFrame();
        NextFrame = Time + .5;
    }
}

void FCampusMigrationDay::WriteObservation(int32 Student, TArrayView<float> Output) const
{
    if (Turns) { Turns->WriteObservation(Student, Output); return; }
    check(!Navigation);
    check(Output.Num() == CampusMigration::ObservationSize);
    FMemory::Memcpy(Output.GetData(), Students[Student].Observation, Output.Num() * sizeof(float));
}

void FCampusMigrationDay::ApplyAction(int32 Student, TConstArrayView<float> Input)
{
    if (Turns) { Turns->ApplyAction(Student, Input); return; }
    check(!Navigation);
    if (!IsWalking(Student) || !FMath::IsFinite(Input[0]) || !FMath::IsFinite(Input[1])) return;
    ++ModelDecisions;
    const auto& State = Students[Student];
    const float BasePace = FMath::Clamp(State.Observation[3] * 2.5f, .30f, 1.f) *
        (1.f - .22f * FMath::Abs(State.Observation[4]) * State.Observation[5]);
    const FVector2f Baseline(BasePace, (State.Observation[6] - State.Observation[0]) * .8f);
    const FVector2f Desired = Baseline + Yield * (FVector2f(Input[0], Input[1]) - Baseline);
    Students[Student].Action = FVector2f(FMath::Clamp(Desired.X, 0.f, 1.f), FMath::Clamp(Desired.Y, -.65f, .65f));
}

void FCampusMigrationDay::WriteTrace(const FString& Line)
{
    FTCHARToUTF8 Utf8(*Line);
    Trace->Serialize(const_cast<ANSICHAR*>(Utf8.Get()), Utf8.Length());
}

void FCampusMigrationDay::RecordFrame()
{
    if (!Trace) return;
    FString Line = FString::Printf(TEXT("{\"t\":%.2f,\"wave\":%d,\"arrived\":%d,\"close_s\":%.2f,\"walk_s\":%.2f,\"decisions\":%lld,\"p\":["),
        Time, Leg + 1, Arrived, ClosePersonSeconds, WalkingPersonSeconds, ModelDecisions);
    bool bFirst = true;
    for (int32 I = 0; I < Students.Num(); ++I)
    {
        const auto& Student = Students[I];
        if (!Student.bWalking) continue;
        if (!bFirst) Line.AppendChar(',');
        bFirst = false;
        Line.Appendf(TEXT("[%d,%d,%d,%d]"), I, FMath::RoundToInt(Student.Position.X),
            FMath::RoundToInt(Student.Position.Y), Student.Stops[Leg + 1]);
    }
    Line += TEXT("]}\n");
    WriteTrace(Line);
}
