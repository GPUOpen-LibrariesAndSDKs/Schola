#include "../Source/CampusCrowd/CampusShortcutRoute.h"
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

static_assert(sizeof(CampusNav::Vec) == 2 * sizeof(float));

extern "C"
{
void nav_step(const std::uint8_t* Cells, int Width, int Height, float Cell, float X, float Y,
              float Dt, int Count, float* States, const float* Goals, const float* Actions,
              std::uint8_t* Results)
{
    const CampusNav::Grid Map{Cells, Width, Height, Cell, {X, Y}};
    for (int I = 0; I < Count; ++I)
    {
        auto* S = States + 4 * I;
        CampusNav::State Student{{S[0], S[1]}, {S[2], S[3]}};
        const CampusNav::Vec Goal{Goals[2*I], Goals[2*I+1]};
        const auto Desired = CampusNav::DesiredVelocity(Student, Goal, {Actions[2*I], Actions[2*I+1]});
        const bool Blocked = CampusNav::Move(Map, Student, Desired, Dt);
        const bool Arrived = CampusNav::Arrived(Map, Student, Goal);
        Results[I] = unsigned(Blocked) | (unsigned(Arrived) << 1);
        S[0] = Student.Position.X;
        S[1] = Student.Position.Y;
        S[2] = Student.Velocity.X;
        S[3] = Student.Velocity.Y;
    }
}

void nav_clear_segments(const std::uint8_t* Cells, int Width, int Height, float Cell, float X, float Y,
                        int Count, const float* Starts, const float* Ends, std::uint8_t* Results)
{
    const CampusNav::Grid Map{Cells, Width, Height, Cell, {X, Y}};
    for (int I = 0; I < Count; ++I)
    {
        const CampusNav::Vec Start{Starts[2*I], Starts[2*I+1]}, End{Ends[2*I], Ends[2*I+1]};
        Results[I] = CampusNav::SegmentClear(Map, Start, End);
    }
}

void nav_shortcut_times(const std::uint8_t* Cells, int Width, int Height, float Cell, float X, float Y,
                        int Count, const CampusNav::Vec* Points, const int* Offsets, float* Seconds)
{
    const CampusNav::Grid Map{Cells, Width, Height, Cell, {X, Y}};
    for (int I = 0; I < Count; ++I)
    {
        const int Offset = Offsets[I];
        CampusNav::ShortcutTimes(Map, Points + Offset, Offsets[I + 1] - Offset, Seconds + Offset);
    }
}

void nav_shortcut_step(const std::uint8_t* Cells, int Width, int Height, float Cell, float X, float Y,
                       float Dt, int Count, float* States, const CampusNav::Vec* Points,
                       const float* Seconds, const int* Offsets, const int* RouteIds, int* Cursors,
                       float* ReplanIn, std::uint8_t* Results)
{
    const CampusNav::Grid Map{Cells, Width, Height, Cell, {X, Y}};
    for (int I = 0; I < Count; ++I)
    {
        const int Route = RouteIds[I];
        const int Offset = Offsets[Route];
        const int Size = Offsets[Route + 1] - Offset;
        auto* S = States + 4 * I;
        CampusNav::State Walker{{S[0], S[1]}, {S[2], S[3]}};
        CampusNav::ShortcutCursor Cursor{Cursors[2 * I], Cursors[2 * I + 1], ReplanIn[I]};
        const bool Blocked = CampusNav::WalkShortcut(Map, Walker, Points + Offset, Seconds + Offset, Size, Cursor, Dt);
        const bool Arrived = CampusNav::Arrived(Map, Walker, Points[Offset + Size - 1]);
        Results[I] = unsigned(Blocked) | (unsigned(Arrived) << 1);
        Cursors[2 * I] = Cursor.Next;
        Cursors[2 * I + 1] = Cursor.Target;
        ReplanIn[I] = Cursor.ReplanIn;
        S[0] = Walker.Position.X;
        S[1] = Walker.Position.Y;
        S[2] = Walker.Velocity.X;
        S[3] = Walker.Velocity.Y;
    }
}

void nav_distance(const std::uint8_t* Cells, int Width, int Height, float Cell, float X, float Y,
                  int GoalX, int GoalY, float* Distances)
{
    const CampusNav::Grid Map{Cells, Width, Height, Cell, {X, Y}};
    const float Infinity = std::numeric_limits<float>::infinity();
    std::fill_n(Distances, std::size_t(Width) * Height, Infinity);
    if (Map.Blocked(GoalX, GoalY)) return;
    using Entry = std::pair<float, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> Queue;
    const int Start = GoalY * Width + GoalX;
    Distances[Start] = 0.f;
    Queue.emplace(0.f, Start);
    const float Diagonal = std::sqrt(2.f) * Cell;
    while (!Queue.empty())
    {
        const auto [Cost, Index] = Queue.top();
        Queue.pop();
        if (Cost != Distances[Index]) continue;
        const int CX = Index % Width, CY = Index / Width;
        for (int DY = -1; DY <= 1; ++DY)
            for (int DX = -1; DX <= 1; ++DX)
            {
                if ((!DX && !DY) || Map.Blocked(CX + DX, CY + DY)) continue;
                if (DX && DY && (Map.Blocked(CX + DX, CY) || Map.Blocked(CX, CY + DY))) continue;
                const int Next = (CY + DY) * Width + CX + DX;
                const float Candidate = Cost + (DX && DY ? Diagonal : Cell);
                if (Candidate >= Distances[Next]) continue;
                Distances[Next] = Candidate;
                Queue.emplace(Candidate, Next);
            }
    }
}
}
