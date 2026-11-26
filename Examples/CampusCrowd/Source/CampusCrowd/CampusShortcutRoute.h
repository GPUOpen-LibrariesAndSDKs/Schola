#pragma once

#include "CampusNavigation.h"

namespace CampusNav
{
constexpr float ShortcutStep = .1f;
constexpr float JoinRadius = Pace * ShortcutStep;
constexpr float TurnTime = Pace / Acceleration;

inline bool SegmentClear(const Grid& Map, Vec Start, Vec End)
{
    const auto Basis = GoalFrame(Start, End);
    return !Map.BlockedAt(Start) && !Map.BlockedAt(End) &&
        Map.Clearance(Start, Basis.Forward, Basis.Distance + Skin) >= Basis.Distance;
}

struct ShortcutCursor
{
    int Next = 1;
    int Target = 1;
    float ReplanIn = 0;
};

inline float TravelTime(const Frame& Chord, Vec Velocity)
{
    return Chord.Distance / Pace + (Pace - Chord.Forward.Dot(Velocity)) / (2 * Acceleration);
}

// The shortest visible suffix is shared by every student with this stop pair.
inline void ShortcutTimes(const Grid& Map, const Vec* Points, int Count, float* Seconds)
{
    Seconds[Count - 1] = 0;
    for (int From = Count - 2; From >= 0; --From)
    {
        Seconds[From] = std::numeric_limits<float>::infinity();
        for (int To = Count - 1; To > From; --To)
        {
            const float Cost = (Points[To] - Points[From]).Length() / Pace + Seconds[To];
            if (Cost < Seconds[From] && SegmentClear(Map, Points[From], Points[To]))
                Seconds[From] = Cost;
        }
    }
}

inline Vec ShortcutVelocity(const State& Walker, Vec Target, bool IsDestination)
{
    const auto Chord = GoalFrame(Walker.Position, Target);
    // Brake within stopping distance of a join so the next turn starts on the route.
    const float Speed = IsDestination ? Pace : std::min(Pace, std::sqrt(2 * Acceleration * Chord.Distance));
    return Chord.Forward * Speed;
}

inline bool CanStartShortcut(const Grid& Map, State Walker, Vec Target, bool IsDestination)
{
    for (float Remaining = 2 * TurnTime; Remaining > 0; Remaining -= ShortcutStep)
    {
        const auto Chord = GoalFrame(Walker.Position, Target);
        if (Chord.Distance <= (IsDestination ? GoalRadius : JoinRadius))
            return SegmentClear(Map, Walker.Position, Target);
        const auto Desired = ShortcutVelocity(Walker, Target, IsDestination);
        if ((Desired - Walker.Velocity).Length() <= Skin)
            return SegmentClear(Map, Walker.Position, Target);
        if (Move(Map, Walker, Desired, std::min(ShortcutStep, Remaining))) return false;
    }
    return SegmentClear(Map, Walker.Position, Target);
}

inline bool WalkShortcut(const Grid& Map, State& Walker, const Vec* Points, const float* Seconds,
                         int Count, ShortcutCursor& Cursor, float Dt)
{
    const int Last = Count - 1;
    const bool Joined = Cursor.Target < Last &&
        (Walker.Position - Points[Cursor.Target]).Length() <= JoinRadius &&
        SegmentClear(Map, Walker.Position, Points[Cursor.Target]);
    if (Joined) Cursor.Next = Cursor.Target + 1;

    // Update the route choice at the training cadence. Movement still sweeps
    // the grid every frame, and joins and blocked steps can trigger an early search.
    if (Joined || Cursor.ReplanIn <= 0.f)
    {
        float Best = std::numeric_limits<float>::infinity();
        int Target = Cursor.Next;
        for (int I = Last; I >= Cursor.Next; --I)
        {
            const auto Chord = GoalFrame(Walker.Position, Points[I]);
            const float Cost = TravelTime(Chord, Walker.Velocity) + Seconds[I];
            if (Cost + Skin >= Best || !SegmentClear(Map, Walker.Position, Points[I]) ||
                !CanStartShortcut(Map, Walker, Points[I], I == Last))
                continue;
            Best = Cost;
            Target = I;
        }
        Cursor.Target = Target;
        if (Cursor.ReplanIn <= 0.f) Cursor.ReplanIn += ShortcutStep;
    }
    const bool Blocked = Move(Map, Walker, ShortcutVelocity(Walker, Points[Cursor.Target], Cursor.Target == Last), Dt);
    Cursor.ReplanIn = Blocked ? 0.f : Cursor.ReplanIn - Dt;
    return Blocked;
}
}
