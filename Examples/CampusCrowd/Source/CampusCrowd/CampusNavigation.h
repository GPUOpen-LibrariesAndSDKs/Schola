#pragma once

// Shared by the headless trainer and Unreal. All lengths are in metres.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace CampusNav
{
constexpr float Pace = 1.4f, Acceleration = 1.85f;
constexpr float GoalRadius = .65f;
constexpr float Skin = .001f;

struct Vec
{
    float X = 0, Y = 0;
    Vec operator+(Vec B) const { return {X + B.X, Y + B.Y}; }
    Vec operator-(Vec B) const { return {X - B.X, Y - B.Y}; }
    Vec operator*(float S) const { return {X * S, Y * S}; }
    float Dot(Vec B) const { return X * B.X + Y * B.Y; }
    float Length() const { return std::sqrt(Dot(*this)); }
    Vec Limited(float Limit) const
    {
        const float L = Length();
        return L > Limit ? *this * (Limit / L) : *this;
    }
};

struct Grid
{
    const std::uint8_t* Bits = nullptr;
    int Width = 0, Height = 0;
    float Cell = .5f;
    Vec Origin;

    bool Blocked(int X, int Y) const
    {
        if (X < 0 || X >= Width || Y < 0 || Y >= Height) return true;
        const auto Index = std::size_t(Y) * Width + X;
        return (Bits[Index / 8] >> (Index % 8)) & 1u;
    }

    bool BlockedAt(Vec Position) const
    {
        const Vec Location = (Position - Origin) * (1.f / Cell);
        return Blocked(static_cast<int>(std::floor(Location.X)), static_cast<int>(std::floor(Location.Y)));
    }

    // Traverse every touched cell, including both side cells at a diagonal corner.
    float Clearance(Vec Source, Vec Direction, float Limit) const
    {
        const Vec Location = (Source - Origin) * (1.f / Cell);
        int X = static_cast<int>(std::floor(Location.X)), Y = static_cast<int>(std::floor(Location.Y));
        if (Blocked(X, Y)) return 0;
        constexpr auto Infinity = std::numeric_limits<float>::infinity();
        const int StepX = Direction.X >= 0 ? 1 : -1, StepY = Direction.Y >= 0 ? 1 : -1;
        const float DX = Direction.X == 0 ? Infinity : Cell / std::abs(Direction.X);
        const float DY = Direction.Y == 0 ? Infinity : Cell / std::abs(Direction.Y);
        float NextX = std::isinf(DX) ? DX : DX * (StepX > 0 ? X + 1 - Location.X : Location.X - X);
        float NextY = std::isinf(DY) ? DY : DY * (StepY > 0 ? Y + 1 - Location.Y : Location.Y - Y);
        for (;;)
        {
            const float Next = std::min(NextX, NextY);
            if (Next > Limit) return Limit;
            if (std::abs(NextX - NextY) <= 1e-6f)
            {
                if (Blocked(X + StepX, Y) || Blocked(X, Y + StepY)) return Next;
                X += StepX;
                Y += StepY;
                NextX += DX;
                NextY += DY;
            }
            else if (NextX < NextY)
            {
                X += StepX;
                NextX += DX;
            }
            else
            {
                Y += StepY;
                NextY += DY;
            }
            if (Blocked(X, Y)) return Next;
        }
    }
};

struct State
{
    Vec Position, Velocity;
};

struct Frame
{
    Vec Forward, Right;
    float Distance;
};

inline Frame GoalFrame(Vec Position, Vec Goal)
{
    const Vec Delta = Goal - Position;
    const float Distance = Delta.Length();
    const Vec Forward = Distance > 1e-6f ? Delta * (1.f / Distance) : Vec{1, 0};
    return {Forward, {-Forward.Y, Forward.X}, Distance};
}

inline Vec DesiredVelocity(const State& Student, Vec Goal, Vec Command)
{
    const auto Basis = GoalFrame(Student.Position, Goal);
    const Vec Action = Command.Limited(1.f);
    return (Basis.Forward * Action.X + Basis.Right * Action.Y) * Pace;
}

inline bool Move(const Grid& Map, State& Student, Vec Desired, float Dt)
{
    Student.Velocity = Student.Velocity + (Desired - Student.Velocity).Limited(Acceleration * Dt);
    const float Speed = Student.Velocity.Length();
    if (Speed < 1e-6f) return false;
    const Vec Direction = Student.Velocity * (1.f / Speed);
    const float Length = Speed * Dt;
    const float Distance = Map.Clearance(Student.Position, Direction, Length + Skin);
    const Vec Target = Student.Position + Direction * Length;
    if (Distance < Length + Skin || Map.BlockedAt(Target))
    {
        const Vec Safe = Student.Position + Direction * std::min(Length, std::max(0.f, Distance - Skin));
        if (!Map.BlockedAt(Safe)) Student.Position = Safe;

        // At a wall, spend the remaining movement along the open axis. Sweep
        // the slide as well so a corner cannot push the walker into a solid cell.
        const Vec Remainder = Target - Student.Position;
        const auto Slide = [&](Vec Axis, float Amount)
        {
            const float Limit = std::abs(Amount);
            if (Limit < Skin) return Vec{};
            const Vec Along = Axis * (Amount < 0 ? -1.f : 1.f);
            const float Available = Map.Clearance(Student.Position, Along, Limit + Skin);
            const Vec Offset = Along * std::min(Limit, std::max(0.f, Available - Skin));
            return Map.BlockedAt(Student.Position + Offset) ? Vec{} : Offset;
        };
        const Vec X = Slide({1, 0}, Remainder.X), Y = Slide({0, 1}, Remainder.Y);
        const Vec Offset = X.Dot(X) >= Y.Dot(Y) ? X : Y;
        Student.Position = Student.Position + Offset;
        Student.Velocity = Offset * (1.f / Dt);
        return true;
    }
    Student.Position = Target;
    return false;
}

inline bool Arrived(const Grid& Map, const State& Student, Vec Goal)
{
    const auto Basis = GoalFrame(Student.Position, Goal);
    return Basis.Distance <= GoalRadius &&
        Map.Clearance(Student.Position, Basis.Forward, Basis.Distance) >= Basis.Distance;
}
}
