// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// Engine-free scaling benchmark for the plugin's learner (Schola::RLTools::FLearner) at 1-200 NPCs.
//
// Answers three questions for a game that ticks its NPCs at a fixed decision rate:
//   1. Game-thread cost: how long Decide + Submit take per NPC decision (median and p99).
//   2. Shared learner: one learner fed by every NPC, trained serially on one worker thread.
//      Does it keep up with N NPCs x rate x GradientSteps updates per second, or does a backlog grow?
//   3. Independent learners: one learner per NPC. Resident memory per learner, and aggregate
//      update throughput when the learners share a fixed pool of worker threads.
// Observations and rewards are synthetic (random); the cost of a decision and an update does not
// depend on the values. Each learner is pre-filled to LearningStarts outside the timed window so
// every measurement is of the training phase.
//
// Usage: ScalingBenchmark [--npcs 1,10,25,50,100,200] [--rate 10] [--seconds 10] [--workers N]
//                         [--gradient-steps 1] [--csv path]
#include "RLToolsLearner.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

using namespace Schola::RLTools;
using Clock = std::chrono::steady_clock;

namespace
{
const char* const Profile = "sac/scale-16x4";
constexpr unsigned O = 16, A = 4, EpisodeLength = 100;

double ResidentMB()
{
#if defined(__APPLE__)
    mach_task_basic_info Info;
    mach_msg_type_number_t Count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&Info), &Count) != KERN_SUCCESS) return 0;
    return Info.resident_size / 1048576.0;
#elif defined(__linux__)
    long Pages = 0, Resident = 0;
    FILE* F = std::fopen("/proc/self/statm", "r");
    if (!F) return 0;
    const int Read = std::fscanf(F, "%ld %ld", &Pages, &Resident);
    std::fclose(F);
    return Read == 2 ? Resident * double(sysconf(_SC_PAGESIZE)) / 1048576.0 : 0;
#else
    return 0;
#endif
}

struct FOptions
{
    std::vector<unsigned> Npcs{1, 10, 25, 50, 100, 200};
    double Rate = 10, Seconds = 10;
    unsigned Workers = std::max(1u, std::thread::hardware_concurrency() > 1 ? std::thread::hardware_concurrency() - 1 : 1);
    unsigned GradientSteps = 1;
    std::string Csv;
};

FSettings MakeSettings(unsigned Seed, unsigned GradientSteps)
{
    FSettings S;
    S.Seed = Seed;
    S.QueueCapacity = 1u << 16;
    S.LearningStarts = S.InitialRandomSteps = 256;
    S.TrainFrequency = 1;
    S.GradientSteps = GradientSteps;
    S.MaxRecordsPerBurst = 256;
    S.MaxUpdatesPerBurst = 16;
    S.PublishEveryUpdates = 16;
    return S;
}

/** One NPC's game-side state: its outstanding decision and a synthetic episode counter. */
struct FAgent
{
    FLearner* Learner = nullptr;
    std::uint64_t Id = 0, Episode = 0;
    unsigned Step = 0;
    FDecision Pending;
    std::vector<float> Observation = std::vector<float>(O);
};

struct FTiming { std::vector<double> Micros; std::uint64_t Rejected = 0, Dropped = 0; };

void RandomObservation(std::mt19937& Rng, std::vector<float>& Out)
{ for (float& V : Out) V = std::uniform_real_distribution<float>(-1, 1)(Rng); }

/** The per-NPC game-thread work for one tick: submit the outcome of the previous decision, decide again. */
void TickAgent(FAgent& Agent, std::mt19937& Rng, FTiming* Timing)
{
    std::vector<float> Next(O);
    RandomObservation(Rng, Next);
    const auto Start = Clock::now();
    if (Agent.Pending)
    {
        const bool End = ++Agent.Step >= EpisodeLength;
        const ESubmitResult R = Agent.Learner->Submit(Agent.Pending, Agent.Pending.GetAction(),
            std::uniform_real_distribution<float>(-1, 1)(Rng), Next, false, End);
        if (Timing && R == ESubmitResult::Dropped) ++Timing->Dropped;
        else if (Timing && R != ESubmitResult::Accepted) ++Timing->Rejected;
        if (End) { Agent.Step = 0; ++Agent.Episode; }
    }
    FDecision Decision;
    Agent.Learner->Decide(Next, Agent.Id, Agent.Episode, false, Decision);
    Agent.Pending = std::move(Decision);
    if (Timing) Timing->Micros.push_back(std::chrono::duration<double, std::micro>(Clock::now() - Start).count());
    Agent.Observation.swap(Next);
}

/** Untimed: tick until the learner has stored LearningStarts records, then drain the warm-up. */
void Prefill(std::vector<FAgent>& Agents, FLearner& Learner, std::mt19937& Rng, unsigned LearningStarts)
{
    while (Learner.ReadStats().Stored < LearningStarts)
    {
        for (auto& Agent : Agents) if (Agent.Learner == &Learner) TickAgent(Agent, Rng, nullptr);
        while (Learner.Train()) {}
    }
    while (Learner.Train()) {}
}

double Percentile(std::vector<double> Values, double P)
{
    if (Values.empty()) return 0;
    std::sort(Values.begin(), Values.end());
    return Values[std::min(Values.size() - 1, static_cast<size_t>(P * (Values.size() - 1) + 0.5))];
}

/** Runs the game loop at Rate Hz for Seconds while Workers threads call Train on Learners round-robin. */
struct FRunResult { FTiming Timing; std::uint64_t Updates = 0, Records = 0, Backlog = 0; double Elapsed = 0, TickOverrun = 0; };
FRunResult RunRealTime(std::vector<FAgent>& Agents, const std::vector<FLearner*>& Learners, const FOptions& Options, unsigned Workers)
{
    FRunResult Result;
    std::vector<std::uint64_t> UpdatesBefore, StoredBefore;
    for (auto* L : Learners) { UpdatesBefore.push_back(L->ReadStats().Updates); StoredBefore.push_back(L->ReadStats().Stored); }
    std::atomic<bool> Done{false};
    std::atomic<unsigned> NextLearner{0};
    std::vector<std::thread> Threads;
    for (unsigned W = 0; W < Workers; ++W)
        Threads.emplace_back([&] {
            while (!Done.load(std::memory_order_relaxed))
            {
                bool Busy = false;
                for (size_t K = 0; K < Learners.size(); ++K)
                    Busy |= Learners[NextLearner.fetch_add(1) % Learners.size()]->Train();
                if (!Busy) std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        });
    std::mt19937 Rng(7);
    const auto Period = std::chrono::duration<double>(1.0 / Options.Rate);
    const unsigned Ticks = static_cast<unsigned>(Options.Seconds * Options.Rate + 0.5);
    const auto Start = Clock::now();
    for (unsigned T = 0; T < Ticks; ++T)
    {
        const auto TickStart = Clock::now();
        for (auto& Agent : Agents) TickAgent(Agent, Rng, &Result.Timing);
        const auto Used = Clock::now() - TickStart;
        if (Used > Period) Result.TickOverrun += std::chrono::duration<double>(Used - Period).count();
        std::this_thread::sleep_until(Start + std::chrono::duration_cast<Clock::duration>(Period * (T + 1)));
    }
    Result.Elapsed = std::chrono::duration<double>(Clock::now() - Start).count();
    Done = true;
    for (auto& Thread : Threads) Thread.join();
    for (size_t K = 0; K < Learners.size(); ++K)
    {
        const FStats S = Learners[K]->ReadStats();
        Result.Updates += S.Updates - UpdatesBefore[K];
        Result.Records += S.Stored - StoredBefore[K];
        Result.Backlog += S.PendingUpdates + S.Queued * Options.GradientSteps;
    }
    return Result;
}

bool ParseOptions(int Argc, char** Argv, FOptions& Options)
{
    for (int I = 1; I + 1 < Argc; I += 2)
    {
        const std::string Name = Argv[I], Value = Argv[I + 1];
        if (Name == "--npcs")
        {
            Options.Npcs.clear();
            std::stringstream Stream(Value);
            for (std::string Item; std::getline(Stream, Item, ',');)
            { const int N = std::atoi(Item.c_str()); if (N <= 0) return false; Options.Npcs.push_back(N); }
        }
        else if (Name == "--rate") Options.Rate = std::atof(Value.c_str());
        else if (Name == "--seconds") Options.Seconds = std::atof(Value.c_str());
        else if (Name == "--workers") Options.Workers = std::atoi(Value.c_str());
        else if (Name == "--gradient-steps") Options.GradientSteps = std::atoi(Value.c_str());
        else if (Name == "--csv") Options.Csv = Value;
        else return false;
    }
    return (Argc % 2) == 1 && !Options.Npcs.empty() && Options.Rate > 0 && Options.Seconds > 0
        && Options.Workers > 0 && Options.GradientSteps > 0;
}
}

int main(int Argc, char** Argv)
{
    FOptions Options;
    if (!ParseOptions(Argc, Argv, Options))
    {
        std::fprintf(stderr, "usage: ScalingBenchmark [--npcs 1,10,25,50,100,200] [--rate 10] [--seconds 10] "
            "[--workers N] [--gradient-steps 1] [--csv path]\n");
        return 2;
    }
    FILE* Csv = Options.Csv.empty() ? nullptr : std::fopen(Options.Csv.c_str(), "w");
    if (Csv) std::fprintf(Csv, "mode,npcs,workers,decision_us_p50,decision_us_p99,tick_overrun_s,records_per_s,"
        "updates_per_s,required_updates_per_s,backlog_updates,mb_per_learner,dropped,rejected\n");
    std::printf("profile=%s rate=%g Hz seconds=%g gradient-steps=%u hardware-threads=%u\n\n", Profile, Options.Rate,
        Options.Seconds, Options.GradientSteps, std::thread::hardware_concurrency());
    std::string Error;
    bool Ok = true;
    // Memory first, in a fresh process heap, so freed learners are not reused and hide the cost.
    double MBPerLearner = 0;
    {
        constexpr unsigned Count = 20;
        const double Before = ResidentMB();
        std::vector<std::unique_ptr<FLearner>> Owned;
        std::vector<FAgent> Agents(Count);
        std::mt19937 Rng(3);
        for (unsigned I = 0; I < Count; ++I)
        {
            Owned.push_back(FLearner::Create(Profile, MakeSettings(I + 1, 1), Error));
            if (!Owned.back()) { std::fprintf(stderr, "error: %s\n", Error.c_str()); return 2; }
            Agents[I].Learner = Owned.back().get();
            Prefill(Agents, *Owned.back(), Rng, MakeSettings(1, 1).LearningStarts);
        }
        MBPerLearner = (ResidentMB() - Before) / Count;
        std::printf("memory: %.1f MB resident per independent learner (%u learners after warm-up)\n\n", MBPerLearner, Count);
    }
    std::printf("%-12s %5s %4s %9s %9s %10s %10s %10s %10s\n", "mode", "npcs", "wrk", "p50 us", "p99 us",
        "records/s", "updates/s", "needed/s", "backlog");
    for (const unsigned N : Options.Npcs)
    {
        // Shared: one learner, every NPC submits to it, one worker trains it serially.
        {
            auto Learner = FLearner::Create(Profile, MakeSettings(1, Options.GradientSteps), Error);
            if (!Learner) { std::fprintf(stderr, "error: %s\n", Error.c_str()); return 2; }
            std::vector<FAgent> Agents(N);
            for (unsigned I = 0; I < N; ++I) { Agents[I].Learner = Learner.get(); Agents[I].Id = I; }
            std::mt19937 Rng(1);
            Prefill(Agents, *Learner, Rng, MakeSettings(1, 1).LearningStarts);
            const FRunResult R = RunRealTime(Agents, {Learner.get()}, Options, 1);
            const double Needed = N * Options.Rate * Options.GradientSteps;
            const double UpdatesPerS = R.Updates / R.Elapsed;
            std::printf("%-12s %5u %4u %9.1f %9.1f %10.0f %10.0f %10.0f %10llu\n", "shared", N, 1u,
                Percentile(R.Timing.Micros, 0.5), Percentile(R.Timing.Micros, 0.99), R.Records / R.Elapsed,
                UpdatesPerS, Needed, (unsigned long long)R.Backlog);
            if (Csv) std::fprintf(Csv, "shared,%u,1,%.2f,%.2f,%.4f,%.1f,%.1f,%.1f,%llu,,%llu,%llu\n", N,
                Percentile(R.Timing.Micros, 0.5), Percentile(R.Timing.Micros, 0.99), R.TickOverrun, R.Records / R.Elapsed,
                UpdatesPerS, Needed, (unsigned long long)R.Backlog, (unsigned long long)R.Timing.Dropped,
                (unsigned long long)R.Timing.Rejected);
            Ok = Ok && R.Timing.Rejected == 0;
        }
        // Independent: one learner per NPC, Workers threads share them round-robin.
        {
            std::vector<std::unique_ptr<FLearner>> Owned;
            std::vector<FLearner*> Learners;
            for (unsigned I = 0; I < N; ++I)
            {
                Owned.push_back(FLearner::Create(Profile, MakeSettings(I + 1, Options.GradientSteps), Error));
                if (!Owned.back()) { std::fprintf(stderr, "error: %s\n", Error.c_str()); return 2; }
                Learners.push_back(Owned.back().get());
            }
            std::vector<FAgent> Agents(N);
            std::mt19937 Rng(2);
            for (unsigned I = 0; I < N; ++I) { Agents[I].Learner = Learners[I]; Agents[I].Id = 0; }
            for (unsigned I = 0; I < N; ++I) Prefill(Agents, *Learners[I], Rng, MakeSettings(1, 1).LearningStarts);
            const unsigned Workers = std::min<unsigned>(Options.Workers, N);
            const FRunResult R = RunRealTime(Agents, Learners, Options, Workers);
            const double Needed = N * Options.Rate * Options.GradientSteps;
            std::printf("%-12s %5u %4u %9.1f %9.1f %10.0f %10.0f %10.0f %10llu\n", "independent", N, Workers,
                Percentile(R.Timing.Micros, 0.5), Percentile(R.Timing.Micros, 0.99), R.Records / R.Elapsed,
                R.Updates / R.Elapsed, Needed, (unsigned long long)R.Backlog);
            if (Csv) std::fprintf(Csv, "independent,%u,%u,%.2f,%.2f,%.4f,%.1f,%.1f,%.1f,%llu,%.2f,%llu,%llu\n", N, Workers,
                Percentile(R.Timing.Micros, 0.5), Percentile(R.Timing.Micros, 0.99), R.TickOverrun, R.Records / R.Elapsed,
                R.Updates / R.Elapsed, Needed, (unsigned long long)R.Backlog, MBPerLearner,
                (unsigned long long)R.Timing.Dropped, (unsigned long long)R.Timing.Rejected);
            Ok = Ok && R.Timing.Rejected == 0;
        }
        std::fflush(stdout);
    }
    if (Csv) std::fclose(Csv);
    return Ok ? 0 : 1;
}
