// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// D3 chase adaptation test run on the plugin's own learner (Schola::RLTools::FLearner), without Unreal.
//
// Protocol (same as Examples/AdaptiveTag/Evidence/SAC/d3/adapt-final): train on `rightturn`, capture an
// immutable snapshot, switch to `corner` and keep learning. Every evaluation runs the deterministic
// actor on the same 100 start states as chase_sim (eval_seed 4242, same seed families).
//
// Criteria (fixed in deliverables/D3/adaptive-npc-evidence.md, never loosened):
//   1. learner >= 0.50 on rightturn at the switch
//   2. learner <= 0.80 on corner right after the switch
//   3. learner back to >= 0.80 on corner within 20,000 moves of the switch
//   4. at the first evaluation at or past 20,000 moves, snapshot is more than 0.20 below the learner
//   5. 1-4 hold on every seed (checked across runs by run-chase.sh)
// After the window the run continues for --post-moves to record stability (reported, not pass/fail).
//
// Checkpoint comparison (both runs from one starting checkpoint, as the project guide's acceptance asks):
//   --save-checkpoint P   after pretraining, save a FullResume checkpoint to P, then continue as usual.
//   --start-checkpoint P  skip pretraining. Load P into two fresh learners in this process: one keeps
//                         learning, the other has learning disabled and is the frozen control. The frozen
//                         learner must still report the loaded version at the end.
//
// Usage: ChaseAdaptation [--seed N] [--replay 10000|20000|60000] [--learning-rate X]
//                        [--updates-per-move N] [--pretrain-episodes N] [--chunk-episodes N]
//                        [--window-moves N] [--post-moves N] [--eval-episodes N] [--csv path]
//                        [--save-checkpoint path | --start-checkpoint path]
// Exit code: 0 if criteria 1-4 pass, 1 if any fails, 2 on a usage or learner error.
#include "RLToolsLearner.h"
#include "ChaseWorld.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace Schola::RLTools;
using namespace online_sac;

namespace
{
constexpr unsigned O = ChaseObservationSize, A = ChaseActionSize;
// Seed families copied from ChaseSession.cpp so the evaluation start states match chase_sim exactly.
constexpr uint64_t TrainFamily = 1, EvalFamily = 2;
uint64_t SeedFor(unsigned Seed, uint64_t Family, unsigned Index)
{ return (static_cast<uint64_t>(Seed) << 32) ^ (Family * 0x9E3779B97F4A7C15ull) ^ Index; }

struct FOptions
{
    unsigned Seed = 1, Replay = 10000, UpdatesPerMove = 8, PretrainEpisodes = 150, ChunkEpisodes = 5;
    unsigned WindowMoves = 20000, PostMoves = 20000, EvalEpisodes = 100;
    float LearningRate = 0.001f; // chase_sim's ONLINE_SAC_LR
    std::string Csv, SaveCheckpoint, StartCheckpoint;
};

struct FEval { double CatchRate = 0, MeanSteps = 0; };

FEval Evaluate(const FPolicy& Policy, const ChaseConfig& Config, RunnerPattern Pattern, unsigned Episodes)
{
    FEval Result;
    unsigned Caught = 0;
    double Steps = 0;
    std::vector<float> Observation(O), Action(A);
    for (unsigned I = 0; I < Episodes; ++I)
    {
        ChaseRng StartRng(SeedFor(Config.eval_seed, EvalFamily, I));
        ChaseWorld World;
        World.Reset(Config, Pattern, StartRng);
        for (;;)
        {
            World.Observe(Observation.data());
            if (!Policy.Evaluate(Observation, Action)) std::fill(Action.begin(), Action.end(), 0.0f);
            const ChaseStepOutcome Outcome = World.Step(Action.data());
            if (Outcome.terminated || Outcome.truncated) break;
        }
        if (World.State().caught) { ++Caught; Steps += World.State().step; }
    }
    Result.CatchRate = Episodes ? static_cast<double>(Caught) / Episodes : 0;
    Result.MeanSteps = Caught ? Steps / Caught : 0;
    return Result;
}

class FRun
{
public:
    FRun(const FOptions& InOptions, std::unique_ptr<FLearner> InLearner)
        : Options(InOptions), Learner(std::move(InLearner)), Rng(SeedFor(InOptions.Seed, TrainFamily, 0)) {}

    // Plays whole episodes with stochastic actions, submitting every move and draining training in lockstep
    // (like chase_sim's lockstep mode) so a seed gives the same result on every run.
    bool TrainEpisodes(RunnerPattern Pattern, unsigned Episodes)
    {
        std::vector<float> Observation(O), Next(O);
        for (unsigned E = 0; E < Episodes; ++E)
        {
            ChaseWorld World;
            World.Reset(Config, Pattern, Rng);
            ++Episode;
            for (;;)
            {
                World.Observe(Observation.data());
                FDecision Decision;
                if (!Learner->Decide(Observation, 0, Episode, false, Decision)) return Fail("Decide failed");
                const std::vector<float> Action = Decision.GetAction();
                const ChaseStepOutcome Outcome = World.Step(Action.data());
                World.Observe(Next.data());
                const ESubmitResult Result = Learner->Submit(Decision, Action, Outcome.reward, Next, Outcome.terminated, Outcome.truncated);
                if (Result != ESubmitResult::Accepted) return Fail("Submit was not accepted");
                // Start mode: the frozen control sees the same move and must refuse it without changing.
                if (Frozen)
                {
                    FDecision FrozenDecision;
                    if (!Frozen->Decide(Observation, 0, Episode, true, FrozenDecision)) return Fail("Frozen Decide failed");
                    if (Frozen->Submit(FrozenDecision, FrozenDecision.GetAction(), Outcome.reward, Next, Outcome.terminated,
                            Outcome.truncated) != ESubmitResult::Frozen) return Fail("Frozen control accepted an experience");
                    ++FrozenSubmits;
                    if (Frozen->Train()) return Fail("Frozen control trained");
                }
                while (Learner->Train()) {}
                ++Moves;
                if (Outcome.terminated || Outcome.truncated) break;
            }
        }
        return true;
    }

    const FOptions& Options;
    std::unique_ptr<FLearner> Learner;
    FLearner* Frozen = nullptr; // start mode only
    uint64_t FrozenSubmits = 0;
    ChaseConfig Config;
    ChaseRng Rng;
    uint64_t Episode = 0, Moves = 0;
private:
    bool Fail(const char* What) { std::fprintf(stderr, "error: %s\n", What); return false; }
};

bool ParseUnsigned(const char* Text, unsigned& Out)
{
    char* End = nullptr;
    const unsigned long Value = std::strtoul(Text, &End, 10);
    if (!*Text || *End || Value > 100000000ul) return false;
    Out = static_cast<unsigned>(Value);
    return true;
}

bool ParseOptions(int Argc, char** Argv, FOptions& Options)
{
    for (int I = 1; I < Argc; ++I)
    {
        const std::string Name = Argv[I];
        if (I + 1 >= Argc) return false;
        const char* Value = Argv[++I];
        bool Ok = true;
        if (Name == "--seed") Ok = ParseUnsigned(Value, Options.Seed);
        else if (Name == "--replay") Ok = ParseUnsigned(Value, Options.Replay);
        else if (Name == "--updates-per-move") Ok = ParseUnsigned(Value, Options.UpdatesPerMove) && Options.UpdatesPerMove;
        else if (Name == "--pretrain-episodes") Ok = ParseUnsigned(Value, Options.PretrainEpisodes);
        else if (Name == "--chunk-episodes") Ok = ParseUnsigned(Value, Options.ChunkEpisodes) && Options.ChunkEpisodes;
        else if (Name == "--window-moves") Ok = ParseUnsigned(Value, Options.WindowMoves);
        else if (Name == "--post-moves") Ok = ParseUnsigned(Value, Options.PostMoves);
        else if (Name == "--eval-episodes") Ok = ParseUnsigned(Value, Options.EvalEpisodes) && Options.EvalEpisodes;
        else if (Name == "--learning-rate") { char* End = nullptr; Options.LearningRate = std::strtof(Value, &End); Ok = *Value && !*End; }
        else if (Name == "--csv") Options.Csv = Value;
        else if (Name == "--save-checkpoint") Options.SaveCheckpoint = Value;
        else if (Name == "--start-checkpoint") Options.StartCheckpoint = Value;
        else Ok = false;
        if (!Ok) return false;
    }
    return Options.SaveCheckpoint.empty() || Options.StartCheckpoint.empty();
}
}

int main(int Argc, char** Argv)
{
    FOptions Options;
    if (!ParseOptions(Argc, Argv, Options))
    {
        std::fprintf(stderr, "usage: ChaseAdaptation [--seed N] [--replay 10000|20000|60000] [--learning-rate X] "
            "[--updates-per-move N] [--pretrain-episodes N] [--chunk-episodes N] [--window-moves N] [--post-moves N] "
            "[--eval-episodes N] [--csv path] [--save-checkpoint path | --start-checkpoint path]\n");
        return 2;
    }
    const std::string Profile = "sac/chase-r" + std::to_string(Options.Replay / 1000) + "k";
    FSettings Settings;
    Settings.Seed = Options.Seed;
    Settings.LearningRate = Options.LearningRate;
    Settings.QueueCapacity = 4096;
    // chase_sim: 1000 random warm-up moves, then UpdatesPerMove gradient steps after every move.
    Settings.LearningStarts = Settings.InitialRandomSteps = 1000;
    Settings.TrainFrequency = 1;
    Settings.GradientSteps = Options.UpdatesPerMove;
    Settings.MaxRecordsPerBurst = 64;
    Settings.MaxUpdatesPerBurst = 64;
    Settings.PublishEveryUpdates = Options.UpdatesPerMove; // one new policy version per move
    std::string Error;
    auto Learner = FLearner::Create(Profile, Settings, Error);
    if (!Learner) { std::fprintf(stderr, "error: %s: %s\n", Profile.c_str(), Error.c_str()); return 2; }

    // Start mode: the adaptive learner and the frozen control both come from the same file.
    std::unique_ptr<FLearner> Frozen;
    FCheckpointInfo Loaded;
    if (!Options.StartCheckpoint.empty())
    {
        Frozen = FLearner::Create(Profile, Settings, Error);
        if (!Frozen) { std::fprintf(stderr, "error: %s: %s\n", Profile.c_str(), Error.c_str()); return 2; }
        for (FLearner* Target : {Learner.get(), Frozen.get()})
        {
            const ECheckpointResult Result = Target->LoadCheckpoint(Options.StartCheckpoint, Error, &Loaded);
            if (Result != ECheckpointResult::Ok)
            {
                std::fprintf(stderr, "error: load %s: %s: %s\n", Options.StartCheckpoint.c_str(), ToString(Result), Error.c_str());
                return 2;
            }
        }
        Frozen->SetLearningEnabled(false);
        std::printf("loaded %s: profile=%s mode=%u version=%llu updates=%llu replay=%llu\n",
            Options.StartCheckpoint.c_str(), Loaded.ProfileId.c_str(), static_cast<unsigned>(Loaded.Mode),
            (unsigned long long)Loaded.Version, (unsigned long long)Loaded.Updates, (unsigned long long)Loaded.ReplaySize);
    }

    FILE* Csv = Options.Csv.empty() ? nullptr : std::fopen(Options.Csv.c_str(), "w");
    if (!Options.Csv.empty() && !Csv) { std::fprintf(stderr, "error: cannot write %s\n", Options.Csv.c_str()); return 2; }
    if (Csv) std::fprintf(Csv, "phase,pattern,episodes,total_moves,moves_since_switch,learner_catch,learner_steps,snapshot_catch,version,replay_size,seconds\n");

    FRun Run(Options, std::move(Learner));
    Run.Frozen = Frozen.get();
    const auto Start = std::chrono::steady_clock::now();
    std::shared_ptr<const FPolicy> Snapshot;
    uint64_t SwitchMoves = 0;
    auto Seconds = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - Start).count(); };
    auto Record = [&](const char* Phase, RunnerPattern Pattern) {
        if (Frozen && Snapshot) Snapshot = Frozen->CapturePolicy(); // re-read: the control must not change
        const auto Current = Run.Learner->CapturePolicy();
        const FEval Learned = Evaluate(*Current, Run.Config, Pattern, Options.EvalEpisodes);
        const double SnapshotCatch = Snapshot ? Evaluate(*Snapshot, Run.Config, Pattern, Options.EvalEpisodes).CatchRate : -1;
        const FStats Stats = Run.Learner->ReadStats();
        const uint64_t Since = Snapshot ? Run.Moves - SwitchMoves : 0;
        std::printf("%-8s %-9s ep=%-5llu moves=%-7llu since=%-6llu learner=%.2f (%.1f steps) snapshot=%s v=%llu replay=%u %.0fs\n",
            Phase, PatternName(Pattern), (unsigned long long)Run.Episode, (unsigned long long)Run.Moves,
            (unsigned long long)Since, Learned.CatchRate, Learned.MeanSteps,
            Snapshot ? std::to_string(SnapshotCatch).substr(0, 4).c_str() : "-",
            (unsigned long long)Stats.Version, Stats.ReplaySize, Seconds());
        std::fflush(stdout);
        if (Csv)
        {
            std::fprintf(Csv, "%s,%s,%llu,%llu,%llu,%.4f,%.2f,%.4f,%llu,%u,%.1f\n", Phase, PatternName(Pattern),
                (unsigned long long)Run.Episode, (unsigned long long)Run.Moves, (unsigned long long)Since,
                Learned.CatchRate, Learned.MeanSteps, SnapshotCatch, (unsigned long long)Stats.Version, Stats.ReplaySize, Seconds());
            std::fflush(Csv);
        }
        return std::pair<FEval, double>{Learned, SnapshotCatch};
    };

    std::printf("profile=%s seed=%u lr=%g updates/move=%u pretrain=%u episodes, window=%u moves, post=%u moves\n",
        Profile.c_str(), Options.Seed, Options.LearningRate, Options.UpdatesPerMove, Options.PretrainEpisodes,
        Options.WindowMoves, Options.PostMoves);

    double PreSwitch = 0;
    FStats FrozenAtLoad;
    if (Frozen)
    {
        // Pretraining happened in the process that saved the file. Training episodes use a different
        // start-state stream from the saving run, which cannot be restored from the checkpoint.
        Run.Rng = ChaseRng(SeedFor(Options.Seed, TrainFamily, 1));
        PreSwitch = Record("loaded", RunnerPattern::RightTurn).first.CatchRate;
        Snapshot = Frozen->CapturePolicy();
        FrozenAtLoad = Frozen->ReadStats();
        const double FrozenRightTurn = Evaluate(*Snapshot, Run.Config, RunnerPattern::RightTurn, Options.EvalEpisodes).CatchRate;
        std::printf("frozen control on rightturn: %.2f (learner %.2f, same file)\n", FrozenRightTurn, PreSwitch);
    }
    else
    {
        // Pre-switch: 5 blocks of PretrainEpisodes/5 on rightturn, as in adapt-final/seed*.script.
        const unsigned Blocks = 5, PerBlock = std::max(1u, Options.PretrainEpisodes / Blocks);
        PreSwitch = Record("pretrain", RunnerPattern::RightTurn).first.CatchRate;
        for (unsigned B = 0; B < Blocks; ++B)
        {
            if (!Run.TrainEpisodes(RunnerPattern::RightTurn, PerBlock)) return 2;
            PreSwitch = Record("pretrain", RunnerPattern::RightTurn).first.CatchRate;
        }
        Snapshot = Run.Learner->CapturePolicy(); // immutable; later training publishes new policies
        if (!Options.SaveCheckpoint.empty())
        {
            const ECheckpointResult Result = Run.Learner->SaveCheckpoint(Options.SaveCheckpoint, ECheckpointMode::FullResume, Error);
            if (Result != ECheckpointResult::Ok)
            {
                std::fprintf(stderr, "error: save %s: %s: %s\n", Options.SaveCheckpoint.c_str(), ToString(Result), Error.c_str());
                return 2;
            }
            const FStats Saved = Run.Learner->ReadStats();
            std::printf("saved %s: version=%llu updates=%llu replay=%u\n", Options.SaveCheckpoint.c_str(),
                (unsigned long long)Saved.Version, (unsigned long long)Saved.Updates, Saved.ReplaySize);
        }
    }
    SwitchMoves = Run.Moves;
    const double AfterSwitch = Record("switch", RunnerPattern::Corner).first.CatchRate;

    uint64_t RecoveredAt = 0;
    bool HaveWindowEval = false;
    double WindowLearner = 0, WindowSnapshot = 0;
    uint64_t WindowAt = 0;
    std::vector<double> PostCatch;
    while (Run.Moves - SwitchMoves < uint64_t(Options.WindowMoves) + Options.PostMoves)
    {
        if (!Run.TrainEpisodes(RunnerPattern::Corner, Options.ChunkEpisodes)) return 2;
        const uint64_t Since = Run.Moves - SwitchMoves;
        const bool Post = HaveWindowEval;
        const auto [Learned, SnapshotCatch] = Record(Post ? "post" : "adapt", RunnerPattern::Corner);
        if (!RecoveredAt && Since <= Options.WindowMoves && std::llround(Learned.CatchRate * 100.0) >= 80) RecoveredAt = Since;
        if (!HaveWindowEval && Since >= Options.WindowMoves)
        {
            HaveWindowEval = true;
            WindowLearner = Learned.CatchRate;
            WindowSnapshot = SnapshotCatch;
            WindowAt = Since;
        }
        else if (Post) PostCatch.push_back(Learned.CatchRate);
    }
    if (Csv) std::fclose(Csv);

    // Compare in whole percentage points: catch rates are counts over the evaluation episodes, and a
    // floating-point gap of exactly 0.20 (e.g. 0.92 - 0.72) must not pass "more than 0.20".
    auto Points = [](double Rate) { return std::llround(Rate * 100.0); };
    const bool C1 = Points(PreSwitch) >= 50, C2 = Points(AfterSwitch) <= 80, C3 = RecoveredAt != 0;
    const bool C4 = HaveWindowEval && Points(WindowLearner) - Points(WindowSnapshot) > 20;
    // Start mode only: the frozen control never trained or published after the load.
    bool ControlFrozen = true;
    if (Frozen)
    {
        const FStats End = Frozen->ReadStats();
        ControlFrozen = End.Version == FrozenAtLoad.Version && End.Updates == FrozenAtLoad.Updates && !End.LearningEnabled
            && Run.FrozenSubmits > 0 && End.Frozen - FrozenAtLoad.Frozen == Run.FrozenSubmits
            && End.Accepted == FrozenAtLoad.Accepted;
    }
    std::printf("\nRESULT seed=%u profile=%s lr=%g%s\n", Options.Seed, Profile.c_str(), Options.LearningRate,
        Frozen ? " start=checkpoint" : "");
    if (Frozen)
        std::printf("%s control-frozen: refused all %llu moves it was given; version %llu, updates %llu unchanged since load\n",
            ControlFrozen ? "PASS" : "FAIL", (unsigned long long)Run.FrozenSubmits,
            (unsigned long long)FrozenAtLoad.Version, (unsigned long long)FrozenAtLoad.Updates);
    std::printf("%s criterion1 rightturn-before-switch: %.2f (>= 0.50)\n", C1 ? "PASS" : "FAIL", PreSwitch);
    std::printf("%s criterion2 corner-after-switch: %.2f (<= 0.80)\n", C2 ? "PASS" : "FAIL", AfterSwitch);
    if (C3) std::printf("PASS criterion3 recovered: >= 0.80 at %llu moves (<= %u)\n", (unsigned long long)RecoveredAt, Options.WindowMoves);
    else std::printf("FAIL criterion3 recovered: never >= 0.80 within %u moves\n", Options.WindowMoves);
    std::printf("%s criterion4 snapshot-gap at %llu moves: learner %.2f, snapshot %.2f, gap %.2f (> 0.20)\n",
        C4 ? "PASS" : "FAIL", (unsigned long long)WindowAt, WindowLearner, WindowSnapshot, WindowLearner - WindowSnapshot);
    if (!PostCatch.empty())
    {
        const auto [Low, High] = std::minmax_element(PostCatch.begin(), PostCatch.end());
        std::printf("INFO post-window stability (%zu evals over %u moves): min %.2f, max %.2f, final %.2f\n",
            PostCatch.size(), Options.PostMoves, *Low, *High, PostCatch.back());
    }
    std::printf("INFO wall time %.0f s, %llu moves\n", Seconds(), (unsigned long long)Run.Moves);
    return C1 && C2 && C3 && C4 && ControlFrozen ? 0 : 1;
}
