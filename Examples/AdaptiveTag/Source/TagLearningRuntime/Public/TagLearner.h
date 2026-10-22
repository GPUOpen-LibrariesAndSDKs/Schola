// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// Pure numeric boundary. Neither Unreal objects nor Schola types enter the learner.
#pragma once
#include <memory>
#ifndef TAGLEARNINGRUNTIME_API
#define TAGLEARNINGRUNTIME_API
#endif

namespace tag_learning {
constexpr unsigned QueueCapacity = 128;

struct Decision {
    float observation = 1; // Constant bias. No runner position, mode, or current route.
    float leftProbability = 0.5f;
    float chosenProbability = 0;
    int route = 0; // The actor's categorical action: -1 LEFT, +1 RIGHT.
    unsigned version = 0;
    bool valid = false;
};
struct Experience {
    Decision decision;
    float reward = 0;
    int route = 0;
    unsigned round = 0;
};
struct Stats {
    unsigned experiences = 0, updates = 0;
    unsigned trainedVersion = 0, appliedVersion = 0;
    unsigned queued = 0, dropped = 0, rejected = 0;
    float expectedLeft = 0.5f, centreLeft = 0.5f;
    float criticMse = 0, batchMs = 0, lastBurstMs = 0, lastReward = 0;
    float actorLoss = 0, entropy = 0, value = 0, advantage = 0;
    double inferenceMs = 0, publicationAgeMs = 0;
    bool ready = false, training = false, enabled = true, healthy = true;
};
class TAGLEARNINGRUNTIME_API Learner {
public:
    explicit Learner(unsigned seed = 2026);
    ~Learner();
    Learner(const Learner&) = delete;
    Learner& operator=(const Learner&) = delete;
    // Game-thread methods. Each decision uses an independently owned inference copy.
    Decision Choose();
    bool Submit(const Experience& experience);
    Stats ReadStats();
    void SetTrainingEnabled(bool enabled);
    void Stop();
private:
    struct Impl;
    std::unique_ptr<Impl> data;
};
}
