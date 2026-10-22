// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#ifdef UE_MODULE_NAME
#include "HAL/Platform.h"
#endif
#include "TagLearner.h"
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
#include <thread>

namespace tag_learning {
namespace {
namespace rlt=rl_tools;
using Clock=std::chrono::steady_clock;
using Device=rlt::devices::DEVICE_FACTORY<>;
using Index=Device::index_t;
using Policy=rlt::numeric_types::Policy<float>;
using Shape=rlt::tensor::Shape<Index,1,1,1>;
using Gradient=rlt::nn::capability::Gradient<rlt::nn::parameters::Adam,true>;
template<Index Outputs> using Network=rlt::nn_models::sequential::Build<Gradient,
    rlt::nn_models::sequential::Module<rlt::nn_models::mlp::BindConfiguration<
        rlt::nn_models::mlp::Configuration<Policy,Index,Outputs,3,16,rlt::nn::activation_functions::RELU,rlt::nn::activation_functions::IDENTITY>>>,Shape>;
using Actor=Network<2>;
using Critic=Network<1>;
using InferenceActor=Actor::CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
template<Index Width> using Tensor=rlt::Tensor<rlt::tensor::Specification<float,Index,rlt::tensor::Shape<Index,1,1,Width>,false>>;
struct AdamParameters:rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<Policy> {
    static constexpr float ALPHA=0.003f;
};
using Optimizer=rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<Policy,Index,AdamParameters>>;
struct ActorAdamParameters:AdamParameters {
    // Faster policy changes for the interactive demo; the critic keeps its original rate.
    static constexpr float ALPHA=0.006f;
};
using ActorOptimizer=rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<Policy,Index,ActorAdamParameters>>;
// Entropy is part of the actor objective; there is no action replacement or probability floor.
constexpr float EntropyWeight=0.4f;
double Ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
struct Distribution {
    double p[2],logp[2],entropy;
    bool valid;
};
Distribution Softmax(float left,float right) {
    Distribution d{};
    if(!std::isfinite(left)||!std::isfinite(right))return d;
    const double peak=std::max(left,right);
    const double logNormalizer=peak+std::log(std::exp(left-peak)+std::exp(right-peak));
    d.logp[0]=left-logNormalizer; d.logp[1]=right-logNormalizer;
    d.p[0]=std::exp(d.logp[0]); d.p[1]=std::exp(d.logp[1]);
    d.entropy=-d.p[0]*d.logp[0]-d.p[1]*d.logp[1];
    d.valid=std::isfinite(d.entropy);
    return d;
}
struct Snapshot {
    Device device;
    InferenceActor actor;
    InferenceActor::Buffer<true> buffer;
    Device::SPEC::RANDOM::ENGINE<> unusedRng;
    unsigned version;
    Distribution distribution;
    Clock::time_point created=Clock::now();
    Snapshot(Device& source,Actor& model,unsigned number):version(number) {
        rlt::malloc(device,actor); rlt::malloc(device,buffer); rlt::malloc(device,unusedRng);
        rlt::init(device,unusedRng,0);
        rlt::copy(source,device,model,actor);
        distribution=Evaluate();
    }
    ~Snapshot() { rlt::free(device,actor); rlt::free(device,buffer); rlt::free(device,unusedRng); }
    Distribution Evaluate() {
        Tensor<1> observation; Tensor<2> logits;
        rlt::set_all(device,observation,1.0f);
        rlt::evaluate(device,actor,observation,logits,buffer,unusedRng);
        return Softmax(rlt::get(device,logits,0,0,0),rlt::get(device,logits,0,0,1));
    }
};
struct Model {
    Device device;
    Actor actor;
    Critic critic;
    Actor::Buffer<true> actorBuffer;
    Critic::Buffer<true> criticBuffer;
    ActorOptimizer actorOptimizer;
    Optimizer criticOptimizer;
    Device::SPEC::RANDOM::ENGINE<> rng;
    unsigned updates=0;
    explicit Model(unsigned seed) {
        rlt::malloc(device,actor); rlt::malloc(device,critic);
        rlt::malloc(device,actorBuffer); rlt::malloc(device,criticBuffer);
        rlt::malloc(device,actorOptimizer); rlt::malloc(device,criticOptimizer);
        rlt::malloc(device,rng); rlt::init(device,rng,seed);
        rlt::init_weights(device,actor,rng); rlt::init_weights(device,critic,rng);
        // The actor itself starts at P(LEFT)=P(RIGHT)=0.5, from its initial weights.
        auto& output=actor.content.output_layer;
        rlt::set_all(device,output.weights.parameters,0.0f);
        rlt::set_all(device,output.biases.parameters,0.0f);
        auto& valueOutput=critic.content.output_layer;
        rlt::set_all(device,valueOutput.weights.parameters,0.0f);
        rlt::set_all(device,valueOutput.biases.parameters,0.0f);
        rlt::init(device,actorOptimizer); rlt::init(device,criticOptimizer);
        rlt::zero_gradient(device,actor); rlt::zero_gradient(device,critic);
        rlt::reset_optimizer_state(device,actorOptimizer,actor);
        rlt::reset_optimizer_state(device,criticOptimizer,critic);
    }
    ~Model() {
        rlt::free(device,actor); rlt::free(device,critic);
        rlt::free(device,actorBuffer); rlt::free(device,criticBuffer);
        rlt::free(device,actorOptimizer); rlt::free(device,criticOptimizer); rlt::free(device,rng);
    }
    bool Train(const Experience& experience,Stats& metrics) {
        Tensor<1> observation,valueGradient; Tensor<2> actorGradient;
        rlt::set_all(device,observation,experience.decision.observation);
        rlt::forward(device,actor,observation,actorBuffer,rng);
        rlt::forward(device,critic,observation,criticBuffer,rng);
        auto logits=rlt::output(device,actor);
        auto valueOutput=rlt::output(device,critic);
        const auto distribution=Softmax(rlt::get(device,logits,0,0,0),rlt::get(device,logits,0,0,1));
        const float value=rlt::get(device,valueOutput,0,0,0);
        const float advantage=experience.reward-value;
        const int action=experience.decision.route<0?0:1;
        // Correct for a decision made with a slightly older inference copy.
        // The ratio is treated as a constant in the policy-gradient estimator.
        const double ratio=distribution.p[action]/experience.decision.chosenProbability;
        if(!distribution.valid || !std::isfinite(value) || !std::isfinite(ratio))return false;
        for(Index i=0;i<2;++i) {
            const double policyGradient=advantage*ratio*(distribution.p[i]-(int(i)==action?1:0));
            const double entropyGradient=EntropyWeight*distribution.p[i]*(distribution.logp[i]+distribution.entropy);
            rlt::set(device,actorGradient,static_cast<float>(policyGradient+entropyGradient),0,0,i);
        }
        rlt::set_all(device,valueGradient,value-experience.reward);
        rlt::zero_gradient(device,actor); rlt::zero_gradient(device,critic);
        rlt::backward(device,actor,observation,actorGradient,actorBuffer);
        rlt::backward(device,critic,observation,valueGradient,criticBuffer);
        rlt::step(device,actorOptimizer,actor); rlt::step(device,criticOptimizer,critic);
        ++updates;
        metrics.criticMse=advantage*advantage; metrics.value=value; metrics.advantage=advantage;
        metrics.entropy=static_cast<float>(distribution.entropy);
        metrics.actorLoss=static_cast<float>(-advantage*ratio*distribution.logp[action]-EntropyWeight*distribution.entropy);
        return std::isfinite(metrics.criticMse) && std::isfinite(metrics.actorLoss);
    }
};
}
struct Learner::Impl {
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Experience> queue;
    std::unique_ptr<Model> model;
    std::unique_ptr<Snapshot> pending,current;
    Stats stats;
    std::atomic<bool> stop{false},enabled{true};
    std::thread worker;
    std::mt19937 actorRandom;
    double lastInferenceMs=0;
    explicit Impl(unsigned seed):model(new Model(seed)),actorRandom(seed+3000) {
        // Initialization is synchronous so even the very first choice belongs to the actor.
        current=std::make_unique<Snapshot>(model->device,model->actor,0);
        stats.ready=true; stats.healthy=current->distribution.valid;
        worker=std::thread([this]{Run();});
    }
    void Publish() {
        auto snapshot=std::make_unique<Snapshot>(model->device,model->actor,model->updates);
        std::lock_guard<std::mutex> lock(mutex);
        stats.healthy=stats.healthy && snapshot->distribution.valid;
        if(stats.healthy)pending=std::move(snapshot);
        stats.trainedVersion=model->updates;
    }
    void Run() {
        while(!stop) {
            Experience experience;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock,[this]{return stop || (!queue.empty() && enabled);});
                if(stop)break;
                experience=queue.front(); queue.pop_front(); stats.training=true;
            }
            const auto before=Clock::now();
            Stats metrics;
            const bool healthy=model->Train(experience,metrics);
            if(healthy)Publish();
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++stats.experiences; stats.updates=model->updates;
                stats.criticMse=metrics.criticMse; stats.actorLoss=metrics.actorLoss; stats.entropy=metrics.entropy;
                stats.value=metrics.value; stats.advantage=metrics.advantage;
                stats.batchMs=stats.lastBurstMs=static_cast<float>(Ms(before)); stats.lastReward=experience.reward;
                stats.healthy=stats.healthy && healthy; stats.training=false;
            }
        }
    }
    void ApplyPending() {
        std::lock_guard<std::mutex> lock(mutex);
        if(pending && enabled)current=std::move(pending);
    }
};
Learner::Learner(unsigned seed):data(new Impl(seed)) {}
Learner::~Learner() { Stop(); }
void Learner::Stop() { data->stop=true; data->wake.notify_all(); if(data->worker.joinable())data->worker.join(); }
void Learner::SetTrainingEnabled(bool enabled) { data->enabled=enabled; data->wake.notify_all(); }
Decision Learner::Choose() {
    const auto start=Clock::now();
    data->ApplyPending();
    const auto distribution=data->current->Evaluate();
    Decision decision;
    decision.version=data->current->version; decision.valid=distribution.valid;
    if(distribution.valid) {
        // This is sampling the categorical actor, not replacing or modifying its action.
        const int action=std::uniform_real_distribution<double>(0,1)(data->actorRandom)<distribution.p[0]?0:1;
        decision.route=action==0?-1:1;
        decision.leftProbability=static_cast<float>(distribution.p[0]);
        decision.chosenProbability=static_cast<float>(distribution.p[action]);
    }
    data->lastInferenceMs=Ms(start);
    return decision;
}
bool Learner::Submit(const Experience& e) {
    std::lock_guard<std::mutex> lock(data->mutex);
    if(!data->enabled || data->stop)return false;
    const float probability=e.decision.route<0?e.decision.leftProbability:1-e.decision.leftProbability;
    if(!e.decision.valid || !std::isfinite(e.reward) || e.decision.observation!=1
        || (e.decision.route!=-1 && e.decision.route!=1) || e.route!=e.decision.route
        || !(e.decision.chosenProbability>0 && e.decision.chosenProbability<=1)
        || !std::isfinite(e.decision.leftProbability) || std::abs(probability-e.decision.chosenProbability)>1e-5f) {
        ++data->stats.rejected; return false;
    }
    if(data->queue.size()>=QueueCapacity) { ++data->stats.dropped; return false; }
    data->queue.push_back(e); data->wake.notify_one(); return true;
}
Stats Learner::ReadStats() {
    data->ApplyPending();
    std::lock_guard<std::mutex> lock(data->mutex);
    Stats stats=data->stats;
    stats.queued=static_cast<unsigned>(data->queue.size()); stats.enabled=data->enabled;
    stats.ready=data->current->distribution.valid; stats.inferenceMs=data->lastInferenceMs;
    stats.appliedVersion=data->current->version;
    stats.expectedLeft=stats.centreLeft=static_cast<float>(data->current->distribution.p[0]);
    stats.entropy=static_cast<float>(data->current->distribution.entropy);
    stats.publicationAgeMs=Ms(data->current->created);
    return stats;
}
}
