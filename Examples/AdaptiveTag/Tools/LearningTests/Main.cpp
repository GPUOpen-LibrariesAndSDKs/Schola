// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "TagLearner.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <thread>

bool Wait(tag_learning::Learner& learner,unsigned count) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(std::chrono::steady_clock::now()<deadline) {
        auto stats=learner.ReadStats();
        if(stats.experiences>=count && !stats.training && stats.queued==0) {
            if(!stats.healthy) std::cout<<"Unhealthy learner at experience "<<count<<" mse="<<stats.criticMse<<" p="<<stats.expectedLeft<<std::endl;
            return stats.healthy;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::cout<<"Timed out waiting for experience "<<count<<std::endl;
    return false;
}
int main(int argc,char** argv) {
    std::ofstream curve(argc>1?argv[1]:"learning-curve.csv");
    curve<<"seed,phase,round,expected_left,centre_left,updates,critic_mse,reward\n";
    bool passed=true;
    for(unsigned seed : {0u,1u,2u,2026u}) {
        tag_learning::Learner learner(seed);
        std::mt19937 random(seed+6000);
        std::uniform_real_distribution<float> draw(0,1);
        unsigned count=0;
        for(unsigned phase=0;phase<2;++phase) {
            float lastProbability=0;
            unsigned catches=0;
            for(unsigned round=0;round<256;++round) {
                auto decision=learner.Choose();
                const int route=decision.route;
                const int runner=draw(random)<(phase==0?0.8f:0.2f) ? -1:1;
                const float reward=route==runner?1.0f:-1.0f;
                passed=passed && decision.valid && decision.observation==1 && (route==-1 || route==1)
                    && decision.leftProbability>=0 && decision.leftProbability<=1
                    && std::abs(decision.chosenProbability-(route<0?decision.leftProbability:1-decision.leftProbability))<1e-5f;
                tag_learning::Experience experience{decision,reward,route,++count};
                if(!learner.Submit(experience) || !Wait(learner,count))return 2;
                const auto stats=learner.ReadStats();
                passed=passed && stats.updates==count;
                curve<<seed<<','<<phase<<','<<round+1<<','<<stats.expectedLeft<<','<<stats.centreLeft<<','<<stats.updates<<','<<stats.criticMse<<','<<reward<<'\n';
                if(round>=192) {lastProbability+=stats.expectedLeft/64; catches+=reward>0;}
            }
            bool learned=phase==0?lastProbability>0.65f:lastProbability<0.35f;
            std::cout<<"seed="<<seed<<" phase="<<phase<<" mean_p_left_last64="<<lastProbability<<" catches_last64="<<catches<<" learned="<<learned<<'\n';
            passed=passed && learned;
        }
        const auto before=learner.ReadStats();
        learner.SetTrainingEnabled(false);
        auto decision=learner.Choose();
        passed=passed && !learner.Submit({decision,1,-1,999});
        unsigned leftSamples=0;
        for(int i=0;i<4096;++i)leftSamples+=learner.Choose().route<0;
        const auto after=learner.ReadStats();
        passed=passed && before.updates==after.updates && before.expectedLeft==after.expectedLeft;
        passed=passed && std::abs(float(leftSamples)/4096-after.expectedLeft)<0.025f;
        learner.SetTrainingEnabled(true);
        auto invalid=tag_learning::Experience{decision,std::numeric_limits<float>::quiet_NaN(),-1,999};
        passed=passed && !learner.Submit(invalid) && learner.ReadStats().rejected==1;
        learner.Stop();
    }
    { tag_learning::Learner cancelled; cancelled.Stop(); }
    for(float reward : {-1.0f,1.0f}) {
        tag_learning::Learner initial(77);
        auto decision=initial.Choose();
        passed=passed && decision.valid && std::abs(decision.leftProbability-0.5f)<1e-6f;
        passed=passed && initial.Submit({decision,reward,decision.route,1}) && Wait(initial,1);
        passed=passed && initial.ReadStats().updates==1; // Actor acts and learns from round one.
        const float pLeft=initial.ReadStats().expectedLeft;
        const float chosenNow=decision.route<0?pLeft:1-pLeft;
        passed=passed && (reward>0?chosenNow>0.5f:chosenNow<0.5f);
        auto mismatch=initial.Choose();
        passed=passed && !initial.Submit({mismatch,1,-mismatch.route,2});
    }
    std::cout<<"PASS="<<passed<<'\n';
    return passed?0:1;
}
