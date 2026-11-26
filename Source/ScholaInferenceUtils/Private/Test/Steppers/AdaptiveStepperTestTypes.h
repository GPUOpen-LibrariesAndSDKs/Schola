// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Agent/AgentInterface.h"
#include "Steppers/AdaptiveAgentInterface.h"
#include "Points/BoxPoint.h"
#include "Spaces/BoxSpace.h"
#include "AdaptiveStepperTestTypes.generated.h"

/** Adaptive test agent: one Box observation value set by the test, 1-D Box action, scripted reward and ending. */
UCLASS()
class UAdaptiveStepperTestAgent : public UObject, public IAgent, public IAdaptiveAgent
{
    GENERATED_BODY()
public:
    void Define_Implementation(FInteractionDefinition& OutDefinition) override
    {
        OutDefinition.ObsSpaceDefn.InitializeAs<FBoxSpace>(TArray<float>{-1}, TArray<float>{1});
        OutDefinition.ActionSpaceDefn.InitializeAs<FBoxSpace>(TArray<float>{-1}, TArray<float>{1});
    }
    void Act_Implementation(const FInstancedStruct& InAction) override
    {
        const FBoxPoint* Box = InAction.GetPtr<FBoxPoint>();
        Applied.Add(Box && Box->Values.Num() > 0 ? Box->Values[0] : -2.0f);
    }
    void Observe_Implementation(FInstancedStruct& OutObservations) override
    {
        ++ObserveCount;
        OutObservations.InitializeAs<FBoxPoint>(TArray<float>{ObservationValue});
    }
    void SetStatus_Implementation(EAgentStatus NewStatus) override { Status = NewStatus; }
    EAgentStatus GetStatus_Implementation() override { return Status; }

    float ConsumeReward_Implementation() override { const float Value = Reward; Reward = 0; ++RewardReads; return Value; }
    void GetEpisodeStatus_Implementation(bool& bOutTerminated, bool& bOutTruncated) override
    { bOutTerminated = bTerminated; bOutTruncated = bTruncated; }
    void ResetEpisode_Implementation() override { ++ResetCount; bTerminated = bTruncated = false; ObservationValue = 0; }

    TArray<float> Applied;
    float ObservationValue = 0;
    float Reward = 0;
    bool bTerminated = false;
    bool bTruncated = false;
    int32 ObserveCount = 0;
    int32 RewardReads = 0;
    int32 ResetCount = 0;
    EAgentStatus Status = EAgentStatus::Running;
};
