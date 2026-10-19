// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "Policies/AdaptiveNPCPolicy.h"
#include "AdaptiveDecisionTestPolicy.generated.h"

/** Test-only controlled responses and clock. No background tasks, Python, sleeps or network. */
UCLASS()
class UAdaptiveDecisionTestPolicy : public UAdaptiveNPCPolicy
{
    GENERATED_BODY()

public:
    bool Start(const FGuid& Id, int64 Agent = 0, int64 Episode = 0) { return StartSession(Id, Agent, Episode); }
    bool Deliver(const FAdaptiveDecision& Result) { return CompleteDecision(Result); }
    bool Reject(const FAdaptiveDecisionRequest& Ticket, const FString& Error) { return FailDecision(Ticket, Error); }

    FAdaptiveDecisionRequest Sent;
    FAdaptiveDecision ImmediateResult;
    double Now = 100.0;
    int32 DispatchCount = 0;
    bool bAcceptDispatch = true;
    bool bCompleteImmediately = false;

protected:
    double GetTimeSeconds() const override { return Now; }
    bool DispatchDecision(const FAdaptiveDecisionRequest& Input) override
    {
        ++DispatchCount;
        Sent = Input;
        if (bCompleteImmediately && bAcceptDispatch)
        {
            auto Result = ImmediateResult;
            Result.SessionId = Input.SessionId;
            Result.AgentId = Input.AgentId;
            Result.EpisodeId = Input.EpisodeId;
            Result.DecisionId = Input.DecisionId;
            CompleteDecision(Result);
        }
        return bAcceptDispatch;
    }
};
