// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Points/Point.h"
#include "StructUtils/InstancedStruct.h"
#include "AdaptiveLearningTypes.generated.h"

/** Local setup and remote session readiness are separate states. */
UENUM(BlueprintType)
enum class EAdaptiveSessionState : uint8
{
    Uninitialized,
    Initialized UMETA(DisplayName = "Locally Initialized"),
    Connecting,
    Ready,
    Failed,
    Stopped
};

/** Only Ready can release an action. Expired means the local action age limit was reached. */
UENUM(BlueprintType)
enum class EAdaptiveRequestStatus : uint8
{
    Unknown,
    Pending,
    Ready,
    Failed,
    Cancelled,
    Expired
};

/** Copied input for one decision. The bridge assigns its identity; this is not a wire format. */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FAdaptiveDecisionRequest
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    FGuid SessionId;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 AgentId = INDEX_NONE;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 EpisodeId = INDEX_NONE;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 DecisionId = INDEX_NONE;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    TInstancedStruct<FPoint> Observation;
};

/**
 * A returned action and the identity of the decision that produced it.
 * Invalid GUIDs and negative IDs mean unassigned; zero is a valid numeric ID/version.
 * This is a data record, not proof that a result has been validated or executed.
 */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FAdaptiveDecision
{
    GENERATED_BODY()

    /** Identifies one learner session; assigned by the future session handshake. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    FGuid SessionId;

    /** NPC identity within the session. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 AgentId = INDEX_NONE;

    /** Encounter identity; resetting an encounter does not restart learning. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 EpisodeId = INDEX_NONE;

    /** Matches this result to its originating request. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 DecisionId = INDEX_NONE;

    /** Inference snapshot that chose the action, which may be older than the latest snapshot. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 PolicyVersion = INDEX_NONE;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    TInstancedStruct<FPoint> Action;
};

/**
 * A completed gameplay outcome for learning, attributed to its original decision.
 * Records are passive containers; submission and validation belong to later integration.
 */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FAdaptiveExperience
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    FGuid SessionId;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 AgentId = INDEX_NONE;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 EpisodeId = INDEX_NONE;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 DecisionId = INDEX_NONE;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    int64 PolicyVersion = INDEX_NONE;

    /** Observation that originated the decision. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    TInstancedStruct<FPoint> Observation;

    /** Command actually applied by gameplay, rather than an unrelated fallback or preferred action. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    TInstancedStruct<FPoint> ExecutedAction;

    /** Accumulated reward for the action interval. Finite negative rewards are valid. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    float Reward = 0.0f;

    /** Outcome observation, captured before resetting an ended encounter. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    TInstancedStruct<FPoint> NextObservation;

    /** The task reached a terminal condition. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    bool bTerminated = false;

    /** A time limit or other external interruption ended the encounter. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive")
    bool bTruncated = false;
};
