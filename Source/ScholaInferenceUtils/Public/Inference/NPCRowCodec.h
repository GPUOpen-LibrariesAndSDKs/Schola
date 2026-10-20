// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Common/InteractionDefinition.h"

/** Types in model bindings. Scheduler transport always uses float rows. */
enum class ENPCRowDType : uint8 { Float, Int64, Bool };

struct SCHOLAINFERENCEUTILS_API FNPCRowElement
{
    FString Name;
    ENPCRowDType DType = ENPCRowDType::Float;
    int32 Offset = 0;
    int32 Width = 0;
};

struct SCHOLAINFERENCEUTILS_API FNPCRowLayout
{
    TArray<FNPCRowElement> Inputs, Outputs;
    int32 ObsSize = 0, ActSize = 0;
    uint32 Hash = 0;
    void UpdateHash();
};

/** Immutable codec. Build once, then share between callers. Discrete values are raw indices. */
class SCHOLAINFERENCEUTILS_API FNPCRowCodec
{
public:
    static TOptional<FNPCRowCodec> Create(const FInteractionDefinition& Definition,
                                         const FNPCRowLayout& Layout, FString& OutError);
    bool Encode(const TInstancedStruct<FPoint>& Observation, TArrayView<float> Row) const;
    /** Allocates on first use only; malformed rows leave the action untouched. */
    bool Decode(TConstArrayView<float> Row, TInstancedStruct<FPoint>& InOutAction) const;

private:
    struct FBinding
    {
        FString Key;
        ESpaceType Type = ESpaceType::Box;
        int32 Offset = 0, Width = 0;
        TArray<int32> High;
        TArray<int32> Shape;
    };
    static bool Build(const TInstancedStruct<FSpace>& Space, const TArray<FNPCRowElement>& Elements,
                      int32 RowSize, TArray<FBinding>& Bindings, bool& bDict, FString& Error);
    static bool ValidateValues(const FBinding& Binding, TConstArrayView<float> Values);
    TArray<FBinding> Inputs, Outputs;
    int32 ObsSize = 0, ActSize = 0;
    bool bInputDict = false, bOutputDict = false;
};
