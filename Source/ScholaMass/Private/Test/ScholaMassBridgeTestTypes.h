#pragma once
#include "Mass/EntityFragments.h"

#include "ScholaMassInferenceProcessors.h"
#include "Inference/NPCBackgroundBackendProvider.h"
#include "Inference/DeterministicBatchBackend.h"

#include "ScholaMassBridgeTestTypes.generated.h"
USTRUCT() struct FScholaMassMigrationTestTag : public FMassTag { GENERATED_BODY() };

UCLASS()
class UScholaMassTestRowProvider : public UNPCBackgroundBackendProvider
{
    GENERATED_BODY()
public:
    virtual TSharedPtr<IBatchInferenceBackend> CreateBackend(const FInteractionDefinition&,FNPCRowLayout& Layout,FString& Error) override
    {
        Layout={};Layout.ObsSize=Layout.ActSize=1;
        Layout.Inputs=Layout.Outputs={{TEXT("value"),ENPCRowDType::Float,0,1}};Layout.UpdateHash();Error.Reset();
        FDeterministicBatchBackendSettings S;S.ObservationSize=S.ActionSize=1;
        return MakeShared<FDeterministicBatchBackend>(S);
    }
};
UCLASS()
class UScholaMassTestGather : public UScholaMassGatherProcessor
{
    GENERATED_BODY()
public:
    UScholaMassTestGather() { bAutoRegisterWithProcessingPhases=false; }
protected:
    virtual void WriteObservations(FMassExecutionContext& Context,TConstArrayView<int32> Indices,TArrayView<float> Rows) override;
};
