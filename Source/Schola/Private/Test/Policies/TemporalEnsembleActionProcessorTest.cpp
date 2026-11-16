// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Policies/TemporalEnsembleActionProcessor.h"

#if WITH_AUTOMATION_TESTS

namespace
{
	/** Processes Actions with CoeffOutput as the ensemble's coefficient output. */
	bool ProcessWithCoeffOutput(FActionProcessor& Processor, TConstArrayView<float> Actions, TConstArrayView<float> CoeffOutput, TArrayView<float> OutAction)
	{
		const TConstArrayView<float> ExtraOutputs[] = { CoeffOutput };
		return Processor.Process({ Actions, ExtraOutputs }, OutAction);
	}

	/** Processes Actions with the temporal ensemble coefficient Coeff. */
	bool ProcessWithCoeff(FActionProcessor& Processor, TConstArrayView<float> Actions, float Coeff, TArrayView<float> OutAction)
	{
		return ProcessWithCoeffOutput(Processor, Actions, MakeArrayView(&Coeff, 1), OutAction);
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTemporalEnsembleActionProcessorAverageTest, "Schola.Policies.ActionProcessor.Temporal Ensemble.Average", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FTemporalEnsembleActionProcessorAverageTest::RunTest(const FString& Parameters)
{
	FTemporalEnsembleActionProcessor Processor;
	TestTrue(TEXT("Init succeeds"), Processor.Init(3, 1));
	TestTrue(TEXT("Ensemble always needs a chunk"), Processor.NeedsNewChunk());

	const TArray<float> First = { 1.0f, 10.0f, 100.0f };
	const TArray<float> Second = { 2.0f, 20.0f, 200.0f };
	const TArray<float> Third = { 3.0f, 30.0f, 300.0f };
	TArray<float> Action;
	Action.SetNum(1);
	TestTrue(TEXT("First Process succeeds"), ProcessWithCoeff(Processor, First, 0.0f, Action));
	TestEqual(TEXT("First step is the only prediction"), Action[0], 1.0f);

	TestTrue(TEXT("Second Process succeeds"), ProcessWithCoeff(Processor, Second, 0.0f, Action));
	TestEqual(TEXT("Overlapping predictions are averaged"), Action[0], 6.0f, 0.0001f);

	TestTrue(TEXT("Third Process succeeds"), ProcessWithCoeff(Processor, Third, 0.0f, Action));
	TestEqual(TEXT("Three predictions are averaged"), Action[0], (100.0f + 20.0f + 3.0f) / 3.0f, 0.0001f);
	TestTrue(TEXT("Ensemble still needs a chunk"), Processor.NeedsNewChunk());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTemporalEnsembleActionProcessorWeightsTest, "Schola.Policies.ActionProcessor.Temporal Ensemble.Weights", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FTemporalEnsembleActionProcessorWeightsTest::RunTest(const FString& Parameters)
{
	FTemporalEnsembleActionProcessor Processor;
	Processor.Init(2, 1);

	const TArray<float> First = { 1.0f, 10.0f };
	const TArray<float> Second = { 2.0f, 20.0f };
	TArray<float> Action;
	Action.SetNum(1);
	const float HalvingCoeff = FMath::Loge(2.0f);
	ProcessWithCoeff(Processor, First, HalvingCoeff, Action);
	ProcessWithCoeff(Processor, Second, HalvingCoeff, Action);
	TestEqual(TEXT("Oldest prediction is weighted 1, the next exp(-k)"), Action[0], (10.0f + 0.5f * 2.0f) / 1.5f, 0.0001f);

	AddExpectedMessage(TEXT("Got 2 floats in temporal_ensemble_coeff, expected one"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("Rejects more than one coefficient"), ProcessWithCoeffOutput(Processor, First, TArray<float>({ 0.1f, 0.2f }), Action));
	AddExpectedMessage(TEXT("Got 0 floats in temporal_ensemble_coeff"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("Rejects a chunk without a coefficient"), Processor.Process({ First }, Action));
	TestEqual(TEXT("Rejected coefficients leave the ensemble unchanged"), Processor.PredictionCounts[0], 2);

	TestTrue(TEXT("A negative coefficient with finite weights is accepted"), ProcessWithCoeff(Processor, First, -1.0f, Action));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTemporalEnsembleActionProcessorExtraOutputsTest, "Schola.Policies.ActionProcessor.Temporal Ensemble.Extra Outputs", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FTemporalEnsembleActionProcessorExtraOutputsTest::RunTest(const FString& Parameters)
{
	FTemporalEnsembleActionProcessor Processor;
	TestEqual(TEXT("Reads the coefficient output"), Processor.GetExtraOutputNames(), TArray<FString>({ TEXT("temporal_ensemble_coeff") }));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTemporalEnsembleActionProcessorResetAndFailureTest, "Schola.Policies.ActionProcessor.Temporal Ensemble.Reset And Failure", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FTemporalEnsembleActionProcessorResetAndFailureTest::RunTest(const FString& Parameters)
{
	FTemporalEnsembleActionProcessor Processor;
	Processor.Init(2, 1);

	const TArray<float> First = { 4.0f, 8.0f };
	TArray<float> Action;
	Action.SetNum(1);
	ProcessWithCoeff(Processor, First, 0.0f, Action);

	AddExpectedMessage(TEXT("Chunk has 0 floats"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("Process without a chunk fails"), Processor.Process({}, Action));
	TestEqual(TEXT("Failed Process leaves the ensemble unchanged"), Processor.PredictionCounts[0], 1);

	Processor.Reset();
	TestEqual(TEXT("Reset zeroes the sums"), Processor.WeightedSums[1], 0.0f);
	TestEqual(TEXT("Reset zeroes the counts"), Processor.PredictionCounts[1], 0);
	const TArray<float> Second = { 5.0f, 6.0f };
	TestTrue(TEXT("Process after Reset succeeds"), ProcessWithCoeff(Processor, Second, 0.0f, Action));
	TestEqual(TEXT("Reset forgets earlier predictions"), Action[0], 5.0f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTemporalEnsembleActionProcessorCopyInPlaceTest, "Schola.Policies.ActionProcessor.Temporal Ensemble.CopyInPlace", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FTemporalEnsembleActionProcessorCopyInPlaceTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FActionProcessor> Source = TInstancedStruct<FActionProcessor>::Make<FTemporalEnsembleActionProcessor>();
	TInstancedStruct<FActionProcessor> Target = TInstancedStruct<FActionProcessor>::Make<FTemporalEnsembleActionProcessor>();
	Source.GetMutable().Init(2, 1);
	Target.GetMutable().Init(2, 1);

	const TArray<float> Chunk = { 4.0f, 8.0f };
	TArray<float> Action;
	Action.SetNum(1);
	ProcessWithCoeff(Source.GetMutable(), Chunk, 0.0f, Action);

	const float* TargetSums = Target.Get<FTemporalEnsembleActionProcessor>().WeightedSums.GetData();
	TestTrue(TEXT("CopyInPlace succeeds"), FActionProcessor::CopyInPlace(Target, Source));
	TestEqual(TEXT("Sums copied"), Target.Get<FTemporalEnsembleActionProcessor>().WeightedSums[1], 8.0f);
	TestEqual(TEXT("Counts copied"), Target.Get<FTemporalEnsembleActionProcessor>().PredictionCounts[1], 1);
	TestTrue(TEXT("Sums are copied in place"), Target.Get<FTemporalEnsembleActionProcessor>().WeightedSums.GetData() == TargetSums);
	return true;
}

#endif // WITH_AUTOMATION_TESTS
