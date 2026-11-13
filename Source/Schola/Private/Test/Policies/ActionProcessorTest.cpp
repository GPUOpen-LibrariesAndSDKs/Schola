// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Policies/ActionProcessor.h"
#include "Policies/ACTActionProcessor.h"
#include "Policies/TemporalEnsembleActionProcessor.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FActionProcessorInitTest, "Schola.Policies.ActionProcessor.Init", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FActionProcessorInitTest::RunTest(const FString& Parameters)
{
	FActionProcessor Processor;
	TestTrue(TEXT("Init accepts a valid layout"), Processor.Init(3, 2));
	TestEqual(TEXT("Chunk size set"), Processor.ChunkSize, 3);
	TestEqual(TEXT("Action size set"), Processor.ActionDim, 2);

	AddExpectedMessage(TEXT("both must be positive"), EAutomationExpectedMessageFlags::Contains, 2);
	TestFalse(TEXT("Init rejects an empty chunk"), Processor.Init(0, 2));
	TestFalse(TEXT("Init rejects an empty action"), Processor.Init(3, 0));
	TestEqual(TEXT("Rejected Init keeps the previous layout"), Processor.ChunkSize, 3);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FActionProcessorProcessTest, "Schola.Policies.ActionProcessor.Process", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FActionProcessorProcessTest::RunTest(const FString& Parameters)
{
	FActionProcessor Processor;
	TArray<float> Action;
	Action.SetNum(2);
	const TArray<float> Chunk = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

	AddExpectedMessage(TEXT("Call Init before Process"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("Process fails before Init"), Processor.Process({ Chunk }, Action));

	Processor.Init(3, 2);
	TestTrue(TEXT("Base processor always needs a new chunk"), Processor.NeedsNewChunk());
	TestTrue(TEXT("Process succeeds"), Processor.Process({ Chunk }, Action));
	TestEqual(TEXT("Plays the first action dim 0"), Action[0], 1.0f);
	TestEqual(TEXT("Plays the first action dim 1"), Action[1], 2.0f);
	TestTrue(TEXT("Still needs a new chunk after Process"), Processor.NeedsNewChunk());

	AddExpectedMessage(TEXT("Chunk has 0 floats, expected 6"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("Process fails without a chunk"), Processor.Process({}, Action));

	TArray<float> WrongAction;
	WrongAction.SetNum(3);
	AddExpectedMessage(TEXT("OutAction has 3 floats"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("Process fails for an action of the wrong size"), Processor.Process({ Chunk }, WrongAction));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FActionProcessorCopyInPlaceTest, "Schola.Policies.ActionProcessor.CopyInPlace", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FActionProcessorCopyInPlaceTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FActionProcessor> Source = TInstancedStruct<FActionProcessor>::Make();
	TInstancedStruct<FActionProcessor> Target = TInstancedStruct<FActionProcessor>::Make();
	TInstancedStruct<FActionProcessor> Empty;
	Source.GetMutable().Init(3, 2);
	Target.GetMutable().Init(3, 2);

	TestTrue(TEXT("Empty processors have the same layout"), FActionProcessor::IsSameLayout(Empty, TInstancedStruct<FActionProcessor>()));
	TestFalse(TEXT("An empty processor does not match a set one"), FActionProcessor::IsSameLayout(Empty, Source));
	TestTrue(TEXT("Same type and layout match"), FActionProcessor::IsSameLayout(Target, Source));

	const FActionProcessor* TargetPtr = Target.GetPtr();
	TestTrue(TEXT("CopyInPlace succeeds for the same layout"), FActionProcessor::CopyInPlace(Target, Source));
	TestTrue(TEXT("CopyInPlace keeps the target's memory"), Target.GetPtr() == TargetPtr);

	TInstancedStruct<FActionProcessor> Other = TInstancedStruct<FActionProcessor>::Make();
	Other.GetMutable().Init(4, 2);
	TestFalse(TEXT("A different chunk size does not match"), FActionProcessor::IsSameLayout(Target, Other));
	TestFalse(TEXT("CopyInPlace rejects a different layout"), FActionProcessor::CopyInPlace(Target, Other));
	TestEqual(TEXT("Rejected CopyInPlace leaves the target unchanged"), Target.Get().ChunkSize, 3);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FActionProcessorCreateForModelTest, "Schola.Policies.ActionProcessor.CreateForModel", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FActionProcessorCreateForModelTest::RunTest(const FString& Parameters)
{
	const TArray<FString> ACT = { TEXT("action"), TEXT("action_processor.ACTActionProcessor") };
	TestTrue(TEXT("The marker names the ACT processor"), FActionProcessor::CreateForModel(ACT).GetScriptStruct() == FACTActionProcessor::StaticStruct());
	const TArray<FString> Ensemble = { TEXT("action"), TEXT("temporal_ensemble_coeff"), TEXT("action_processor.TemporalEnsembleActionProcessor") };
	TestTrue(TEXT("The marker names the temporal ensemble"),
		FActionProcessor::CreateForModel(Ensemble).GetScriptStruct() == FTemporalEnsembleActionProcessor::StaticStruct());

	AddExpectedMessage(TEXT("needs exactly one output named action_processor.<processor>"), EAutomationExpectedMessageFlags::Contains, 2);
	const TArray<FString> NoMarker = { TEXT("action") };
	TestFalse(TEXT("A model without a marker has no processor"), FActionProcessor::CreateForModel(NoMarker).IsValid());
	const TArray<FString> TwoMarkers = { TEXT("action"), TEXT("action_processor.ACTActionProcessor"), TEXT("action_processor.TemporalEnsembleActionProcessor") };
	TestFalse(TEXT("A model with two markers has no processor"), FActionProcessor::CreateForModel(TwoMarkers).IsValid());

	AddExpectedMessage(TEXT("which is not a struct derived from FActionProcessor"), EAutomationExpectedMessageFlags::Contains, 3);
	const TArray<FString> Unknown = { TEXT("action"), TEXT("action_processor.MissingActionProcessor") };
	TestFalse(TEXT("A marker naming no struct has no processor"), FActionProcessor::CreateForModel(Unknown).IsValid());
	const TArray<FString> NotAProcessor = { TEXT("action"), TEXT("action_processor.Vector") };
	TestFalse(TEXT("A marker naming another struct has no processor"), FActionProcessor::CreateForModel(NotAProcessor).IsValid());
	const TArray<FString> WrongCase = { TEXT("action"), TEXT("action_processor.actactionprocessor") };
	TestFalse(TEXT("Struct names are case sensitive"), FActionProcessor::CreateForModel(WrongCase).IsValid());
	return true;
}

#endif // WITH_AUTOMATION_TESTS
