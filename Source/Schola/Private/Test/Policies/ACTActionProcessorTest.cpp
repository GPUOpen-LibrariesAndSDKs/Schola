// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Policies/ACTActionProcessor.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FACTActionProcessorWholeChunkTest, "Schola.Policies.ActionProcessor.ACT.Whole Chunk", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FACTActionProcessorWholeChunkTest::RunTest(const FString& Parameters)
{
	FACTActionProcessor Processor;
	TestTrue(TEXT("Init succeeds"), Processor.Init(3, 2));
	TestTrue(TEXT("New processor needs a chunk"), Processor.NeedsNewChunk());

	const TArray<float> Chunk = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	TArray<float> Action;
	Action.SetNum(2);

	TestTrue(TEXT("First Process succeeds"), Processor.Process({ Chunk }, Action));
	TestEqual(TEXT("Action 0 dim 0"), Action[0], 1.0f);
	TestEqual(TEXT("Action 0 dim 1"), Action[1], 2.0f);
	TestFalse(TEXT("Current chunk still has actions"), Processor.NeedsNewChunk());

	TestTrue(TEXT("Second Process works without a chunk"), Processor.Process({}, Action));
	TestEqual(TEXT("Action 1 dim 0"), Action[0], 3.0f);

	const TArray<float> IgnoredChunk = { 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f };
	TestTrue(TEXT("Third Process ignores a chunk it does not need"), Processor.Process({ IgnoredChunk }, Action));
	TestEqual(TEXT("Action 2 dim 0"), Action[0], 5.0f);
	TestTrue(TEXT("Current chunk is used up"), Processor.NeedsNewChunk());

	AddExpectedMessage(TEXT("Chunk has 0 floats"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("Process without a chunk fails once the current chunk is used up"), Processor.Process({}, Action));
	TestTrue(TEXT("Failed Process leaves the current chunk used up"), Processor.NeedsNewChunk());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FACTActionProcessorRefillAndResetTest, "Schola.Policies.ActionProcessor.ACT.Refill And Reset", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FACTActionProcessorRefillAndResetTest::RunTest(const FString& Parameters)
{
	FACTActionProcessor Processor;
	Processor.Init(2, 1);

	TArray<float> Action;
	Action.SetNum(1);
	const TArray<float> First = { 1.0f, 2.0f };
	Processor.Process({ First }, Action);
	Processor.Process({}, Action);
	TestEqual(TEXT("Second action played"), Action[0], 2.0f);
	TestTrue(TEXT("Needs a new chunk after the whole chunk"), Processor.NeedsNewChunk());

	const TArray<float> Second = { 4.0f, 5.0f };
	TestTrue(TEXT("Refill succeeds"), Processor.Process({ Second }, Action));
	TestEqual(TEXT("Refill plays the new chunk from the start"), Action[0], 4.0f);

	Processor.Reset();
	TestTrue(TEXT("Reset needs a new chunk"), Processor.NeedsNewChunk());
	TestEqual(TEXT("Reset zeroes the current chunk"), Processor.CurrentChunk[0], 0.0f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FACTActionProcessorCopyInPlaceTest, "Schola.Policies.ActionProcessor.ACT.CopyInPlace", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FACTActionProcessorCopyInPlaceTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FActionProcessor> Source = TInstancedStruct<FActionProcessor>::Make<FACTActionProcessor>();
	TInstancedStruct<FActionProcessor> Target = TInstancedStruct<FActionProcessor>::Make<FACTActionProcessor>();
	Source.GetMutable().Init(2, 1);
	Target.GetMutable().Init(2, 1);

	TArray<float> Action;
	Action.SetNum(1);
	const TArray<float> Chunk = { 4.0f, 8.0f };
	Source.GetMutable().Process({ Chunk }, Action);

	const float* TargetChunk = Target.Get<FACTActionProcessor>().CurrentChunk.GetData();
	TestTrue(TEXT("CopyInPlace succeeds"), FActionProcessor::CopyInPlace(Target, Source));
	TestEqual(TEXT("Current chunk copied"), Target.Get<FACTActionProcessor>().CurrentChunk[1], 8.0f);
	TestEqual(TEXT("Chunk position copied"), Target.Get<FACTActionProcessor>().NextIndex, 1);
	TestTrue(TEXT("Current chunk is copied in place"), Target.Get<FACTActionProcessor>().CurrentChunk.GetData() == TargetChunk);

	TestTrue(TEXT("Copy plays on from the source's position"), Target.GetMutable().Process({}, Action));
	TestEqual(TEXT("Copy plays the second action"), Action[0], 8.0f);

	TInstancedStruct<FActionProcessor> Base = TInstancedStruct<FActionProcessor>::Make();
	Base.GetMutable().Init(2, 1);
	TestFalse(TEXT("A different processor type does not match"), FActionProcessor::IsSameLayout(Target, Base));
	return true;
}

#endif // WITH_AUTOMATION_TESTS
