// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Policies/ActionChunker.h"

#if WITH_AUTOMATION_TESTS

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

namespace ScholaActionChunkerTestPrivate
{
	static FActionChunkerSettings MakeEnsembleSettings(float Coeff)
	{
		FActionChunkerSettings Settings;
		Settings.ChunkMode = EActionChunkMode::TemporalEnsemble;
		Settings.TemporalEnsembleCoeff = Coeff;
		return Settings;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FActionChunkerQueueConsumeTest, "Schola.Policies.ActionChunker.Queue Consume", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FActionChunkerQueueConsumeTest::RunTest(const FString& Parameters)
{
	FActionChunker Chunker;
	Chunker.Init(3, 2);
	TestTrue(TEXT("Empty queue needs a chunk"), Chunker.NeedsNewChunk());

	const TArray<float> Chunk = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	TArray<float> Action;
	Action.SetNum(2);

	TestTrue(TEXT("First consume succeeds"), Chunker.Consume(Chunk, Action));
	TestEqualExactFloat(TEXT("Queue action 0 dim 0"), Action[0], 1.0f);
	TestEqualExactFloat(TEXT("Queue action 0 dim 1"), Action[1], 2.0f);
	TestFalse(TEXT("Queue still has actions"), Chunker.NeedsNewChunk());

	TestTrue(TEXT("Second consume ignores a missing chunk"), Chunker.Consume({}, Action));
	TestEqualExactFloat(TEXT("Queue action 1 dim 0"), Action[0], 3.0f);
	TestEqualExactFloat(TEXT("Queue action 1 dim 1"), Action[1], 4.0f);

	TestTrue(TEXT("Third consume succeeds"), Chunker.Consume({}, Action));
	TestEqualExactFloat(TEXT("Queue action 2 dim 0"), Action[0], 5.0f);
	TestTrue(TEXT("Queue is exhausted"), Chunker.NeedsNewChunk());

	AddExpectedMessage(TEXT("Queue is empty"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("Consume without a chunk fails when empty"), Chunker.Consume({}, Action));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FActionChunkerEnsembleConsumeTest, "Schola.Policies.ActionChunker.Temporal Ensemble Consume", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FActionChunkerEnsembleConsumeTest::RunTest(const FString& Parameters)
{
	FActionChunker Chunker;
	Chunker.Init(3, 1, ScholaActionChunkerTestPrivate::MakeEnsembleSettings(0.0f));
	TestTrue(TEXT("Ensemble always needs a chunk"), Chunker.NeedsNewChunk());

	const TArray<float> FirstChunk = { 1.0f, 10.0f, 100.0f };
	const TArray<float> SecondChunk = { 2.0f, 20.0f, 200.0f };
	TArray<float> Action;
	Action.SetNum(1);

	TestTrue(TEXT("First ensemble consume succeeds"), Chunker.Consume(FirstChunk, Action));
	TestEqualExactFloat(TEXT("First step is the newest prediction"), Action[0], 1.0f);

	TestTrue(TEXT("Second ensemble consume succeeds"), Chunker.Consume(SecondChunk, Action));
	TestEqualExactFloat(TEXT("Overlapping predictions are averaged"), Action[0], 6.0f);
	TestTrue(TEXT("Ensemble still needs a chunk"), Chunker.NeedsNewChunk());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FActionChunkerResetAndCopyFromTest, "Schola.Policies.ActionChunker.Reset And CopyFrom", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FActionChunkerResetAndCopyFromTest::RunTest(const FString& Parameters)
{
	FActionChunker Source;
	FActionChunker Target;
	Source.Init(2, 1, ScholaActionChunkerTestPrivate::MakeEnsembleSettings(0.0f));
	Target.Init(2, 1, ScholaActionChunkerTestPrivate::MakeEnsembleSettings(0.0f));

	TArray<float> Action;
	Action.SetNum(1);
	const TArray<float> Chunk = { 4.0f, 8.0f };
	TestTrue(TEXT("Seed ensemble"), Source.Consume(Chunk, Action));

	const float* TargetChunkData = Target.ChunkBuffer.GetData();
	TestTrue(TEXT("CopyFrom a matching chunker succeeds"), Target.CopyFrom(Source));
	TestEqualExactFloat(TEXT("Ensemble copied"), Target.ChunkBuffer[0], Source.ChunkBuffer[0]);
	TestTrue(TEXT("Chunk buffer is copied in place"), Target.ChunkBuffer.GetData() == TargetChunkData);

	FActionChunker QueueChunker;
	QueueChunker.Init(2, 1);
	TestFalse(TEXT("CopyFrom rejects a different mode"), Target.CopyFrom(QueueChunker));

	Source.Reset();
	TestTrue(TEXT("Reset empties the ensemble"), Source.NeedsNewChunk());
	TestEqualExactFloat(TEXT("Reset zeroes the ensemble"), Source.ChunkBuffer[0], 0.0f);
	return true;
}

#endif // WITH_AUTOMATION_TESTS
