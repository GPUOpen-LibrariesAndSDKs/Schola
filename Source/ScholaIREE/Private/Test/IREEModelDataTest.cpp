// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "IREEModelData.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FIREEModelMetadataParseTest,
	"Schola.Policies.IREE.IREEModelData.MetadataParse",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FIREEModelMetadataParseTest::RunTest(const FString& Parameters)
{
	UIREEModelData* ModelData = NewObject<UIREEModelData>();
	ModelData->MetadataJson = TEXT(R"({
		"entry_function": "main",
		"inputs": [
			{"name": "obs", "dtype": "float32", "shape": [1, 4]}
		],
		"outputs": [
			{"name": "action", "dtype": "float32", "shape": [1, 2]}
		],
		"state": [
			{"name": "h", "dtype": "float32", "shape": [1, 1, 8],
			 "state_metadata": {"has_seq_dim": false, "max_seq_len": null, "seq_dim": null}}
		],
		"target_backends": ["llvm-cpu"],
		"iree_version": "test",
		"batch_size": 1
	})");

	TestTrue(TEXT("ParseMetadata succeeds"), ModelData->ParseMetadata());
	TestEqual(TEXT("entry_function"), ModelData->Metadata.EntryFunction, FString(TEXT("main")));
	TestEqual(TEXT("inputs count (observations only)"), ModelData->Metadata.Inputs.Num(), 1);
	TestEqual(TEXT("input name"), ModelData->Metadata.Inputs[0].Name, FString(TEXT("obs")));
	TestEqual(TEXT("input dtype"), ModelData->Metadata.Inputs[0].Dtype, FString(TEXT("float32")));
	TestEqual(TEXT("outputs count (actions only)"), ModelData->Metadata.Outputs.Num(), 1);
	TestEqual(TEXT("output name"), ModelData->Metadata.Outputs[0].Name, FString(TEXT("action")));
	if (TestEqual(TEXT("states count"), ModelData->Metadata.States.Num(), 1))
	{
		const FScholaIREEStateInfo& State = ModelData->Metadata.States[0];
		TestEqual(TEXT("state name"), State.Name, FString(TEXT("h")));
		TestEqual(TEXT("state in name"), State.GetInName(), FString(TEXT("state_in_h")));
		TestEqual(TEXT("state out name"), State.GetOutName(), FString(TEXT("state_out_h")));
		TestEqual(TEXT("state shape rank"), State.Shape.Num(), 3);
		TestFalse(TEXT("state has no seq dim"), State.StateMetadata.bHasSeqDim);
		TestEqual(TEXT("state max_seq_len null"), State.StateMetadata.MaxSeqLen, static_cast<int32>(INDEX_NONE));
	}
	const TArray<int32>& InputShape = ModelData->Metadata.Inputs[0].Shape;
	if (TestEqual(TEXT("input shape rank"), InputShape.Num(), 2))
	{
		TestEqual(TEXT("input shape dim0"), InputShape[0], 1);
		TestEqual(TEXT("input shape dim1"), InputShape[1], 4);
	}
	return true;
}

#endif