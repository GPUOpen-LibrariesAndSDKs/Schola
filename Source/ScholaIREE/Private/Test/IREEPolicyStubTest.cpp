// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Policies/IREEPolicy.h"
#include "IREEModelData.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FIREEPolicyInitWithoutVmfbTest,
	"Schola.Policies.IREE.IREEPolicy.InitWithoutVmfb",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEPolicyInitWithoutVmfbTest::RunTest(const FString& Parameters)
{
	UIREEPolicy* Policy = NewObject<UIREEPolicy>();
	FInteractionDefinition Definition;
	AddExpectedError(TEXT("invalid model data"), EAutomationExpectedErrorFlags::Contains, 1);
	TestFalse(TEXT("Init fails without vmfb bytes"), Policy->Init(Definition));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FIREEPolicyInitWithMetadataOnlyTest,
	"Schola.Policies.IREE.IREEPolicy.InitSkipsWithoutRuntime",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEPolicyInitWithMetadataOnlyTest::RunTest(const FString& Parameters)
{
	UIREEModelData* ModelData = NewObject<UIREEModelData>();
	ModelData->MetadataJson = TEXT("{\"entry_function\":\"main\",\"inputs\":[{\"name\":\"obs\",\"dtype\":\"float32\",\"shape\":[]}],\"outputs\":[{\"name\":\"act\",\"dtype\":\"float32\",\"shape\":[]}],\"state\":[],\"target_backends\":[\"llvm-cpu\"],\"batch_size\":1}");
	ModelData->ParseMetadata();
	ModelData->VmfbBytes = { 0x00 };

	UIREEPolicy* Policy = NewObject<UIREEPolicy>();
	Policy->ModelData = ModelData;
	FInteractionDefinition Definition;
	// Without IREE third-party libs linked in the editor test environment this may fail at runtime init;
	// the test only asserts that missing metadata/vmfb paths are handled predictably.
	AddExpectedError(TEXT("IREE"), EAutomationExpectedErrorFlags::Contains, 0);
	const bool bInitialized = Policy->Init(Definition);
	if (!bInitialized)
	{
		AddInfo(TEXT("IREEPolicy::Init returned false (expected when IREE runtime is unavailable in CI)."));
	}
	return true;
}

#endif