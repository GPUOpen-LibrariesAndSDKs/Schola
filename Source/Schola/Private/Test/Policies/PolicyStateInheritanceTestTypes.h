// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "UObject/Object.h"
#include "Policies/BlueprintPolicy.h"
#include "Policies/PolicyStateInterface.h"
#include "Test/Policies/ChatHistoryTestPolicy.h"
#include "PolicyStateInheritanceTestTypes.generated.h"

/** Unrelated interface, listed before IPolicyState so the IPolicyState subobject sits at a non-zero offset. */
UINTERFACE(meta = (CannotImplementInterfaceInBlueprint))
class UTestUnrelatedInterface : public UInterface
{
	GENERATED_BODY()
};

class ITestUnrelatedInterface
{
	GENERATED_BODY()

public:
	virtual int32 GetMarker() const { return 0; }
};

/** State implementing two interfaces, with IPolicyState as the second base. */
UCLASS()
class UTestMultiInterfaceState : public UObject, public ITestUnrelatedInterface, public IPolicyState
{
	GENERATED_BODY()

public:
	UPROPERTY()
	int32 Value = 0;

	int32 GetMarker() const override { return 42; }

	void Reset() override { Value = 0; }

	bool CopyFrom(const IPolicyState& Other) override
	{
		const UTestMultiInterfaceState* OtherState = Cast<UTestMultiInterfaceState>(Other._getUObject());
		if (!OtherState)
		{
			return false;
		}
		Value = OtherState->Value;
		return true;
	}
};

/** Sub-interface of IPolicyState, e.g. for states that also expose a sequence length. */
UINTERFACE(meta = (CannotImplementInterfaceInBlueprint))
class UTestRecurrentPolicyState : public UPolicyState
{
	GENERATED_BODY()
};

class ITestRecurrentPolicyState : public IPolicyState
{
	GENERATED_BODY()

public:
	virtual int32 GetSequenceLength() const PURE_VIRTUAL(ITestRecurrentPolicyState::GetSequenceLength, return 0;);
};

/** State implementing only the sub-interface. */
UCLASS()
class UTestRecurrentState : public UObject, public ITestRecurrentPolicyState
{
	GENERATED_BODY()

public:
	UPROPERTY()
	TArray<float> History;

	int32 GetSequenceLength() const override { return History.Num(); }

	void Reset() override { History.Reset(); }

	bool CopyFrom(const IPolicyState& Other) override
	{
		const UTestRecurrentState* OtherState = Cast<UTestRecurrentState>(Other._getUObject());
		if (!OtherState)
		{
			return false;
		}
		History = OtherState->History;
		return true;
	}
};

/** State subclass of another state class, overriding a virtual. */
UCLASS()
class UTestDerivedChatHistoryState : public UTestChatHistoryState
{
	GENERATED_BODY()

public:
	UPROPERTY()
	int32 ResetCount = 0;

	void Reset() override
	{
		Super::Reset();
		ResetCount++;
	}
};

/** Concrete UBlueprintPolicy, so CreateInitialState goes through the reflected BlueprintNativeEvent. */
UCLASS()
class UTestReflectedStatePolicy : public UBlueprintPolicy
{
	GENERATED_BODY()

public:
	TScriptInterface<IPolicyState> CreateInitialState_Implementation(UObject* InOuter) const override
	{
		return NewObject<UTestDerivedChatHistoryState>(InOuter);
	}
};
