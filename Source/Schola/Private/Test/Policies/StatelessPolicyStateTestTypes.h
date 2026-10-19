// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Policies/BlueprintPolicy.h"
#include "StatelessPolicyStateTestTypes.generated.h"

/** Concrete UBlueprintPolicy with no overrides, so ReceiveCreateInitialState runs the default through the reflected BlueprintNativeEvent. */
UCLASS()
class UTestDefaultBlueprintPolicy : public UBlueprintPolicy
{
	GENERATED_BODY()
};

/** UBlueprintPolicy whose ReceiveCreateInitialState returns None, like a Blueprint override that forgets to set its return value. */
UCLASS()
class UTestNoneStateBlueprintPolicy : public UBlueprintPolicy
{
	GENERATED_BODY()

protected:
	TScriptInterface<IPolicyState> ReceiveCreateInitialState_Implementation() const override
	{
		return nullptr;
	}
};
