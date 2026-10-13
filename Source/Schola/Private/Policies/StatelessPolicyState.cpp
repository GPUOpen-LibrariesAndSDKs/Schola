// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Policies/StatelessPolicyState.h"
#include "UObject/Package.h"

bool UStatelessPolicyState::CopyFrom(const IPolicyState& Other)
{
	return Other._getUObject()->IsA<UStatelessPolicyState>();
}

TScriptInterface<IPolicyState> UStatelessPolicyState::Get()
{
	check(IsInGameThread());
	static UStatelessPolicyState* Instance = nullptr;
	if (!Instance)
	{
		Instance = NewObject<UStatelessPolicyState>(GetTransientPackage(), TEXT("SharedStatelessPolicyState"));
		Instance->AddToRoot();
	}
	return Instance;
}
