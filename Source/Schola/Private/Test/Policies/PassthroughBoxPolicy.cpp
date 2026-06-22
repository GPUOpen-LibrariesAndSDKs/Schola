// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Test/Policies/PassthroughBoxPolicy.h"

#include "Points/BoxPoint.h"

bool UPassthroughBoxPolicy::Think(const TInstancedStruct<FPoint>& InObservations, TInstancedStruct<FPoint>& OutAction)
{
	const FBoxPoint* Src = InObservations.GetPtr<FBoxPoint>();
	if (!Src)
	{
		return false;
	}
	OutAction.InitializeAs<FBoxPoint>(Src->Values, Src->Shape);
	return true;
}

bool UPassthroughBoxPolicy::Init(const FInteractionDefinition& InPolicyDefinition)
{
	(void)InPolicyDefinition;
	return true;
}

bool UPassthroughBoxPolicy::IsInferenceBusy() const
{
	return false;
}
