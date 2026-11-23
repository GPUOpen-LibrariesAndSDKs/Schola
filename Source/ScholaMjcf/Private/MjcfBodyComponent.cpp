// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "MjcfBodyComponent.h"

UMjcfBodyComponent::UMjcfBodyComponent()
{
	SetCollisionEnabled(ECollisionEnabled::NoCollision);
	SetGenerateOverlapEvents(false);
	SetCanEverAffectNavigation(false);
}
