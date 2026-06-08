// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Points/TextPoint.h"

void FTextPoint::Accept(ConstPointVisitor& Visitor) const
{
	Visitor(*this);
}

void FTextPoint::Accept(PointVisitor& Visitor)
{
	Visitor(*this);
}
