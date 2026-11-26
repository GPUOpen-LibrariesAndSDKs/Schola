// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "IREEUtils/IREEBuffer.h"
#include "IREEUtils/IREEBufferVisitor.h"

void FIREEDictBuffer::Accept(FIREEBufferVisitor& Visitor) const
{
	Visitor(*this);
}

void FIREEDiscreteBuffer::Accept(FIREEBufferVisitor& Visitor) const
{
	Visitor(*this);
}

void FIREEMultiBinaryBuffer::Accept(FIREEBufferVisitor& Visitor) const
{
	Visitor(*this);
}

void FIREEBoxBuffer::Accept(FIREEBufferVisitor& Visitor) const
{
	Visitor(*this);
}

void FIREEMultiDiscreteBuffer::Accept(FIREEBufferVisitor& Visitor) const
{
	Visitor(*this);
}
