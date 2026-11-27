// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "IREEModelData.h"
#include "IREEUtils/IREETensorBinding.h"
#include "IREEUtils/IREEBufferVisitor.h"
#include "IREEUtils/IREEBuffer.h"
#include "StructUtils/InstancedStruct.h"
#include "Spaces/Space.h"

/**
 * @brief Visitor that maps IREE buffers to backend-neutral tensor bindings by tensor name.
 */
struct SCHOLAIREE_API FIREEBindingCreator : public FIREEBufferVisitor
{
	TInstancedStruct<FIREEPointBuffer>& Buffer;
	TArray<FIREETensorBinding>& Bindings;
	TConstArrayView<FScholaIREETensorInfo> TensorDescs;
	bool bError = false;
	int Index = 0;

	FIREEBindingCreator(
		TInstancedStruct<FIREEPointBuffer>& InBuffer,
		TConstArrayView<FScholaIREETensorInfo> InTensorDescs,
		TArray<FIREETensorBinding>& OutBindings)
		: Buffer(InBuffer), Bindings(OutBindings), TensorDescs(InTensorDescs)
	{
	}

	void operator()(const FIREEDictBuffer& InBuffer) override;
	void operator()(const FIREEBoxBuffer& InBuffer) override;
	void operator()(const FIREEMultiBinaryBuffer& InBuffer) override;
	void operator()(const FIREEDiscreteBuffer& InBuffer) override;
	void operator()(const FIREEMultiDiscreteBuffer& InBuffer) override;

	/** @return true if an error occurred, false on success */
	static bool CreateBindings(
		const TInstancedStruct<FSpace>& InSpace,
		TInstancedStruct<FIREEPointBuffer>& InBuffer,
		TConstArrayView<FScholaIREETensorInfo> InTensorDescs,
		TArray<FIREETensorBinding>& OutBindings);
};
