// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "IREEModelData.h"
#include "IREEUtils/IREETensorBinding.h"

#ifndef SCHOLA_IREE_HAS_RUNTIME
#define SCHOLA_IREE_HAS_RUNTIME 0
#endif

/**
 * @brief RAII wrapper around the IREE embedded C runtime for a single loaded module.
 *
 * When ``SCHOLA_IREE_HAS_RUNTIME`` is 0 (ThirdParty IREE libs not populated),
 * ``Initialize`` always fails with a logged error and ``Invoke`` is a no-op.
 */
class SCHOLAIREE_API FIREERuntimeSession
{
public:
	FIREERuntimeSession();
	~FIREERuntimeSession();

	FIREERuntimeSession(const FIREERuntimeSession&) = delete;
	FIREERuntimeSession& operator=(const FIREERuntimeSession&) = delete;

	/**
	 * Reads the first ``iree.reflection`` value stored under @p Key on any exported function of the module.
	 * Loads the module in a temporary ``local-sync`` session, so it is safe to call at import time.
	 * ``VmfbBytes`` only needs to outlive the call.
	 */
	static bool ReadReflectionAttribute(TConstArrayView<uint8> VmfbBytes, const FString& Key, FString& OutValue);

	bool Initialize(const FString& DeviceUri, TConstArrayView<uint8> VmfbBytes);
	bool IsReady() const;

	bool Invoke(
		const FString& EntryFunction,
		TConstArrayView<FIREETensorBinding> InputBindings,
		TConstArrayView<FScholaIREETensorInfo> InputDescs,
		TArrayView<FIREETensorBinding> OutputBindings,
		TConstArrayView<FScholaIREETensorInfo> OutputDescs);

private:
#if SCHOLA_IREE_HAS_RUNTIME
	/** Device buffer reused for one input tensor across calls, so Invoke only uploads data into it. */
	struct FInputSlot
	{
		struct iree_hal_buffer_view_t* View = nullptr;
		uint64 SizeInBytes = 0;
	};

	bool PrepareInput(int32 Index, const FIREETensorBinding& Binding, const FScholaIREETensorInfo& Desc);
	void ReleaseInputSlots();

	struct iree_runtime_instance_t* Instance = nullptr;
	struct iree_runtime_session_t* Session = nullptr;
	TArray<FInputSlot> InputSlots;
#else
	void* Instance = nullptr;
	void* Session = nullptr;
#endif
	FString LoadedDeviceUri;
	TArray<uint8> OwnedVmfbBytes;
	bool bReady = false;
};
