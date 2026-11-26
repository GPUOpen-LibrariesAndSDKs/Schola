// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "IREE/IREERuntimeSession.h"
#include "LogScholaIREE.h"

#if SCHOLA_IREE_HAS_RUNTIME
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/runtime/api.h"
#include "iree/vm/api.h"
#endif

FIREERuntimeSession::FIREERuntimeSession() = default;

FIREERuntimeSession::~FIREERuntimeSession()
{
#if SCHOLA_IREE_HAS_RUNTIME
	ReleaseInputSlots();
	if (Session)
	{
		iree_runtime_session_release(Session);
		Session = nullptr;
	}
	if (Instance)
	{
		iree_runtime_instance_release(Instance);
		Instance = nullptr;
	}
#endif
	bReady = false;
}

bool FIREERuntimeSession::IsReady() const
{
	return bReady;
}

#if !SCHOLA_IREE_HAS_RUNTIME

bool FIREERuntimeSession::ReadReflectionAttribute(TConstArrayView<uint8> VmfbBytes, const FString& Key, FString& OutValue)
{
	UE_LOGFMT(LogScholaIREE, Error, "FIREERuntimeSession::ReadReflectionAttribute: IREE runtime was not linked (SCHOLA_IREE_HAS_RUNTIME=0)");
	return false;
}

bool FIREERuntimeSession::Initialize(const FString& DeviceUri, TConstArrayView<uint8> VmfbBytes)
{
	UE_LOGFMT(
		LogScholaIREE,
		Error,
		"FIREERuntimeSession::Initialize: IREE runtime was not linked (SCHOLA_IREE_HAS_RUNTIME=0). "
		"Populate Source/ThirdParty/IREE/{{Platform}} with Resources/Build/iree/build_iree_windows.ps1 / build_iree_linux.sh and rebuild. Device={0} VMFBBytes={1}",
		DeviceUri,
		VmfbBytes.Num());
	return false;
}

bool FIREERuntimeSession::Invoke(
	const FString& EntryFunction,
	TConstArrayView<FIREETensorBinding> InputBindings,
	TConstArrayView<FScholaIREETensorInfo> InputDescs,
	TArrayView<FIREETensorBinding> OutputBindings,
	TConstArrayView<FScholaIREETensorInfo> OutputDescs)
{
	UE_LOGFMT(LogScholaIREE, Error, "FIREERuntimeSession::Invoke: IREE runtime not linked (entry={0})", EntryFunction);
	return false;
}

#else

namespace
{
	bool LogIreeStatus(const TCHAR* Context, iree_status_t Status)
	{
		if (iree_status_is_ok(Status))
		{
			return true;
		}
		iree_host_size_t MessageLength = 0;
		iree_status_format(Status, /*max_length=*/0, nullptr, &MessageLength);
		TArray<ANSICHAR> Message;
		Message.SetNumUninitialized(static_cast<int32>(MessageLength) + 1);
		iree_status_format(Status, MessageLength + 1, Message.GetData(), &MessageLength);
		Message[MessageLength] = '\0';
		const FString MessageString = UTF8_TO_TCHAR(Message.GetData());
		UE_LOGFMT(LogScholaIREE, Error, "{0}: {1}", Context, MessageString);
		iree_status_ignore(Status);
		return false;
	}

	iree_hal_element_type_t DtypeToElementType(const FString& Dtype)
	{
		if (Dtype.Equals(TEXT("float32"), ESearchCase::IgnoreCase))
		{
			return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
		}
		if (Dtype.Equals(TEXT("float64"), ESearchCase::IgnoreCase))
		{
			return IREE_HAL_ELEMENT_TYPE_FLOAT_64;
		}
		if (Dtype.Equals(TEXT("int64"), ESearchCase::IgnoreCase))
		{
			return IREE_HAL_ELEMENT_TYPE_INT_64;
		}
		if (Dtype.Equals(TEXT("int32"), ESearchCase::IgnoreCase))
		{
			return IREE_HAL_ELEMENT_TYPE_INT_32;
		}
		if (Dtype.Equals(TEXT("bool"), ESearchCase::IgnoreCase))
		{
			return IREE_HAL_ELEMENT_TYPE_BOOL_8;
		}
		return IREE_HAL_ELEMENT_TYPE_NONE;
	}

	bool CreateInputBufferView(
		iree_runtime_session_t* Session,
		iree_allocator_t HostAllocator,
		const FIREETensorBinding& Binding,
		const FScholaIREETensorInfo& Desc,
		iree_hal_buffer_view_t** OutView,
		uint64& OutSizeInBytes)
	{
		const iree_hal_element_type_t ElementType = DtypeToElementType(Desc.Dtype);
		if (ElementType == IREE_HAL_ELEMENT_TYPE_NONE)
		{
			UE_LOGFMT(LogScholaIREE, Error, "IREERuntimeSession: unsupported dtype {0} for {1}", Desc.Dtype, Desc.Name);
			return false;
		}

		TArray<iree_hal_dim_t, TInlineAllocator<8>> Dims;
		if (Desc.Shape.Num() > 0)
		{
			for (int32 Dim : Desc.Shape)
			{
				Dims.Add(static_cast<iree_hal_dim_t>(FMath::Max(Dim, 1)));
			}
		}
		else
		{
			// Fallback: treat as a flat 1-D byte-sized view when shape metadata is missing.
			Dims.Add(static_cast<iree_hal_dim_t>(Binding.SizeInBytes));
		}

		iree_device_size_t AllocationSize = 0;
		iree_status_t Status = iree_hal_buffer_compute_view_size(
			Dims.Num(), Dims.GetData(), ElementType, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &AllocationSize);
		if (!LogIreeStatus(TEXT("iree_hal_buffer_compute_view_size"), Status))
		{
			return false;
		}

		iree_hal_buffer_params_t Params = {};
		Params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
		Params.access = IREE_HAL_MEMORY_ACCESS_ALL;
		Params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

		iree_hal_buffer_t* Buffer = nullptr;
		Status = iree_hal_allocator_allocate_buffer(iree_runtime_session_device_allocator(Session), Params, AllocationSize, &Buffer);
		if (!LogIreeStatus(TEXT("iree_hal_allocator_allocate_buffer"), Status))
		{
			return false;
		}

		Status = iree_hal_buffer_view_create(
			Buffer, Dims.Num(), Dims.GetData(), ElementType, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, HostAllocator, OutView);
		iree_hal_buffer_release(Buffer);
		if (!LogIreeStatus(TEXT("iree_hal_buffer_view_create"), Status))
		{
			return false;
		}

		OutSizeInBytes = static_cast<uint64>(AllocationSize);
		return true;
	}

	bool CopyBufferViewToBinding(iree_hal_device_t* Device, iree_hal_buffer_view_t* View, FIREETensorBinding& Binding)
	{
		const iree_host_size_t ByteLength = iree_hal_buffer_view_byte_length(View);
		if (!Binding.Data || Binding.SizeInBytes < ByteLength)
		{
			UE_LOGFMT(LogScholaIREE, Error, "IREERuntimeSession: output binding too small ({0} < {1})", Binding.SizeInBytes, ByteLength);
			return false;
		}

		iree_hal_buffer_t* Buffer = iree_hal_buffer_view_buffer(View);
		iree_status_t Status = iree_hal_device_transfer_d2h(
			Device,
			Buffer,
			0,
			Binding.Data,
			ByteLength,
			IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
			iree_infinite_timeout());
		return LogIreeStatus(TEXT("iree_hal_device_transfer_d2h"), Status);
	}
}

void FIREERuntimeSession::ReleaseInputSlots()
{
	for (FInputSlot& Slot : InputSlots)
	{
		if (Slot.View)
		{
			iree_hal_buffer_view_release(Slot.View);
		}
	}
	InputSlots.Reset();
}

bool FIREERuntimeSession::PrepareInput(int32 Index, const FIREETensorBinding& Binding, const FScholaIREETensorInfo& Desc)
{
	if (!Binding.Data || Binding.SizeInBytes == 0)
	{
		UE_LOGFMT(LogScholaIREE, Error, "IREERuntimeSession: empty binding for tensor {0}", Desc.Name);
		return false;
	}

	FInputSlot& Slot = InputSlots[Index];
	if (!Slot.View)
	{
		if (!CreateInputBufferView(Session, iree_runtime_instance_host_allocator(Instance), Binding, Desc, &Slot.View, Slot.SizeInBytes))
		{
			return false;
		}
	}

	if (Binding.SizeInBytes != Slot.SizeInBytes)
	{
		UE_LOGFMT(LogScholaIREE, Error, "IREERuntimeSession: binding for tensor {0} is {1} bytes but its shape needs {2}",
			Desc.Name, Binding.SizeInBytes, Slot.SizeInBytes);
		return false;
	}

	iree_hal_buffer_t* Buffer = iree_hal_buffer_view_buffer(Slot.View);
	iree_status_t Status = iree_hal_device_transfer_h2d(
		iree_runtime_session_device(Session),
		Binding.Data,
		Buffer,
		0,
		Slot.SizeInBytes,
		IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
		iree_infinite_timeout());
	return LogIreeStatus(TEXT("iree_hal_device_transfer_h2d"), Status);
}

bool FIREERuntimeSession::ReadReflectionAttribute(TConstArrayView<uint8> VmfbBytes, const FString& Key, FString& OutValue)
{
	OutValue.Reset();
	if (VmfbBytes.Num() == 0)
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREERuntimeSession::ReadReflectionAttribute: empty vmfb");
		return false;
	}

	// Load the module through a throwaway synchronous CPU session. This uses only the public runtime API
	// (the bytecode module header needs flatcc headers that are not vendored under ThirdParty/IREE).
	FIREERuntimeSession Temp;
	if (!Temp.Initialize(TEXT("local-sync"), VmfbBytes))
	{
		return false;
	}

	// The module we appended is registered last, after the HAL module(s).
	iree_vm_context_t* Context = iree_runtime_session_context(Temp.Session);
	const iree_host_size_t ModuleCount = iree_vm_context_module_count(Context);
	if (ModuleCount == 0)
	{
		return false;
	}
	iree_vm_module_t* Module = iree_vm_context_module_at(Context, ModuleCount - 1);

	const iree_string_view_t KeyView = iree_make_cstring_view(TCHAR_TO_UTF8(*Key));
	const iree_vm_module_signature_t Signature = iree_vm_module_signature(Module);
	bool bFound = false;
	for (iree_host_size_t Ordinal = 0; Ordinal < Signature.export_function_count && !bFound; ++Ordinal)
	{
		iree_vm_function_t Function;
		if (!iree_status_is_ok(iree_vm_module_lookup_function_by_ordinal(Module, IREE_VM_FUNCTION_LINKAGE_EXPORT, Ordinal, &Function)))
		{
			continue;
		}
		const iree_string_view_t Value = iree_vm_function_lookup_attr_by_name(&Function, KeyView);
		if (Value.size > 0)
		{
			// The view points into the module, so copy before Temp (and the module) is destroyed.
			OutValue = FString(FUTF8ToTCHAR(Value.data, static_cast<int32>(Value.size)));
			bFound = true;
		}
	}

	if (!bFound)
	{
		UE_LOGFMT(LogScholaIREE, Warning, "FIREERuntimeSession::ReadReflectionAttribute: no iree.reflection key {0} on any export", Key);
	}
	return bFound;
}

bool FIREERuntimeSession::Initialize(const FString& DeviceUri, TConstArrayView<uint8> VmfbBytes)
{
	bReady = false;
	if (VmfbBytes.Num() == 0)
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREERuntimeSession::Initialize: empty vmfb");
		return false;
	}

	iree_runtime_instance_options_t InstanceOptions;
	iree_runtime_instance_options_initialize(&InstanceOptions);
	iree_runtime_instance_options_use_all_available_drivers(&InstanceOptions);
	iree_status_t Status = iree_runtime_instance_create(&InstanceOptions, iree_allocator_system(), &Instance);
	if (!LogIreeStatus(TEXT("iree_runtime_instance_create"), Status))
	{
		return false;
	}

	iree_hal_device_t* Device = nullptr;
	Status = iree_runtime_instance_try_create_default_device(
		Instance,
		iree_make_cstring_view(TCHAR_TO_UTF8(*DeviceUri)),
		&Device);
	if (!LogIreeStatus(TEXT("iree_runtime_instance_try_create_default_device"), Status))
	{
		return false;
	}

	iree_runtime_session_options_t SessionOptions;
	iree_runtime_session_options_initialize(&SessionOptions);
	Status = iree_runtime_session_create_with_device(
		Instance,
		&SessionOptions,
		Device,
		iree_runtime_instance_host_allocator(Instance),
		&Session);
	iree_hal_device_release(Device);
	if (!LogIreeStatus(TEXT("iree_runtime_session_create_with_device"), Status))
	{
		return false;
	}

	OwnedVmfbBytes = TArray<uint8>(VmfbBytes);
	Status = iree_runtime_session_append_bytecode_module_from_memory(
		Session,
		iree_make_const_byte_span(OwnedVmfbBytes.GetData(), OwnedVmfbBytes.Num()),
		iree_allocator_null());
	if (!LogIreeStatus(TEXT("iree_runtime_session_append_bytecode_module_from_memory"), Status))
	{
		return false;
	}

	LoadedDeviceUri = DeviceUri;
	bReady = true;
	return true;
}

bool FIREERuntimeSession::Invoke(
	const FString& EntryFunction,
	TConstArrayView<FIREETensorBinding> InputBindings,
	TConstArrayView<FScholaIREETensorInfo> InputDescs,
	TArrayView<FIREETensorBinding> OutputBindings,
	TConstArrayView<FScholaIREETensorInfo> OutputDescs)
{
	if (!Session || !bReady)
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREERuntimeSession::Invoke: session not initialized");
		return false;
	}
	if (InputBindings.Num() != InputDescs.Num() || OutputBindings.Num() != OutputDescs.Num())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREERuntimeSession::Invoke: binding/desc count mismatch");
		return false;
	}

	const FString QualifiedName = FString::Printf(TEXT("module.%s"), *EntryFunction);

	iree_runtime_call_t Call;
	iree_status_t Status = iree_runtime_call_initialize_by_name(
		Session,
		iree_make_cstring_view(TCHAR_TO_UTF8(*QualifiedName)),
		&Call);
	if (!LogIreeStatus(TEXT("iree_runtime_call_initialize_by_name"), Status))
	{
		return false;
	}

	if (InputSlots.Num() != InputBindings.Num())
	{
		ReleaseInputSlots();
		InputSlots.SetNum(InputBindings.Num());
	}

	for (int32 Index = 0; Index < InputBindings.Num(); ++Index)
	{
		if (!PrepareInput(Index, InputBindings[Index], InputDescs[Index]))
		{
			iree_runtime_call_deinitialize(&Call);
			return false;
		}
		Status = iree_runtime_call_inputs_push_back_buffer_view(&Call, InputSlots[Index].View);
		if (!LogIreeStatus(TEXT("iree_runtime_call_inputs_push_back_buffer_view"), Status))
		{
			iree_runtime_call_deinitialize(&Call);
			return false;
		}
	}

	Status = iree_runtime_call_invoke(&Call, /*flags=*/0);
	if (!LogIreeStatus(TEXT("iree_runtime_call_invoke"), Status))
	{
		iree_runtime_call_deinitialize(&Call);
		return false;
	}

	for (int32 Index = 0; Index < OutputDescs.Num(); ++Index)
	{
		iree_hal_buffer_view_t* View = nullptr;
		Status = iree_runtime_call_outputs_pop_front_buffer_view(&Call, &View);
		if (!LogIreeStatus(TEXT("iree_runtime_call_outputs_pop_front_buffer_view"), Status))
		{
			iree_runtime_call_deinitialize(&Call);
			return false;
		}
		const bool bCopied = CopyBufferViewToBinding(iree_runtime_session_device(Session), View, OutputBindings[Index]);
		iree_hal_buffer_view_release(View);
		if (!bCopied)
		{
			iree_runtime_call_deinitialize(&Call);
			return false;
		}
	}

	iree_runtime_call_deinitialize(&Call);
	return true;
}

#endif // SCHOLA_IREE_HAS_RUNTIME
