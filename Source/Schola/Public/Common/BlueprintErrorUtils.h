// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StructUtils/InstancedStruct.h"
#include "EngineLogs.h"
#if WITH_EDITOR
#include "Blueprint/BlueprintExceptionInfo.h"
#endif

/**
 * @brief Raises a non-fatal Blueprint script error with a custom message.
 * @param InFunctionName The name of the function where the error occurred.
 * @param InMessage The error message describing what went wrong.
 */
inline void RaiseBlueprintError(const FString& InFunctionName, const FString& InMessage)
{
#if !(UE_BUILD_TEST || UE_BUILD_SHIPPING)
	FFrame* TopFrame = FFrame::GetThreadLocalTopStackFrame();
	if (TopFrame)
	{
		const FString ErrorMessage = FString::Printf(TEXT("%s: %s"), *InFunctionName, *InMessage);
	#if WITH_EDITOR
		const FBlueprintExceptionInfo ExceptionInfo(EBlueprintExceptionType::NonFatalError, FText::FromString(ErrorMessage));
		FBlueprintCoreDelegates::ThrowScriptException(TopFrame->Object, *TopFrame, ExceptionInfo);
	#else
		UE_LOG(LogBlueprintUserMessages, Error, TEXT("%s:\n%s"), *ErrorMessage, *TopFrame->GetStackTrace());
	#endif	// WITH_EDITOR
	}
#endif	// !(UE_BUILD_TEST || UE_BUILD_SHIPPING)
}

/**
 * @brief Raises a Blueprint script error when an invalid InstancedStruct is passed.
 * @param InFunctionName The name of the function where the error occurred.
 */
inline void RaiseInvalidInstancedStructError(const FString& InFunctionName)
{
#if !(UE_BUILD_TEST || UE_BUILD_SHIPPING)
	FFrame* TopFrame = FFrame::GetThreadLocalTopStackFrame();
	if (TopFrame)
	{
		const FString ErrorMessage = FString::Printf(TEXT("%s: Invalid InstancedStruct passed."), *InFunctionName);
	#if WITH_EDITOR
		const FBlueprintExceptionInfo ExceptionInfo(EBlueprintExceptionType::NonFatalError, FText::FromString(ErrorMessage));
		FBlueprintCoreDelegates::ThrowScriptException(TopFrame->Object, *TopFrame, ExceptionInfo);
	#else
		UE_LOG(LogBlueprintUserMessages, Error, TEXT("%s:\n%s"), *ErrorMessage, *TopFrame->GetStackTrace());
	#endif	// WITH_EDITOR
	}
#endif	// !(UE_BUILD_TEST || UE_BUILD_SHIPPING)
}

/**
 * @brief Raises a Blueprint script error when an InstancedStruct has the wrong type.
 * @tparam T The expected type of the InstancedStruct.
 * @param InStruct The InstancedStruct with the wrong type.
 * @param ExpectedType The expected type name.
 * @param InFunctionName The name of the function where the error occurred.
 */
template<typename T>
inline void RaiseInstancedStructTypeMismatchError(const TInstancedStruct<T>& InStruct, const FString& ExpectedType, const FString& InFunctionName)
{
#if !(UE_BUILD_TEST || UE_BUILD_SHIPPING)
	FFrame* TopFrame = FFrame::GetThreadLocalTopStackFrame();
	if (TopFrame)
	{
		const FString ErrorMessage = FString::Printf(TEXT("%s: Type mismatch. Expected %s but received %s."), 
			*InFunctionName, *ExpectedType, *InStruct.GetScriptStruct()->GetName());
	#if WITH_EDITOR
		const FBlueprintExceptionInfo ExceptionInfo(EBlueprintExceptionType::NonFatalError, FText::FromString(ErrorMessage));
		FBlueprintCoreDelegates::ThrowScriptException(TopFrame->Object, *TopFrame, ExceptionInfo);
	#else
		UE_LOG(LogBlueprintUserMessages, Error, TEXT("%s:\n%s"), *ErrorMessage, *TopFrame->GetStackTrace());
	#endif	// WITH_EDITOR
	}
#endif	// !(UE_BUILD_TEST || UE_BUILD_SHIPPING)
}

/**
 * @brief Raises a Blueprint script error when an InstancedStruct has the wrong type.
 * @param InStruct The InstancedStruct with the wrong type.
 * @param ExpectedType The expected type name.
 * @param InFunctionName The name of the function where the error occurred.
 */
 inline void RaiseInstancedStructTypeMismatchError(const FInstancedStruct& InStruct, const FString& ExpectedType, const FString& InFunctionName)
 {
 #if !(UE_BUILD_TEST || UE_BUILD_SHIPPING)
	 FFrame* TopFrame = FFrame::GetThreadLocalTopStackFrame();
	 if (TopFrame)
	 {
		 const FString ErrorMessage = FString::Printf(TEXT("%s: Type mismatch. Expected %s but received %s."), 
			 *InFunctionName, *ExpectedType, *InStruct.GetScriptStruct()->GetName());
	 #if WITH_EDITOR
		 const FBlueprintExceptionInfo ExceptionInfo(EBlueprintExceptionType::NonFatalError, FText::FromString(ErrorMessage));
		 FBlueprintCoreDelegates::ThrowScriptException(TopFrame->Object, *TopFrame, ExceptionInfo);
	 #else
		 UE_LOG(LogBlueprintUserMessages, Error, TEXT("%s:\n%s"), *ErrorMessage, *TopFrame->GetStackTrace());
	 #endif	// WITH_EDITOR
	 }
 #endif	// !(UE_BUILD_TEST || UE_BUILD_SHIPPING)
 }