// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Sensors/CameraSensor.h"
#include "Sensors/CameraSensorUtils.h"
#include "LogScholaInteractors.h"
#include "RHICommandList.h"


void UCameraSensor::InitSensor_Implementation()
{	
	
	if (!TextureTarget)
	{
		UE_LOGFMT(LogScholaInteractors, Verbose, "UCameraSensor::InitSensor_Implementation(): TextureTarget not found. Creating new TextureTarget.");
		TextureTarget = NewObject<UTextureRenderTarget2D>();
		TextureTarget->bGPUSharedFlag = 1;
		TextureTarget->InitAutoFormat(128, 128);
	}

	TextureTarget->bNoFastClear = 0;
	TextureTarget->bHDR_DEPRECATED = 0;
}

void UCameraSensor::GetObservationSpace_Implementation(FInstancedStruct& OutObservationSpace) const
{
	int		  Width = TextureTarget->GetSurfaceWidth();
	int		  Height = TextureTarget->GetSurfaceHeight();
	FBoxSpace SpaceDefinition;
	int		  NumChannels = GetNumChannels();
	SpaceDefinition.Dimensions.Init(FBoxSpaceDimension(0.0, 1.0), Width * Height * NumChannels);
	
	// If a channel is in InvalidChannels, it cannot be observed, or if the channel hasn't been filtered
	// e.g. Channels = R|G, InvalidChannels = A, bHasR will be True but bHasA will be False

	SpaceDefinition.Shape = { NumChannels, Height, Width };

	OutObservationSpace.InitializeAs<FBoxSpace>(MoveTemp(SpaceDefinition));
}

int UCameraSensor::GetNumChannels() const
{
	uint8 EnabledValidChannels = EnabledChannels & ~GetInvalidChannels();
	bool  bHasR = (EnabledValidChannels & static_cast<uint8>(EChannels::R));
	bool  bHasG = (EnabledValidChannels & static_cast<uint8>(EChannels::G));
	bool  bHasB = (EnabledValidChannels & static_cast<uint8>(EChannels::B));
	bool  bHasA = (EnabledValidChannels & static_cast<uint8>(EChannels::A));

	return (bHasR + bHasG + bHasB + bHasA);
}

void UCameraSensor::CollectObservations_Implementation(FInstancedStruct& OutObservations)
{
	if (!TextureTarget)
	{
		UE_LOGFMT(LogScholaInteractors, Error, "UCameraSensor::CollectObservations_Implementation(): RenderTarget not found. Not collecting Observations.");
		return;
	}

	OutObservations.InitializeAs<FBoxPoint>();
	FBoxPoint& OutBoxPoint = OutObservations.GetMutable<FBoxPoint>();

	const uint8 EnabledValidChannels = EnabledChannels & ~GetInvalidChannels();
	if (!FCameraSensorUtils::ReadRenderTargetToBoxPoint(TextureTarget, EnabledValidChannels, OutBoxPoint))
	{
		UE_LOGFMT(LogScholaInteractors, Error, "UCameraSensor::CollectObservations_Implementation(): Failed to read render target into observations.");
	}
}

FString UCameraSensor::GenerateId() const
{	
	
	FString					  Output = FString("Camera"); 

	//Add CaptureSource Enum to Id
	
	Output = Output.Append("_").Append(UEnum::GetValueAsString<ESceneCaptureSource>(this->CaptureSource));

	// Add Render Target Enum to Id
	if (this->TextureTarget)
	{
		Output = Output.Append("_").Append(UEnum::GetValueAsString<ETextureRenderTargetFormat>(this->TextureTarget->RenderTargetFormat));
	}
	
	Output.Append("_");
	//Add channels to Id
	uint8 InvalidChannels = GetInvalidChannels();

	if (EnabledChannels & ~InvalidChannels & static_cast<uint8>(EChannels::R))
	{
		Output = Output.Append("R");
	}
	
	if (EnabledChannels & ~InvalidChannels & static_cast<uint8>(EChannels::G))
	{
		Output = Output.Append("G");
	}

	if (EnabledChannels & ~InvalidChannels & static_cast<uint8>(EChannels::B))
	{
		Output = Output.Append("B");
	}

	if (EnabledChannels & ~InvalidChannels & static_cast<uint8>(EChannels::A))
	{
		Output = Output.Append("A");
	}

	//Add width and height
	Output = Output.Appendf(TEXT("_W%.3f_H%.3f"), TextureTarget->GetSurfaceWidth(), TextureTarget->GetSurfaceHeight()); // Width and Height
	return Output;
}

uint8 UCameraSensor::GetInvalidChannels() const
{
	uint8 InvalidChannels = 0;

	switch (this->CaptureSource)
	{
		case ESceneCaptureSource::SCS_SceneColorHDRNoAlpha:
		case ESceneCaptureSource::SCS_FinalColorLDR:
		case ESceneCaptureSource::SCS_DeviceDepth:
		case ESceneCaptureSource::SCS_Normal:
		case ESceneCaptureSource::SCS_BaseColor:
		case ESceneCaptureSource::SCS_FinalColorHDR:
		case ESceneCaptureSource::SCS_FinalToneCurveHDR:
			InvalidChannels = static_cast<uint8>(EChannels::A);
			break;

		case ESceneCaptureSource::SCS_SceneColorSceneDepth:
			break;

		case ESceneCaptureSource::SCS_SceneDepth:
			InvalidChannels = static_cast<uint8>(EChannels::G | EChannels::B | EChannels::A);
			break;

		default:
			break;
	}

	// Set channels based on the RenderTarget
	if (this->TextureTarget)
	{
		switch (this->TextureTarget->RenderTargetFormat)
		{
			case ETextureRenderTargetFormat::RTF_RG32f:
			case ETextureRenderTargetFormat::RTF_RG16f:
			case ETextureRenderTargetFormat::RTF_RG8:
				InvalidChannels = InvalidChannels | static_cast<uint8>(EChannels::B | EChannels::A);
				break;
			case ETextureRenderTargetFormat::RTF_R32f:
			case ETextureRenderTargetFormat::RTF_R16f:
			case ETextureRenderTargetFormat::RTF_R8:
				InvalidChannels = InvalidChannels | static_cast<uint8>(EChannels::G | EChannels::B | EChannels::A);
				break;
			default:
				break;
		}
	}

	return InvalidChannels;
}