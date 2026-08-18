// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Sensors/CameraSensorUtils.h"
#include "Sensors/CameraSensor.h"
#include "Engine/TextureRenderTarget2D.h"
#include "LogScholaInteractors.h"

namespace
{
	int32 CountEnabledChannels(uint8 EnabledValidChannels)
	{
		int32 NumChannels = 0;
		if (EnabledValidChannels & static_cast<uint8>(EChannels::R))
		{
			++NumChannels;
		}
		if (EnabledValidChannels & static_cast<uint8>(EChannels::G))
		{
			++NumChannels;
		}
		if (EnabledValidChannels & static_cast<uint8>(EChannels::B))
		{
			++NumChannels;
		}
		if (EnabledValidChannels & static_cast<uint8>(EChannels::A))
		{
			++NumChannels;
		}
		return NumChannels;
	}
}

namespace FCameraSensorUtils
{
	bool ConvertBitmapToBoxPoint(
		const TArray<FColor>& Bitmap,
		int32 Width,
		int32 Height,
		uint8 EnabledValidChannels,
		FBoxPoint& OutBoxPoint)
	{
		const int32 ChannelStride = Width * Height;

		if (Bitmap.Num() != ChannelStride)
		{
			UE_LOGFMT(
				LogScholaInteractors,
				Error,
				"FCameraSensorUtils::ConvertBitmapToBoxPoint(): Bitmap size ({0}) does not match Width * Height ({1}).",
				Bitmap.Num(),
				ChannelStride);
			return false;
		}

		const int32 NumChannels = CountEnabledChannels(EnabledValidChannels);

		OutBoxPoint.Values.SetNum(ChannelStride * NumChannels);
		OutBoxPoint.Shape = { NumChannels, Height, Width };

		for (int32 PixelIndex = 0; PixelIndex < ChannelStride; ++PixelIndex)
		{
			int32 ChannelIndex = 0;

			if (EnabledValidChannels & static_cast<uint8>(EChannels::R))
			{
				OutBoxPoint.Values[ChannelIndex * ChannelStride + PixelIndex] =
					static_cast<float>(Bitmap[PixelIndex].R) / 255.0f;
				++ChannelIndex;
			}

			if (EnabledValidChannels & static_cast<uint8>(EChannels::G))
			{
				OutBoxPoint.Values[ChannelIndex * ChannelStride + PixelIndex] =
					static_cast<float>(Bitmap[PixelIndex].G) / 255.0f;
				++ChannelIndex;
			}

			if (EnabledValidChannels & static_cast<uint8>(EChannels::B))
			{
				OutBoxPoint.Values[ChannelIndex * ChannelStride + PixelIndex] =
					static_cast<float>(Bitmap[PixelIndex].B) / 255.0f;
				++ChannelIndex;
			}

			if (EnabledValidChannels & static_cast<uint8>(EChannels::A))
			{
				OutBoxPoint.Values[ChannelIndex * ChannelStride + PixelIndex] =
					static_cast<float>(Bitmap[PixelIndex].A) / 255.0f;
			}
		}

		return true;
	}

	bool ReadRenderTargetToBoxPoint(
		UTextureRenderTarget2D* TextureTarget,
		uint8 EnabledValidChannels,
		FBoxPoint& OutBoxPoint)
	{
		if (!TextureTarget)
		{
			UE_LOGFMT(LogScholaInteractors, Error, "FCameraSensorUtils::ReadRenderTargetToBoxPoint(): TextureTarget is null.");
			return false;
		}

		FTextureRenderTargetResource* Resource = TextureTarget->GameThread_GetRenderTargetResource();
		if (!Resource)
		{
			UE_LOGFMT(LogScholaInteractors, Error, "FCameraSensorUtils::ReadRenderTargetToBoxPoint(): Render target resource is null.");
			return false;
		}

		TArray<FColor> Bitmap;
		const int32 Width = TextureTarget->GetSurfaceWidth();
		const int32 Height = TextureTarget->GetSurfaceHeight();

		if (!Resource->ReadPixels(Bitmap))
		{
			UE_LOGFMT(LogScholaInteractors, Error, "FCameraSensorUtils::ReadRenderTargetToBoxPoint(): ReadPixels failed.");
			return false;
		}

		return ConvertBitmapToBoxPoint(Bitmap, Width, Height, EnabledValidChannels, OutBoxPoint);
	}
}
