// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Points/BoxPoint.h"

class UTextureRenderTarget2D;

/**
 * @brief Utilities for converting camera render targets into Schola box points.
 */
namespace CameraSensorUtils
{
	/**
	 * @brief Converts a row-major FColor bitmap into a normalized FBoxPoint.
	 *
	 * Output shape is [NumChannels, Height, Width] in row-major (CHW) layout.
	 * Each channel plane matches the row-major ReadPixels bitmap order:
	 * index = c * H * W + h * W + w.
	 *
	 * @param[in] Bitmap Row-major pixel data with length Width * Height.
	 * @param[in] Width Texture width in pixels.
	 * @param[in] Height Texture height in pixels.
	 * @param[in] EnabledValidChannels Bitmask of RGBA channels to include (already filtered).
	 * @param[out] OutBoxPoint Populated box point; left unchanged when Bitmap size does not match Width * Height.
	 * @return True if the bitmap was converted successfully.
	 */
	SCHOLAINTERACTORS_API bool ConvertBitmapToBoxPoint(
		const TArray<FColor>& Bitmap,
		int32 Width,
		int32 Height,
		uint8 EnabledValidChannels,
		FBoxPoint& OutBoxPoint);

	/**
	 * @brief Reads pixels from a render target into a normalized FBoxPoint.
	 *
	 * @param[in] TextureTarget Render target to read from.
	 * @param[in] EnabledValidChannels Bitmask of RGBA channels to include (already filtered).
	 * @param[out] OutBoxPoint Populated box point.
	 * @return True if pixels were read and converted successfully.
	 */
	SCHOLAINTERACTORS_API bool ReadRenderTargetToBoxPoint(
		UTextureRenderTarget2D* TextureTarget,
		uint8 EnabledValidChannels,
		FBoxPoint& OutBoxPoint);
}
