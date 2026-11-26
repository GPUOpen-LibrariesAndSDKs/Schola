// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "IREEUtils/IREEBufferVisitor.h"
#include "IREEUtils/IREEBuffer.h"
#include "StructUtils/InstancedStruct.h"
#include "Points/Point.h"
#include "Spaces/Space.h"
#include "LogScholaIREE.h"

/**
 * @brief Visitor that converts Schola points into IREE buffers
 *
 * This visitor reads Schola point structures and converts them into IREE buffers
 * suitable for neural network inference, guided by space definitions.
 */
struct SCHOLAIREE_API FIREEPointToBufferConverter : public FIREEBufferVisitor
{
	/** Reference to the output buffer being populated */
	TInstancedStruct<FIREEPointBuffer>& OutputBuffer;

	/** Input point to convert */
	const TInstancedStruct<FPoint>& InputPoint;

	/** Space definition guiding the conversion */
	const TInstancedStruct<FSpace>& Space;

	/**
	 * @brief Constructor that initializes the converter
	 * @param[out] OutBuffer The buffer to populate with point data
	 * @param[in] InPoint The point to convert
	 * @param[in] InSpace The space definition guiding the conversion
	 */
	FIREEPointToBufferConverter(
		TInstancedStruct<FIREEPointBuffer>& OutBuffer,
		const TInstancedStruct<FPoint>&		InPoint,
		const TInstancedStruct<FSpace>&		InSpace)
		: OutputBuffer(OutBuffer), InputPoint(InPoint), Space(InSpace)
	{
	}

	/**
	 * @brief Visit operation for dictionary buffers
	 * @param[in] InBuffer The dictionary buffer to populate
	 */
	void operator()(const FIREEDictBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for box buffers
	 * @param[in] InBuffer The box buffer to populate
	 */
	void operator()(const FIREEBoxBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for multi-binary buffers
	 * @param[in] InBuffer The multi-binary buffer to populate
	 */
	void operator()(const FIREEMultiBinaryBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for discrete buffers
	 * @param[in] InBuffer The discrete buffer to populate
	 */
	void operator()(const FIREEDiscreteBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for multi-discrete buffers
	 * @param[in] InBuffer The multi-discrete buffer to populate
	 */
	void operator()(const FIREEMultiDiscreteBuffer& InBuffer) override;

	/**
	 * @brief Static utility function to convert a point to a buffer
	 * @param[in] InPoint The point to convert
	 * @param[out] OutBuffer The buffer to populate
	 * @param[in] InSpace The space definition guiding the conversion
	 */
	inline static void ConvertPointToBuffer(const TInstancedStruct<FPoint>& InPoint, TInstancedStruct<FIREEPointBuffer>& OutBuffer, const TInstancedStruct<FSpace>& InSpace)
	{
		if (!InPoint.IsValid() || !OutBuffer.IsValid() || !InSpace.IsValid())
		{
			UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::ConvertPointToBuffer(): Invalid input parameters");
			return;
		}

		FIREEPointToBufferConverter Converter(OutBuffer, InPoint, InSpace);
		OutBuffer.Get<FIREEPointBuffer>().Accept(Converter);
	}
};
