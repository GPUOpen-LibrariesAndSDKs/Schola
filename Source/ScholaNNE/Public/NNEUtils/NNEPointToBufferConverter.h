// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "NNEUtils/NNEBufferVisitor.h"
#include "NNEUtils/NNEBuffer.h"
#include "StructUtils/InstancedStruct.h"
#include "Points/Point.h"
#include "Spaces/Space.h"
#include "LogScholaNNE.h"

/**
 * @brief Visitor that converts Schola points into NNE buffers
 * 
 * This visitor reads Schola point structures and converts them into NNE buffers
 * suitable for neural network inference, guided by space definitions.
 */
struct SCHOLANNE_API FNNEPointToBufferConverter : public FNNEBufferVisitor
{
	/** Reference to the output buffer being populated */
    TInstancedStruct<FNNEPointBuffer>& OutputBuffer;
    
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
    FNNEPointToBufferConverter(
        TInstancedStruct<FNNEPointBuffer>& OutBuffer,
        const TInstancedStruct<FPoint>& InPoint,
        const TInstancedStruct<FSpace>& InSpace)
        : OutputBuffer(OutBuffer), InputPoint(InPoint), Space(InSpace)
    {
    }

	/**
	 * @brief Visit operation for dictionary buffers
	 * @param[in] InBuffer The dictionary buffer to populate
	 */
    void operator()(const FNNEDictBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for box buffers
	 * @param[in] InBuffer The box buffer to populate
	 */
    void operator()(const FNNEBoxBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for multi-binary buffers
	 * @param[in] InBuffer The multi-binary buffer to populate
	 */
    void operator()(const FNNEMultiBinaryBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for discrete buffers
	 * @param[in] InBuffer The discrete buffer to populate
	 */
    void operator()(const FNNEDiscreteBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for multi-discrete buffers
	 * @param[in] InBuffer The multi-discrete buffer to populate
	 */
    void operator()(const FNNEMultiDiscreteBuffer& InBuffer) override;

	/**
	 * @brief Static utility function to convert a point to a buffer
	 * @param[in] InPoint The point to convert
	 * @param[out] OutBuffer The buffer to populate
	 * @param[in] InSpace The space definition guiding the conversion
	 */
    inline static void ConvertPointToBuffer(const TInstancedStruct<FPoint>& InPoint, TInstancedStruct<FNNEPointBuffer>& OutBuffer, const TInstancedStruct<FSpace>& InSpace)
    {
        if (!InPoint.IsValid() || !OutBuffer.IsValid() || !InSpace.IsValid())
        {
            UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::ConvertPointToBuffer(): Invalid input parameters");
            return;
        }

        FNNEPointToBufferConverter Converter(OutBuffer, InPoint, InSpace);
        OutBuffer.Get<FNNEPointBuffer>().Accept(Converter);
    }
};


