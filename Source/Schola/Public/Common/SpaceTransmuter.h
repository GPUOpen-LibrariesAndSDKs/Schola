// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Spaces/SpaceVisitor.h"
// Removed erroneous generated header include. This file has no UHT-reflected types
// so no corresponding .generated.h will be produced.


/**
 * @class SpaceTransmuter
 * @brief A visitor class that can serialize space definitions to a buffer.
 * 
 * This template class implements the visitor pattern to traverse and serialize
 * different types of spaces into a buffer of type T. It can be used to convert
 * space definitions into various serialized formats for storage or transmission.
 * 
 * @tparam T The type of buffer to serialize to.
 */
template<typename T>
class SpaceTransmuter : public ConstSpaceVisitor
{
public:
	/**
	 * @brief Processes a multi-binary space for serialization.
	 * @param[in] Space The multi-binary space to process.
	 */
	virtual void operator()(const FMultiBinarySpace& Space) override;

	/**
	 * @brief Processes a discrete space for serialization.
	 * @param[in] Space The discrete space to process.
	 */
	virtual void operator()(const FDiscreteSpace& Space) override;

	/**
	 * @brief Processes a multi-discrete space for serialization.
	 * @param[in] Space The multi-discrete space to process.
	 */
	virtual void operator()(const FMultiDiscreteSpace& Space) override;

	/**
	 * @brief Processes a box space for serialization.
	 * @param[in] Space The box space to process.
	 */
	virtual void operator()(const FBoxSpace& Space) override;

	/**
	 * @brief Processes a dictionary space for serialization.
	 * @param[in] Space The dictionary space to process.
	 */
	virtual void operator()(const FDictSpace& Space) override;

	/**
	 * @brief Processes a text space for serialization.
	 * @param[in] Space The text space to process.
	 */
	virtual void operator()(const FTextSpace& Space) override;

	/**
	 * @brief Builds and returns a new buffer containing the serialized space.
	 * @return Pointer to the newly created buffer.
	 */
	T* Build();

	/**
	 * @brief Builds the serialized space into an existing empty buffer.
	 * @param[in,out] EmptyBuffer Reference to an empty buffer to populate with serialized data.
	 */
	void Build(T& EmptyBuffer);
};
