// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

struct FMultiBinarySpace;
struct FDiscreteSpace;
struct FBoxSpace;
struct FDictSpace;
struct FMultiDiscreteSpace;
struct FTextSpace;

/**
 * @class SpaceVisitor
 * @brief Base class for visitors that can operate on mutable spaces.
 * 
 * This class implements the visitor pattern for spaces, allowing operations
 * to be performed on any space type without knowing the concrete type at
 * compile time. Derived classes override the operator() methods to implement
 * specific operations for each space type.
 */
class SCHOLA_API SpaceVisitor
{
public:
	/**
	 * @brief Visits a MultiBinarySpace.
	 * @param[in,out] Space The MultiBinarySpace to visit.
	 */
	virtual void operator()(FMultiBinarySpace& Space){};

	/**
	 * @brief Visits a DiscreteSpace.
	 * @param[in,out] Space The DiscreteSpace to visit.
	 */
	virtual void operator()(FDiscreteSpace& Space){};

	/**
	 * @brief Visits a MultiDiscreteSpace.
	 * @param[in,out] Space The MultiDiscreteSpace to visit.
	 */
	virtual void operator()(FMultiDiscreteSpace& Space){};

	/**
	 * @brief Visits a BoxSpace.
	 * @param[in,out] Space The BoxSpace to visit.
	 */
	virtual void operator()(FBoxSpace& Space){};

	/**
	 * @brief Visits a DictSpace.
	 * @param[in,out] Space The DictSpace to visit.
	 */
	virtual void operator()(FDictSpace& Space){};

	/**
	 * @brief Visits a TextSpace.
	 * @param[in,out] Space The TextSpace to visit.
	 */
	virtual void operator()(FTextSpace& Space){};

};

/**
 * @class ConstSpaceVisitor
 * @brief Base class for visitors that can operate on const spaces.
 * 
 * This class implements the visitor pattern for const spaces, allowing
 * read-only operations to be performed on any space type without knowing
 * the concrete type at compile time. Derived classes override the operator()
 * methods to implement specific operations for each space type.
 */
class SCHOLA_API ConstSpaceVisitor
{
public:
	/**
	 * @brief Visits a const MultiBinarySpace.
	 * @param[in] Space The const MultiBinarySpace to visit.
	 */
	virtual void operator()(const FMultiBinarySpace& Space){};

	/**
	 * @brief Visits a const DiscreteSpace.
	 * @param[in] Space The const DiscreteSpace to visit.
	 */
	virtual void operator()(const FDiscreteSpace& Space){};

	/**
	 * @brief Visits a const MultiDiscreteSpace.
	 * @param[in] Space The const MultiDiscreteSpace to visit.
	 */
	virtual void operator()(const FMultiDiscreteSpace& Space){};

	/**
	 * @brief Visits a const BoxSpace.
	 * @param[in] Space The const BoxSpace to visit.
	 */
	virtual void operator()(const FBoxSpace& Space){};

	/**
	 * @brief Visits a const DictSpace.
	 * @param[in] Space The const DictSpace to visit.
	 */
	virtual void operator()(const FDictSpace& Space){};

	/**
	 * @brief Visits a const TextSpace.
	 * @param[in] Space The const TextSpace to visit.
	 */
	virtual void operator()(const FTextSpace& Space){};

};
