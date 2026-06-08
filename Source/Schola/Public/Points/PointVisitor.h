// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

struct FMultiBinaryPoint;
struct FDiscretePoint;
struct FMultiDiscretePoint;
struct FBoxPoint;
struct FDictPoint;
struct FTextPoint;

/**
 * @class PointVisitor
 * @brief Base class for visitors that can operate on mutable points.
 * 
 * This class implements the visitor pattern for points, allowing operations
 * to be performed on any point type without knowing the concrete type at
 * compile time. Derived classes override the operator() methods to implement
 * specific operations for each point type.
 */
class SCHOLA_API PointVisitor
{
public:
	/**
	 * @brief Visits a MultiBinaryPoint.
	 * @param[in,out] Point The MultiBinaryPoint to visit.
	 */
	virtual void operator()(FMultiBinaryPoint& Point){};

	/**
	 * @brief Visits a DiscretePoint.
	 * @param[in,out] Point The DiscretePoint to visit.
	 */
	virtual void operator()(FDiscretePoint& Point){};

	/**
	 * @brief Visits a BoxPoint.
	 * @param[in,out] Point The BoxPoint to visit.
	 */
	virtual void operator()(FBoxPoint& Point){};

	/**
	 * @brief Visits a DictPoint.
	 * @param[in,out] Point The DictPoint to visit.
	 */
	virtual void operator()(FDictPoint& Point){};

	/**
	 * @brief Visits a MultiDiscretePoint.
	 * @param[in,out] Point The MultiDiscretePoint to visit.
	 */
	virtual void operator()(FMultiDiscretePoint& Point){};

	/**
	 * @brief Visits a TextPoint.
	 * @param[in,out] Point The TextPoint to visit.
	 */
	virtual void operator()(FTextPoint& Point){};

};

/**
 * @class ConstPointVisitor
 * @brief Base class for visitors that can operate on const points.
 * 
 * This class implements the visitor pattern for const points, allowing
 * read-only operations to be performed on any point type without knowing
 * the concrete type at compile time. Derived classes override the operator()
 * methods to implement specific operations for each point type.
 */
class ConstPointVisitor
{
public:
	/**
	 * @brief Visits a const MultiBinaryPoint.
	 * @param[in] Point The const MultiBinaryPoint to visit.
	 */
	virtual void operator()(const FMultiBinaryPoint& Point){};

	/**
	 * @brief Visits a const DiscretePoint.
	 * @param[in] Point The const DiscretePoint to visit.
	 */
	virtual void operator()(const FDiscretePoint& Point){};

	/**
	 * @brief Visits a const BoxPoint.
	 * @param[in] Point The const BoxPoint to visit.
	 */
	virtual void operator()(const FBoxPoint& Point){};

	/**
	 * @brief Visits a const DictPoint.
	 * @param[in] Point The const DictPoint to visit.
	 */
	virtual void operator()(const FDictPoint& Point){};

	/**
	 * @brief Visits a const MultiDiscretePoint.
	 * @param[in] Point The const MultiDiscretePoint to visit.
	 */
	virtual void operator()(const FMultiDiscretePoint& Point){};

	/**
	 * @brief Visits a const TextPoint.
	 * @param[in] Point The const TextPoint to visit.
	 */
	virtual void operator()(const FTextPoint& Point){};
};
