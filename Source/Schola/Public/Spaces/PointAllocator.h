// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
// Point type headers
#include "Points/MultiBinaryPoint.h"
#include "Points/DiscretePoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Points/BoxPoint.h"
#include "Points/DictPoint.h"
#include "Points/TextPoint.h"
// Space type headers (needed for field access in allocation)
#include "Spaces/MultiBinarySpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DictSpace.h"
#include "Spaces/TextSpace.h"


/**
 * @class PointAllocator
 * @brief Visitor class for allocating points that conform to a given space.
 * 
 * This class implements the visitor pattern to traverse space definitions
 * and allocate corresponding point instances with the correct structure and
 * default values. It's used to create points that match a space's requirements.
 */
class SCHOLA_API PointAllocator : public ConstSpaceVisitor
{
public:
	/**
	 * @brief Constructs a PointAllocator for a specific point instance.
	 * @param[in,out] InOutPoint Reference to the point instance to allocate.
	 */
	explicit PointAllocator(TInstancedStruct<FPoint>& InOutPoint)
		: PointToAllocate(InOutPoint)
	{}

	/**
	 * @brief Allocates a MultiBinaryPoint for a MultiBinarySpace.
	 * @param[in] InSpace The MultiBinarySpace to allocate for.
	 */
	void operator()(const FMultiBinarySpace& InSpace) override
	{
		PointToAllocate.InitializeAs<FMultiBinaryPoint>();
		FMultiBinaryPoint& P = PointToAllocate.GetMutable<FMultiBinaryPoint>();
		P.Values.Init(false, InSpace.Shape); // Shape is an int (number of dimensions)
	}

	/**
	 * @brief Allocates a DiscretePoint for a DiscreteSpace.
	 * @param[in] InSpace The DiscreteSpace to allocate for.
	 */
	void operator()(const FDiscreteSpace& InSpace) override
	{
		PointToAllocate.InitializeAs<FDiscretePoint>();
		FDiscretePoint& P = PointToAllocate.GetMutable<FDiscretePoint>();
		P.Value = 0;
	}

	/**
	 * @brief Allocates a MultiDiscretePoint for a MultiDiscreteSpace.
	 * @param[in] InSpace The MultiDiscreteSpace to allocate for.
	 */
	void operator()(const FMultiDiscreteSpace& InSpace) override
	{
		PointToAllocate.InitializeAs<FMultiDiscretePoint>();
		FMultiDiscretePoint& P = PointToAllocate.GetMutable<FMultiDiscretePoint>();
		P.Values.Init(0, InSpace.High.Num());
	}

	/**
	 * @brief Allocates a BoxPoint for a BoxSpace.
	 * @param[in] InSpace The BoxSpace to allocate for.
	 */
	void operator()(const FBoxSpace& InSpace) override
	{
		PointToAllocate.InitializeAs<FBoxPoint>();
		FBoxPoint& P = PointToAllocate.GetMutable<FBoxPoint>();
		P.Values.Init(0.0f, InSpace.Dimensions.Num());
	}

	/**
	 * @brief Allocates a TextPoint for a TextSpace.
	 * @param[in] InSpace The TextSpace to allocate for.
	 */
	void operator()(const FTextSpace& InSpace) override
	{
		PointToAllocate.InitializeAs<FTextPoint>();
	}

	/**
	 * @brief Allocates a DictPoint for a DictSpace.
	 * 
	 * Recursively allocates sub-points for each entry in the dictionary space.
	 * 
	 * @param[in] InSpace The DictSpace to allocate for.
	 */
	void operator()(const FDictSpace& InSpace) override
	{
		PointToAllocate.InitializeAs<FDictPoint>();
		FDictPoint& P = PointToAllocate.GetMutable<FDictPoint>();
		// Recursively allocate sub-points for each entry
		for (const TPair<FString, TInstancedStruct<FSpace>>& Pair : InSpace.Spaces)
		{
			TInstancedStruct<FPoint> NewPoint;
			PointAllocator SubAllocator(NewPoint);
			Pair.Value.Get<FSpace>().Accept(SubAllocator);
			P.Points.Add(Pair.Key, MoveTemp(NewPoint));
		}
	}

	/**
	 * @brief Static utility function to allocate a point for a given space.
	 * @param[in] InSpace The space to allocate a point for.
	 * @param[out] PointToAllocate Output parameter that receives the allocated point.
	 */
	inline static void AllocatePoint(const TInstancedStruct<FSpace>& InSpace, TInstancedStruct<FPoint>& PointToAllocate)
	{
		PointAllocator Allocator(PointToAllocate);
		InSpace.Get<FSpace>().Accept(Allocator);
	}

private:
	TInstancedStruct<FPoint>& PointToAllocate;
};
