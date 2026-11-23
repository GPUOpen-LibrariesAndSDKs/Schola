// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

/**
 * @brief Converts MuJoCo positions, rotations, and mesh vertices to Unreal's.
 *
 * MuJoCo uses meters and right-handed axes, and Unreal uses centimeters and left-handed axes. Both are Z-up, so X and Z are
 * kept and Y is mirrored, which is also how Unreal imports OBJ files.
 *
 * Placing converted geometry with a converted position and rotation gives the same result as converting geometry that was
 * placed in MuJoCo, so meshes and the transforms that place them can be converted separately.
 */
class SCHOLAMJCF_API FMjcfConversion
{
public:
	static constexpr double CentimetersPerMeter = 100.0;

	static FVector ConvertPosition(const FVector& Position);

	/**
	 * @param Rotation A rotation of MuJoCo coordinates. MJCF writes quaternions as "w x y z", while FQuat takes (X, Y, Z, W).
	 */
	static FQuat ConvertRotation(const FQuat& Rotation);

	/** Returns the matrix that takes a mesh file's vertices to Unreal, applying the MJCF mesh scale first. */
	static FMatrix MeshToUnreal(const FVector& MeshScale);
};
