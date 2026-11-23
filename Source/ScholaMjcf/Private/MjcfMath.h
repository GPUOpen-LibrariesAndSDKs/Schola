// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

/** Math the MJCF parser needs that doesn't depend on the file being read. Everything is in MuJoCo's units and axes. */
class FMjcfMath
{
public:
	/** Returns the rotation that takes +Z to Direction, choosing the same one MuJoCo does for zaxis and fromto. */
	static FQuat RotationFromZ(const FVector& Direction);

	/**
	 * Lowercase axes in Sequence rotate with the frame and uppercase axes stay fixed, as in MuJoCo's eulerseq.
	 *
	 * @param Sequence Three of the letters xyzXYZ.
	 */
	static FQuat EulerToRotation(const FVector& Radians, const FString& Sequence);

	/**
	 * Sets OutRotation to the rotation whose X axis is along XAxis and whose Y axis is the part of YAxis perpendicular to it,
	 * as in MuJoCo's xyaxes. Returns false, leaving OutRotation unchanged, if either axis is zero or the two are parallel.
	 */
	static bool RotationFromXYAxes(const FVector& XAxis, const FVector& YAxis, FQuat& OutRotation);

	/** Returns the eigenvalues of the symmetric matrix Inertia, and the rotation whose axes are the matching eigenvectors. */
	static void DiagonalizeInertia(const double (&Inertia)[3][3], FVector& OutMoments, FQuat& OutRotation);
};
