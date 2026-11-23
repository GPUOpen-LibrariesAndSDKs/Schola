// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "MjcfMath.h"

FQuat FMjcfMath::RotationFromZ(const FVector& Direction)
{
	const FVector Z = Direction.GetSafeNormal();
	const FVector Axis = FVector::CrossProduct(FVector::UpVector, Z);
	const double Sin = Axis.Size();
	if (Sin < 1e-10)
	{
		return Z.Z < 0 ? FQuat(1, 0, 0, 0) : FQuat::Identity;
	}
	return FQuat(Axis / Sin, FMath::Atan2(Sin, Z.Z));
}

FQuat FMjcfMath::EulerToRotation(const FVector& Radians, const FString& Sequence)
{
	FQuat Result = FQuat::Identity;
	for (int32 Index = 0; Index < 3; ++Index)
	{
		const TCHAR Axis = FChar::ToLower(Sequence[Index]);
		const FVector AxisVector = Axis == TEXT('x') ? FVector::ForwardVector : Axis == TEXT('y') ? FVector::RightVector : FVector::UpVector;
		const FQuat Step(AxisVector, Radians[Index]);
		Result = FChar::IsLower(Sequence[Index]) ? Result * Step : Step * Result;
	}
	return Result;
}

bool FMjcfMath::RotationFromXYAxes(const FVector& XAxis, const FVector& YAxis, FQuat& OutRotation)
{
	const FVector X = XAxis.GetSafeNormal();
	const FVector Y = (YAxis - X * FVector::DotProduct(X, YAxis)).GetSafeNormal();
	if (X.IsZero() || Y.IsZero())
	{
		return false;
	}
	OutRotation = FQuat(FMatrix(X, Y, FVector::CrossProduct(X, Y), FVector::ZeroVector));
	return true;
}

void FMjcfMath::DiagonalizeInertia(const double (&Inertia)[3][3], FVector& OutMoments, FQuat& OutRotation)
{
	double A[3][3];
	double V[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
	FMemory::Memcpy(A, Inertia, sizeof(A));

	// Jacobi's method: each rotation zeroes one off-diagonal entry, and the sweeps converge in a handful of passes for 3x3.
	static const int32 Pairs[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
	for (int32 Sweep = 0; Sweep < 50; ++Sweep)
	{
		const double OffDiagonal = FMath::Abs(A[0][1]) + FMath::Abs(A[0][2]) + FMath::Abs(A[1][2]);
		const double Diagonal = FMath::Abs(A[0][0]) + FMath::Abs(A[1][1]) + FMath::Abs(A[2][2]);
		if (OffDiagonal <= 1e-15 * Diagonal)
		{
			break;
		}
		for (const int32 (&Pair)[2] : Pairs)
		{
			const int32 P = Pair[0];
			const int32 Q = Pair[1];
			if (A[P][Q] == 0)
			{
				continue;
			}
			const double Theta = (A[Q][Q] - A[P][P]) / (2 * A[P][Q]);
			const double T = (Theta >= 0 ? 1.0 : -1.0) / (FMath::Abs(Theta) + FMath::Sqrt(Theta * Theta + 1));
			const double C = 1 / FMath::Sqrt(T * T + 1);
			const double S = T * C;

			double J[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
			J[P][P] = C;
			J[Q][Q] = C;
			J[P][Q] = S;
			J[Q][P] = -S;

			double AJ[3][3] = {};
			double NewV[3][3] = {};
			for (int32 Row = 0; Row < 3; ++Row)
			{
				for (int32 Column = 0; Column < 3; ++Column)
				{
					for (int32 K = 0; K < 3; ++K)
					{
						AJ[Row][Column] += A[Row][K] * J[K][Column];
						NewV[Row][Column] += V[Row][K] * J[K][Column];
					}
				}
			}
			for (int32 Row = 0; Row < 3; ++Row)
			{
				for (int32 Column = 0; Column < 3; ++Column)
				{
					A[Row][Column] = 0;
					for (int32 K = 0; K < 3; ++K)
					{
						A[Row][Column] += J[K][Row] * AJ[K][Column];
					}
				}
			}
			FMemory::Memcpy(V, NewV, sizeof(V));
		}
	}

	OutMoments = FVector(A[0][0], A[1][1], A[2][2]);
	FVector Axes[3];
	for (int32 Index = 0; Index < 3; ++Index)
	{
		Axes[Index] = FVector(V[0][Index], V[1][Index], V[2][Index]);
	}
	if (FVector::DotProduct(FVector::CrossProduct(Axes[0], Axes[1]), Axes[2]) < 0)
	{
		Axes[2] = -Axes[2];
	}
	OutRotation = FQuat(FMatrix(Axes[0], Axes[1], Axes[2], FVector::ZeroVector)).GetNormalized();
}
