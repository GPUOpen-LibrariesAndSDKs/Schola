// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Spaces/BoxSpace.h"

int GetTotalSize(const TArray<int>& Shape)
{
	if(Shape.Num() == 0)
	{
		return 0;
	}
	else
	{
		int TotalSize = 1;
		for (const int Dim : Shape)
		{
			TotalSize *= Dim;
		}
		return TotalSize;
	}
}

int GetTotalSize(std::initializer_list<int> Shape)
{
	if (Shape.size() == 0)
	{
		return 0;
	}
	else
	{
		int TotalSize = 1;
		for (const int Dim : Shape)
		{
			TotalSize *= Dim;
		}
		return TotalSize;
	}
}

FBoxSpace::FBoxSpace()
{
}

FBoxSpace::FBoxSpace(const TArray<float>& Low, const TArray<float>& High, const TArray<int>& Shape)
{
	verifyf(Low.Num() == High.Num(), TEXT("High and Low must have the same shape"));
	
	int TotalShapeSize = GetTotalSize(Shape);
	if (TotalShapeSize == 0)
	{
		this->Shape = TArray<int>({ High.Num() });
	}
	else
	{
		verifyf(TotalShapeSize == High.Num(), TEXT("High and Low must match the shape of the BoxSpace"));
		this->Shape = Shape;
	}
	//Can use just low here since both high and low are the same size
	for (int i = 0; i < Low.Num(); i++)
	{
		this->Dimensions.Add(FBoxSpaceDimension(Low[i], High[i]));
	}
	this->Shape = Shape;
}

FBoxSpace::FBoxSpace(const TArray<FBoxSpaceDimension>& Dimensions, const TArray<int>& Shape)
{
	int TotalShapeSize = GetTotalSize(Shape);
	if (TotalShapeSize == 0)
	{
		this->Shape = TArray<int>({ Dimensions.Num() });
	}
	else
	{
		verifyf(TotalShapeSize == Dimensions.Num(), TEXT("Dimensions must match the shape of the BoxSpace"));
		this->Shape = Shape;
	}
	for (const FBoxSpaceDimension& Dimension : Dimensions)
	{
		this->Dimensions.Add(Dimension);
	}
}

FBoxSpace::FBoxSpace(std::initializer_list<float> Low, std::initializer_list<float> High, std::initializer_list<int> Shape)
{
	auto LowIt = Low.begin();
	auto HighIt = High.begin();
	checkf(Low.size() == High.size(), TEXT("Size mismatch between low and high arrays"));
	int TotalShapeSize = GetTotalSize(Shape);
	if (TotalShapeSize == 0)
	{
		this->Shape = {(int)Low.size()};
	}
	else
	{
		verifyf(TotalShapeSize == Low.size(), TEXT("Dimensions must match the shape of the BoxSpace"));
		this->Shape = Shape;
	}
	
	while (LowIt != Low.end() && HighIt != High.end())
	{
		this->Dimensions.Add(FBoxSpaceDimension(*LowIt, *HighIt));
		LowIt++;
		HighIt++;
	}
}

FBoxSpace::FBoxSpace(const TArray<int>& Shape)
{
	this->Dimensions.SetNumUninitialized(GetTotalSize(Shape));
	this->Shape = Shape;
}

void FBoxSpace::Copy(const FBoxSpace& Other)
{
	this->Dimensions = TArray<FBoxSpaceDimension>(Other.Dimensions);
	this->Shape = Other.Shape;
}

ESpaceValidationResult FBoxSpace::Validate(const TInstancedStruct<FPoint>& Point) const
{
	if (!Point.GetPtr<FBoxPoint>())
	{
		return ESpaceValidationResult::WrongDataType;
	}
	const FBoxPoint& TypedObservation = Point.Get<FBoxPoint>();
	// Check that the real number of dimensions matches
	if (Dimensions.Num() != TypedObservation.Values.Num())
	{
		return ESpaceValidationResult::WrongDimensions;
	}
	
	if (Shape.Num() > 0 && TypedObservation.Shape.Num() > 0)
	{
		if (Shape != TypedObservation.Shape)
		{
			return ESpaceValidationResult::WrongDimensions;
		}
	}

	for (int i = 0; i < Dimensions.Num(); i++)
	{
		if (TypedObservation.Values[i] > Dimensions[i].High || TypedObservation.Values[i] < Dimensions[i].Low)
		{
			return ESpaceValidationResult::OutOfBounds;
		}
	}

	return ESpaceValidationResult::Success;
}

FBoxPoint FBoxSpace::NormalizeObservation(const FBoxPoint& Observation) const
{
	FBoxPoint OutObservation = FBoxPoint(this->GetFlattenedSize());
	// Use bounding box to normalize the observations, we can safely do so because it has already been validated
	for (int i = 0; i < Dimensions.Num(); i++)
	{
		OutObservation.Add(Dimensions[i].NormalizeValue(Observation.Values[i]));
	}
	return OutObservation;
}

FBoxSpace FBoxSpace::GetNormalizedObservationSpace() const
{
	FBoxSpace OutBoxSpace;

	// Set extent to be between 0 and 1 for normalized observations.
	for (int i = 0; i < Dimensions.Num(); i++)
	{
		OutBoxSpace.Add(FBoxSpaceDimension::ZeroOneUnitDimension());
	}
	return OutBoxSpace;
}

FBoxSpace::~FBoxSpace()
{
	this->Dimensions.Empty();
	this->Shape.Empty();
}

int FBoxSpace::GetNumDimensions() const
{
	return Dimensions.Num();
}


int FBoxSpace::GetFlattenedSize() const
{
	return Dimensions.Num();
}

bool FBoxSpace::IsEmpty() const
{
	return this->Dimensions.IsEmpty();
}

FString FBoxSpace::ToString() const
{
	FString DimPart;
	for (int32 i = 0; i < Dimensions.Num(); ++i)
	{
		if (i > 0)
		{
			DimPart += TEXT(", ");
		}
		DimPart += FString::Printf(TEXT("[%.6g, %.6g]"), Dimensions[i].Low, Dimensions[i].High);
	}
	FString ShapePart;
	if (Shape.Num() > 0)
	{
		ShapePart += TEXT(", Shape=[");
		for (int32 i = 0; i < Shape.Num(); ++i)
		{
			if (i > 0)
			{
				ShapePart += TEXT(", ");
			}
			ShapePart += FString::FromInt(Shape[i]);
		}
		ShapePart += TEXT("]");
	}
	return FString::Printf(TEXT("BoxSpace(Dimensions={%s}%s)"), *DimPart, *ShapePart);
}

void FBoxSpace::Add(float Low, float High)
{
	this->Dimensions.Emplace(Low, High);
}

void FBoxSpace::Add(const FBoxSpaceDimension& Dimension)
{
	this->Dimensions.Add(Dimension);
}