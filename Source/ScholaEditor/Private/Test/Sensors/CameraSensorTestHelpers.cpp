// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Test/Sensors/CameraSensorTestHelpers.h"

#include "Sensors/CameraSensorUtils.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/Canvas.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture2D.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "RHICommandList.h"

bool FScholaCameraSensorTestWorld::Setup(FAutomationTestBase* InTest)
{
	Test = InTest;
	if (!Wrapper.CreateTestWorld(EWorldType::Game))
	{
		Test->AddError(TEXT("Failed to create camera sensor test world"));
		return false;
	}
	Phase = EPhase::Created;

	if (!Wrapper.GetTestWorld())
	{
		Test->AddError(TEXT("Failed to get camera sensor test world"));
		Wrapper.DestroyTestWorld(true);
		Phase = EPhase::None;
		return false;
	}

	if (!Wrapper.BeginPlayInTestWorld())
	{
		Test->AddError(TEXT("Failed to begin play in camera sensor test world"));
		Wrapper.DestroyTestWorld(true);
		Phase = EPhase::None;
		return false;
	}

	Phase = EPhase::Playing;
	for (int32 i = 0; i < 3; ++i)
	{
		Wrapper.TickTestWorld(0.016f);
	}
	return true;
}

void FScholaCameraSensorTestWorld::Tick(float DeltaSeconds)
{
	if (Phase == EPhase::Playing)
	{
		Wrapper.TickTestWorld(DeltaSeconds);
	}
}

FScholaCameraSensorTestWorld::~FScholaCameraSensorTestWorld()
{
	if (Phase == EPhase::Playing)
	{
		Wrapper.EndPlayInTestWorld();
	}
	if (Phase == EPhase::Playing || Phase == EPhase::Created)
	{
		Wrapper.DestroyTestWorld(true);
	}
}

namespace ScholaCameraSensorTest
{
	void BuildKnownColorBitmap(int32 Width, int32 Height, TArray<FColor>& OutBitmap)
	{
		const int32 PixelCount = Width * Height;
		OutBitmap.SetNum(PixelCount);

		for (int32 H = 0; H < Height; ++H)
		{
			for (int32 W = 0; W < Width; ++W)
			{
				const int32 PixelIndex = H * Width + W;
				OutBitmap[PixelIndex] = FColor(
					static_cast<uint8>(W * Height + H),
					static_cast<uint8>(H * Width + W),
					static_cast<uint8>((W + H) % 256),
					255);
			}
		}
	}

	UTextureRenderTarget2D* CreateInitializedRenderTarget(
		UObject* Outer,
		int32 Width,
		int32 Height,
		ETextureRenderTargetFormat Format)
	{
		UTextureRenderTarget2D* RenderTarget = NewObject<UTextureRenderTarget2D>(Outer);
		RenderTarget->RenderTargetFormat = Format;
		RenderTarget->InitAutoFormat(Width, Height);
		RenderTarget->UpdateResourceImmediate(true);
		return RenderTarget;
	}

	bool FillRenderTargetSolidColor(UWorld* World, UTextureRenderTarget2D* RenderTarget, const FLinearColor& Color)
	{
		if (!World || !RenderTarget)
		{
			return false;
		}

		UKismetRenderingLibrary::ClearRenderTarget2D(World, RenderTarget, Color);
		FlushRendering();
		return true;
	}

	bool FillRenderTargetFromBitmap(
		UWorld* World,
		UTextureRenderTarget2D* RenderTarget,
		const TArray<FColor>& Bitmap,
		int32 Width,
		int32 Height)
	{
		if (!World || !RenderTarget || Bitmap.Num() != Width * Height)
		{
			return false;
		}

		UTexture2D* SourceTexture = UTexture2D::CreateTransient(Width, Height, PF_B8G8R8A8);
		if (!SourceTexture)
		{
			return false;
		}

		SourceTexture->CompressionSettings = TC_EditorIcon;
		SourceTexture->SRGB = false;
		SourceTexture->AddToRoot();

		FTexture2DMipMap& Mip = SourceTexture->GetPlatformData()->Mips[0];
		void* TextureData = Mip.BulkData.Lock(LOCK_READ_WRITE);
		FMemory::Memcpy(TextureData, Bitmap.GetData(), Bitmap.Num() * sizeof(FColor));
		Mip.BulkData.Unlock();
		SourceTexture->UpdateResource();

		UCanvas* Canvas = nullptr;
		FVector2D CanvasSize;
		FDrawToRenderTargetContext Context;
		UKismetRenderingLibrary::BeginDrawCanvasToRenderTarget(World, RenderTarget, Canvas, CanvasSize, Context);
		if (Canvas)
		{
			Canvas->K2_DrawTexture(
				SourceTexture,
				FVector2D::ZeroVector,
				CanvasSize,
				FVector2D::ZeroVector);
		}
		UKismetRenderingLibrary::EndDrawCanvasToRenderTarget(World, Context);
		FlushRendering();

		SourceTexture->RemoveFromRoot();
		return true;
	}

	void FlushRendering()
	{
		FlushRenderingCommands();
	}

	float GetBoxPointChannelValue(
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		int32 PixelY,
		int32 PixelX,
		int32 Width,
		int32 Height)
	{
		const int32 ChannelStride = Width * Height;
		const int32 PixelIndex = PixelY * Width + PixelX;
		return BoxPoint.Values[ChannelIndex * ChannelStride + PixelIndex];
	}

	bool AssertBoxPointShape(
		FAutomationTestBase& Test,
		const FBoxPoint& BoxPoint,
		int32 ExpectedChannels,
		int32 ExpectedHeight,
		int32 ExpectedWidth,
		const TCHAR* Context)
	{
		bool bOk = true;
		bOk &= Test.TestEqual(FString::Printf(TEXT("%s: shape rank"), Context), BoxPoint.Shape.Num(), 3);
		bOk &= Test.TestEqual(FString::Printf(TEXT("%s: channels"), Context), BoxPoint.Shape[0], ExpectedChannels);
		bOk &= Test.TestEqual(FString::Printf(TEXT("%s: height"), Context), BoxPoint.Shape[1], ExpectedHeight);
		bOk &= Test.TestEqual(FString::Printf(TEXT("%s: width"), Context), BoxPoint.Shape[2], ExpectedWidth);
		bOk &= Test.TestEqual(
			FString::Printf(TEXT("%s: value count"), Context),
			BoxPoint.Values.Num(),
			ExpectedChannels * ExpectedHeight * ExpectedWidth);
		return bOk;
	}

	bool AssertBoxPointChannelNear(
		FAutomationTestBase& Test,
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		int32 PixelY,
		int32 PixelX,
		int32 Width,
		int32 Height,
		float ExpectedValue,
		float Tolerance,
		const TCHAR* Context)
	{
		const float Actual = GetBoxPointChannelValue(BoxPoint, ChannelIndex, PixelY, PixelX, Width, Height);
		return Test.TestTrue(
			FString::Printf(
				TEXT("%s: channel %d at (%d,%d) expected ~%.3f got %.3f"),
				Context,
				ChannelIndex,
				PixelX,
				PixelY,
				ExpectedValue,
				Actual),
			FMath::IsNearlyEqual(Actual, ExpectedValue, Tolerance));
	}

	float ComputeBoxPointRegionMean(
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		const FIntRect& Region,
		int32 Width,
		int32 Height)
	{
		const int32 StartX = FMath::Clamp(Region.Min.X, 0, Width);
		const int32 EndX = FMath::Clamp(Region.Max.X, 0, Width);
		const int32 StartY = FMath::Clamp(Region.Min.Y, 0, Height);
		const int32 EndY = FMath::Clamp(Region.Max.Y, 0, Height);

		float Sum = 0.0f;
		int32 Count = 0;
		for (int32 Y = StartY; Y < EndY; ++Y)
		{
			for (int32 X = StartX; X < EndX; ++X)
			{
				Sum += GetBoxPointChannelValue(BoxPoint, ChannelIndex, Y, X, Width, Height);
				++Count;
			}
		}
		return Count > 0 ? Sum / static_cast<float>(Count) : 0.0f;
	}

	static float ComputeRegionMean(
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		int32 StartY,
		int32 EndY,
		int32 StartX,
		int32 EndX,
		int32 Width,
		int32 Height)
	{
		return ComputeBoxPointRegionMean(
			BoxPoint,
			ChannelIndex,
			FIntRect(StartX, StartY, EndX, EndY),
			Width,
			Height);
	}

	bool AssertBoxPointRegionMeanAbove(
		FAutomationTestBase& Test,
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		int32 StartY,
		int32 EndY,
		int32 StartX,
		int32 EndX,
		int32 Width,
		int32 Height,
		float MinMean,
		const TCHAR* Context)
	{
		const float Mean = ComputeRegionMean(BoxPoint, ChannelIndex, StartY, EndY, StartX, EndX, Width, Height);
		return Test.TestTrue(
			FString::Printf(TEXT("%s: region mean %.3f should be above %.3f"), Context, Mean, MinMean),
			Mean > MinMean);
	}

	bool AssertBoxPointRegionMeanBelow(
		FAutomationTestBase& Test,
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		int32 StartY,
		int32 EndY,
		int32 StartX,
		int32 EndX,
		int32 Width,
		int32 Height,
		float MaxMean,
		const TCHAR* Context)
	{
		const float Mean = ComputeRegionMean(BoxPoint, ChannelIndex, StartY, EndY, StartX, EndX, Width, Height);
		return Test.TestTrue(
			FString::Printf(TEXT("%s: region mean %.3f should be below %.3f"), Context, Mean, MaxMean),
			Mean < MaxMean);
	}

	bool AssertBoxPointRegionBrighter(
		FAutomationTestBase& Test,
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		const FIntRect& BrightRegion,
		const FIntRect& DimRegion,
		int32 Width,
		int32 Height,
		float MinDelta,
		const TCHAR* Context)
	{
		const float BrightMean = ComputeBoxPointRegionMean(BoxPoint, ChannelIndex, BrightRegion, Width, Height);
		const float DimMean = ComputeBoxPointRegionMean(BoxPoint, ChannelIndex, DimRegion, Width, Height);
		return Test.TestTrue(
			FString::Printf(
				TEXT("%s: channel %d bright-region mean %.3f should exceed dim-region mean %.3f by >= %.3f"),
				Context,
				ChannelIndex,
				BrightMean,
				DimMean,
				MinDelta),
			(BrightMean - DimMean) >= MinDelta);
	}

	UCameraSensor* SpawnCameraSensor(
		UWorld* World,
		const FVector& Location,
		const FRotator& Rotation,
		int32 RenderTargetWidth,
		int32 RenderTargetHeight,
		ESceneCaptureSource CaptureSource,
		uint8 EnabledChannels,
		bool bCallInitSensor,
		bool bCreateRenderTarget)
	{
		if (!World)
		{
			return nullptr;
		}

		FActorSpawnParameters SpawnParams;
		SpawnParams.ObjectFlags = RF_Transient;
		AActor* Owner = World->SpawnActor<AActor>(AActor::StaticClass(), Location, Rotation, SpawnParams);
		if (!Owner)
		{
			return nullptr;
		}

		USceneComponent* Root = NewObject<USceneComponent>(Owner, TEXT("Root"));
		Root->SetMobility(EComponentMobility::Movable);
		Owner->SetRootComponent(Root);
		Root->RegisterComponent();

		UCameraSensor* Sensor = NewObject<UCameraSensor>(Owner, TEXT("CameraSensor"));
		Sensor->SetupAttachment(Root);
		Sensor->SetRelativeLocation(FVector::ZeroVector);
		Sensor->RegisterComponent();

		Sensor->CaptureSource = CaptureSource;
		Sensor->EnabledChannels = EnabledChannels;
		Sensor->bCaptureEveryFrame = false;

		if (bCreateRenderTarget)
		{
			Sensor->TextureTarget = CreateInitializedRenderTarget(
				Sensor,
				RenderTargetWidth,
				RenderTargetHeight,
				ETextureRenderTargetFormat::RTF_RGBA8);
			Sensor->TextureTarget->bGPUSharedFlag = true;
		}

		if (bCallInitSensor)
		{
			Sensor->InitSensor_Implementation();
		}

		return Sensor;
	}

	UStaticMeshComponent* SpawnColoredCube(
		UWorld* World,
		const FVector& Location,
		const FLinearColor& Color,
		const FVector& Scale,
		UStaticMesh* CubeMesh)
	{
		if (!World)
		{
			return nullptr;
		}

		if (!CubeMesh)
		{
			CubeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
		}
		if (!CubeMesh)
		{
			return nullptr;
		}

		UMaterialInterface* BaseMaterial = LoadObject<UMaterialInterface>(
			nullptr,
			TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
		if (!BaseMaterial)
		{
			return nullptr;
		}

		FActorSpawnParameters SpawnParams;
		SpawnParams.ObjectFlags = RF_Transient;
		AActor* CubeActor = World->SpawnActor<AActor>(AActor::StaticClass(), Location, FRotator::ZeroRotator, SpawnParams);
		if (!CubeActor)
		{
			return nullptr;
		}

		UStaticMeshComponent* MeshComponent = NewObject<UStaticMeshComponent>(CubeActor, TEXT("CubeMesh"));
		MeshComponent->SetStaticMesh(CubeMesh);
		MeshComponent->SetMobility(EComponentMobility::Movable);
		MeshComponent->SetWorldScale3D(Scale);
		CubeActor->SetRootComponent(MeshComponent);
		MeshComponent->RegisterComponent();

		UMaterialInstanceDynamic* DynamicMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, MeshComponent);
		DynamicMaterial->SetVectorParameterValue(TEXT("Color"), Color);
		MeshComponent->SetMaterial(0, DynamicMaterial);

		return MeshComponent;
	}

	void CaptureAndCollect(UCameraSensor* Sensor, FScholaCameraSensorTestWorld& TestWorld, FInstancedStruct& OutObservations)
	{
		if (!Sensor)
		{
			return;
		}

		Sensor->CaptureScene();
		for (int32 i = 0; i < 3; ++i)
		{
			TestWorld.Tick(0.016f);
		}
		FlushRendering();
		Sensor->CollectObservations_Implementation(OutObservations);
	}

	uint8 GetEnabledValidChannels(const UCameraSensor* Sensor)
	{
		if (!Sensor)
		{
			return 0;
		}
		return Sensor->EnabledChannels & ~Sensor->GetInvalidChannels();
	}
}
