// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Test/Sensors/CameraSensorTestHelpers.h"

#include "Sensors/CameraSensor.h"
#include "Sensors/CameraSensorUtils.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Points/BoxPoint.h"

#if WITH_DEV_AUTOMATION_TESTS

using namespace ScholaCameraSensorTest;

static constexpr float GColorTolerance = 0.02f;

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorReadback_SolidRedBase_Test,
	"Schola.Sensors.CameraSensor.Readback.SolidRed.Base",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)

bool FCameraSensorReadback_SolidRedBase_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	UWorld* World = TestWorld.GetWorld();
	constexpr int32 Width = 8;
	constexpr int32 Height = 8;

	UTextureRenderTarget2D* RenderTarget = CreateInitializedRenderTarget(GetTransientPackage(), Width, Height);
	TestTrue(TEXT("Render target created"), RenderTarget != nullptr);
	if (!RenderTarget || !FillRenderTargetSolidColor(World, RenderTarget, FLinearColor(1.0f, 0.0f, 0.0f, 1.0f)))
	{
		return false;
	}

	const uint8 EnabledChannels =
		static_cast<uint8>(EChannels::R) | static_cast<uint8>(EChannels::G) | static_cast<uint8>(EChannels::B);

	FBoxPoint BoxPoint;
	TestTrue(
		TEXT("ReadRenderTargetToBoxPoint succeeds"),
		FCameraSensorUtils::ReadRenderTargetToBoxPoint(RenderTarget, EnabledChannels, BoxPoint));

	AssertBoxPointShape(*this, BoxPoint, 3, Height, Width, TEXT("SolidRed.Base"));
	AssertBoxPointChannelNear(*this, BoxPoint, 0, Height / 2, Width / 2, Width, Height, 1.0f, GColorTolerance, TEXT("SolidRed.Base R"));
	AssertBoxPointChannelNear(*this, BoxPoint, 1, Height / 2, Width / 2, Width, Height, 0.0f, GColorTolerance, TEXT("SolidRed.Base G"));
	AssertBoxPointChannelNear(*this, BoxPoint, 2, Height / 2, Width / 2, Width, Height, 0.0f, GColorTolerance, TEXT("SolidRed.Base B"));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorReadback_SolidRedWithAlpha_Test,
	"Schola.Sensors.CameraSensor.Readback.SolidRed.WithAlpha",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)

bool FCameraSensorReadback_SolidRedWithAlpha_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	constexpr int32 Width = 8;
	constexpr int32 Height = 8;

	UTextureRenderTarget2D* RenderTarget = CreateInitializedRenderTarget(GetTransientPackage(), Width, Height);
	if (!RenderTarget || !FillRenderTargetSolidColor(TestWorld.GetWorld(), RenderTarget, FLinearColor(1.0f, 0.0f, 0.0f, 1.0f)))
	{
		return false;
	}

	const uint8 EnabledChannels = 15;

	FBoxPoint BoxPoint;
	TestTrue(
		TEXT("ReadRenderTargetToBoxPoint succeeds"),
		FCameraSensorUtils::ReadRenderTargetToBoxPoint(RenderTarget, EnabledChannels, BoxPoint));

	AssertBoxPointShape(*this, BoxPoint, 4, Height, Width, TEXT("SolidRed.WithAlpha"));
	AssertBoxPointChannelNear(*this, BoxPoint, 3, Height / 2, Width / 2, Width, Height, 1.0f, GColorTolerance, TEXT("SolidRed.WithAlpha A"));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorReadback_ViaCollectObservations_Test,
	"Schola.Sensors.CameraSensor.Readback.ViaCollectObservations",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)

bool FCameraSensorReadback_ViaCollectObservations_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	constexpr int32 Width = 8;
	constexpr int32 Height = 8;

	UCameraSensor* Sensor = NewObject<UCameraSensor>(GetTransientPackage());
	Sensor->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
	Sensor->EnabledChannels = 15;
	Sensor->TextureTarget = CreateInitializedRenderTarget(Sensor, Width, Height);

	if (!FillRenderTargetSolidColor(TestWorld.GetWorld(), Sensor->TextureTarget, FLinearColor(0.0f, 1.0f, 0.0f, 1.0f)))
	{
		return false;
	}

	const uint8 EnabledValidChannels = GetEnabledValidChannels(Sensor);

	FBoxPoint DirectBoxPoint;
	TestTrue(
		TEXT("Direct readback succeeds"),
		FCameraSensorUtils::ReadRenderTargetToBoxPoint(Sensor->TextureTarget, EnabledValidChannels, DirectBoxPoint));

	FInstancedStruct Observations;
	Sensor->CollectObservations_Implementation(Observations);
	TestTrue(TEXT("CollectObservations returns BoxPoint"), Observations.GetScriptStruct() == FBoxPoint::StaticStruct());

	const FBoxPoint& CollectedBoxPoint = Observations.Get<FBoxPoint>();
	TestEqual(TEXT("Collected shape matches direct readback"), CollectedBoxPoint.Shape, DirectBoxPoint.Shape);
	TestEqual(TEXT("Collected values match direct readback"), CollectedBoxPoint.Values, DirectBoxPoint.Values);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorReadback_ChannelMaskROnly_Test,
	"Schola.Sensors.CameraSensor.Readback.ChannelMask.R_Only",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)

bool FCameraSensorReadback_ChannelMaskROnly_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	constexpr int32 Width = 4;
	constexpr int32 Height = 3;

	TArray<FColor> Bitmap;
	BuildKnownColorBitmap(Width, Height, Bitmap);

	UTextureRenderTarget2D* RenderTarget = CreateInitializedRenderTarget(GetTransientPackage(), Width, Height);
	if (!FillRenderTargetFromBitmap(TestWorld.GetWorld(), RenderTarget, Bitmap, Width, Height))
	{
		return false;
	}

	const uint8 EnabledChannels = static_cast<uint8>(EChannels::R);

	FBoxPoint BoxPoint;
	TestTrue(
		TEXT("ReadRenderTargetToBoxPoint succeeds"),
		FCameraSensorUtils::ReadRenderTargetToBoxPoint(RenderTarget, EnabledChannels, BoxPoint));

	AssertBoxPointShape(*this, BoxPoint, 1, Height, Width, TEXT("ChannelMask.R_Only"));

	for (int32 H = 0; H < Height; ++H)
	{
		for (int32 W = 0; W < Width; ++W)
		{
			const int32 PixelIndex = H * Width + W;
			const float ExpectedR = static_cast<float>(Bitmap[PixelIndex].R) / 255.0f;
			TestEqual(
				FString::Printf(TEXT("R channel at (%d, %d)"), W, H),
				BoxPoint.Values[PixelIndex],
				ExpectedR);
		}
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorReadback_ChannelMaskRGOnly_Test,
	"Schola.Sensors.CameraSensor.Readback.ChannelMask.RG_Only",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)

bool FCameraSensorReadback_ChannelMaskRGOnly_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	constexpr int32 Width = 4;
	constexpr int32 Height = 3;

	TArray<FColor> Bitmap;
	BuildKnownColorBitmap(Width, Height, Bitmap);

	UTextureRenderTarget2D* RenderTarget = CreateInitializedRenderTarget(
		GetTransientPackage(),
		Width,
		Height,
		ETextureRenderTargetFormat::RTF_RG8);

	if (!FillRenderTargetFromBitmap(TestWorld.GetWorld(), RenderTarget, Bitmap, Width, Height))
	{
		return false;
	}

	const uint8 EnabledChannels = static_cast<uint8>(EChannels::R) | static_cast<uint8>(EChannels::G);

	FBoxPoint BoxPoint;
	TestTrue(
		TEXT("ReadRenderTargetToBoxPoint succeeds"),
		FCameraSensorUtils::ReadRenderTargetToBoxPoint(RenderTarget, EnabledChannels, BoxPoint));

	AssertBoxPointShape(*this, BoxPoint, 2, Height, Width, TEXT("ChannelMask.RG_Only"));

	const int32 ChannelStride = Width * Height;
	for (int32 H = 0; H < Height; ++H)
	{
		for (int32 W = 0; W < Width; ++W)
		{
			const int32 PixelIndex = H * Width + W;
			const float ExpectedR = static_cast<float>(Bitmap[PixelIndex].R) / 255.0f;
			const float ExpectedG = static_cast<float>(Bitmap[PixelIndex].G) / 255.0f;
			TestEqual(
				FString::Printf(TEXT("R channel at (%d, %d)"), W, H),
				BoxPoint.Values[0 * ChannelStride + PixelIndex],
				ExpectedR);
			TestEqual(
				FString::Printf(TEXT("G channel at (%d, %d)"), W, H),
				BoxPoint.Values[1 * ChannelStride + PixelIndex],
				ExpectedG);
		}
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorReadback_NonSquare_Test,
	"Schola.Sensors.CameraSensor.Readback.NonSquare",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)
bool FCameraSensorReadback_NonSquare_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	constexpr int32 Width = 320;
	constexpr int32 Height = 240;

	UTextureRenderTarget2D* RenderTarget = CreateInitializedRenderTarget(GetTransientPackage(), Width, Height);
	if (!FillRenderTargetSolidColor(TestWorld.GetWorld(), RenderTarget, FLinearColor(0.2f, 0.4f, 0.6f, 1.0f)))
	{
		return false;
	}

	const uint8 EnabledChannels =
		static_cast<uint8>(EChannels::R) | static_cast<uint8>(EChannels::G) | static_cast<uint8>(EChannels::B);

	FBoxPoint BoxPoint;
	TestTrue(
		TEXT("ReadRenderTargetToBoxPoint succeeds"),
		FCameraSensorUtils::ReadRenderTargetToBoxPoint(RenderTarget, EnabledChannels, BoxPoint));

	AssertBoxPointShape(*this, BoxPoint, 3, Height, Width, TEXT("NonSquare"));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorReadback_NullTarget_Test,
	"Schola.Sensors.CameraSensor.Readback.NullTarget",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)
bool FCameraSensorReadback_NullTarget_Test::RunTest(const FString& Parameters)
{
	AddExpectedError(TEXT("RenderTarget not found. Not collecting Observations."), EAutomationExpectedErrorFlags::Contains, 1);
	AddExpectedError(TEXT("TextureTarget is null."), EAutomationExpectedErrorFlags::Contains, 1);

	UCameraSensor* Sensor = NewObject<UCameraSensor>(GetTransientPackage());
	Sensor->TextureTarget = nullptr;

	FInstancedStruct Observations;
	Sensor->CollectObservations_Implementation(Observations);

	FBoxPoint BoxPoint;
	TestFalse(
		TEXT("ReadRenderTargetToBoxPoint fails for null target"),
		FCameraSensorUtils::ReadRenderTargetToBoxPoint(nullptr, 15, BoxPoint));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorReadback_UninitializedResource_Test,
	"Schola.Sensors.CameraSensor.Readback.UninitializedResource",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)
bool FCameraSensorReadback_UninitializedResource_Test::RunTest(const FString& Parameters)
{
	AddExpectedError(TEXT("Render target resource is null."), EAutomationExpectedErrorFlags::Contains, 1);

	UTextureRenderTarget2D* RenderTarget = NewObject<UTextureRenderTarget2D>(GetTransientPackage());
	RenderTarget->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;
	RenderTarget->SizeX = 8;
	RenderTarget->SizeY = 8;

	FBoxPoint BoxPoint;
	TestFalse(
		TEXT("ReadRenderTargetToBoxPoint fails without initialized resource"),
		FCameraSensorUtils::ReadRenderTargetToBoxPoint(RenderTarget, 15, BoxPoint));

	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
