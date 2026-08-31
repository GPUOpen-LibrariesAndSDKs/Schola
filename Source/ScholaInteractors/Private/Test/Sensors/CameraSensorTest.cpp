// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/CameraSensorUtils.h"
#include "Engine/TextureRenderTarget2D.h"

#if WITH_DEV_AUTOMATION_TESTS

// These tests exercise channel/observation-space logic only (no pixel readback or scene capture).
// Do not add EAutomationTestFlags::NonNullRHI unless a test reads GPU resources (e.g. CollectObservations).
static UCameraSensor* CreateCameraSensorWithRenderTarget(
	int32 Width,
	int32 Height,
	ETextureRenderTargetFormat Format,
	ESceneCaptureSource CaptureSource,
	uint8 EnabledChannels)
{
	UCameraSensor* Sensor = NewObject<UCameraSensor>();
	UTextureRenderTarget2D* RenderTarget = NewObject<UTextureRenderTarget2D>();
	RenderTarget->RenderTargetFormat = Format;
	// Set dimensions directly; InitAutoFormat would allocate RHI resources unnecessarily here.
	RenderTarget->SizeX = Width;
	RenderTarget->SizeY = Height;

	Sensor->TextureTarget = RenderTarget;
	Sensor->CaptureSource = CaptureSource;
	Sensor->EnabledChannels = EnabledChannels;

	return Sensor;
}

// Test GetInvalidChannels() with various CaptureSource settings

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_FinalColorLDR_Test, "Schola.Sensors.CameraSensor.InvalidChannels.FinalColorLDR", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_FinalColorLDR_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128, 
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_FinalColorLDR,
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// FinalColorLDR should only invalidate the Alpha channel
	TestEqual(TEXT("InvalidChannels should be A only"), InvalidChannels, static_cast<uint8>(EChannels::A));
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_SceneDepth_Test, "Schola.Sensors.CameraSensor.InvalidChannels.SceneDepth", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_SceneDepth_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_SceneDepth,
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// SceneDepth should only allow R channel
	uint8 Expected = static_cast<uint8>(EChannels::G) | static_cast<uint8>(EChannels::B) | static_cast<uint8>(EChannels::A);
	TestEqual(TEXT("InvalidChannels should be G|B|A"), InvalidChannels, Expected);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_SceneColorSceneDepth_Test, "Schola.Sensors.CameraSensor.InvalidChannels.SceneColorSceneDepth", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_SceneColorSceneDepth_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_SceneColorSceneDepth,
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// SceneColorSceneDepth should allow all channels
	TestEqual(TEXT("InvalidChannels should be 0 (all channels valid)"), InvalidChannels, static_cast<uint8>(0));
	
	return true;
}

// Test GetInvalidChannels() with various TextureTarget formats

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_RTF_R8_Test, "Schola.Sensors.CameraSensor.InvalidChannels.RTF_R8", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_RTF_R8_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_R8,
		ESceneCaptureSource::SCS_SceneColorSceneDepth, // Allows all channels from CaptureSource
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// RTF_R8 should only allow R channel
	uint8 Expected = static_cast<uint8>(EChannels::G) | static_cast<uint8>(EChannels::B) | static_cast<uint8>(EChannels::A);
	TestEqual(TEXT("InvalidChannels should be G|B|A for RTF_R8"), InvalidChannels, Expected);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_RTF_RG8_Test, "Schola.Sensors.CameraSensor.InvalidChannels.RTF_RG8", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_RTF_RG8_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RG8,
		ESceneCaptureSource::SCS_SceneColorSceneDepth, // Allows all channels from CaptureSource
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// RTF_RG8 should only allow R and G channels
	uint8 Expected = static_cast<uint8>(EChannels::B) | static_cast<uint8>(EChannels::A);
	TestEqual(TEXT("InvalidChannels should be B|A for RTF_RG8"), InvalidChannels, Expected);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_RTF_R16f_Test, "Schola.Sensors.CameraSensor.InvalidChannels.RTF_R16f", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_RTF_R16f_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_R16f,
		ESceneCaptureSource::SCS_SceneColorSceneDepth,
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// RTF_R16f should only allow R channel
	uint8 Expected = static_cast<uint8>(EChannels::G) | static_cast<uint8>(EChannels::B) | static_cast<uint8>(EChannels::A);
	TestEqual(TEXT("InvalidChannels should be G|B|A for RTF_R16f"), InvalidChannels, Expected);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_RTF_RG16f_Test, "Schola.Sensors.CameraSensor.InvalidChannels.RTF_RG16f", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_RTF_RG16f_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RG16f,
		ESceneCaptureSource::SCS_SceneColorSceneDepth,
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// RTF_RG16f should only allow R and G channels
	uint8 Expected = static_cast<uint8>(EChannels::B) | static_cast<uint8>(EChannels::A);
	TestEqual(TEXT("InvalidChannels should be B|A for RTF_RG16f"), InvalidChannels, Expected);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_RTF_R32f_Test, "Schola.Sensors.CameraSensor.InvalidChannels.RTF_R32f", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_RTF_R32f_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_R32f,
		ESceneCaptureSource::SCS_SceneColorSceneDepth,
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// RTF_R32f should only allow R channel
	uint8 Expected = static_cast<uint8>(EChannels::G) | static_cast<uint8>(EChannels::B) | static_cast<uint8>(EChannels::A);
	TestEqual(TEXT("InvalidChannels should be G|B|A for RTF_R32f"), InvalidChannels, Expected);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_RTF_RG32f_Test, "Schola.Sensors.CameraSensor.InvalidChannels.RTF_RG32f", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_RTF_RG32f_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RG32f,
		ESceneCaptureSource::SCS_SceneColorSceneDepth,
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// RTF_RG32f should only allow R and G channels
	uint8 Expected = static_cast<uint8>(EChannels::B) | static_cast<uint8>(EChannels::A);
	TestEqual(TEXT("InvalidChannels should be B|A for RTF_RG32f"), InvalidChannels, Expected);
	
	return true;
}

// Test combined CaptureSource and TextureTarget restrictions

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_Combined_SceneDepth_R8_Test, "Schola.Sensors.CameraSensor.InvalidChannels.Combined.SceneDepth_R8", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_Combined_SceneDepth_R8_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_R8,
		ESceneCaptureSource::SCS_SceneDepth,
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// Both SceneDepth and RTF_R8 only allow R channel, so G|B|A should be invalid
	uint8 Expected = static_cast<uint8>(EChannels::G) | static_cast<uint8>(EChannels::B) | static_cast<uint8>(EChannels::A);
	TestEqual(TEXT("InvalidChannels should be G|B|A"), InvalidChannels, Expected);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorInvalidChannels_Combined_FinalColorLDR_RG8_Test, "Schola.Sensors.CameraSensor.InvalidChannels.Combined.FinalColorLDR_RG8", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorInvalidChannels_Combined_FinalColorLDR_RG8_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RG8,
		ESceneCaptureSource::SCS_FinalColorLDR,
		15 // All channels enabled
	);
	
	uint8 InvalidChannels = Sensor->GetInvalidChannels();
	
	// FinalColorLDR invalidates A, RTF_RG8 invalidates B|A
	// Combined should be B|A
	uint8 Expected = static_cast<uint8>(EChannels::B) | static_cast<uint8>(EChannels::A);
	TestEqual(TEXT("InvalidChannels should be B|A"), InvalidChannels, Expected);
	
	return true;
}

// Test GetNumChannels() with different EnabledChannels settings

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorNumChannels_AllEnabled_RGBA8_Test, "Schola.Sensors.CameraSensor.NumChannels.AllEnabled_RGBA8", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorNumChannels_AllEnabled_RGBA8_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_SceneColorSceneDepth, // Allows all channels
		15 // All channels enabled (R|G|B|A)
	);
	
	int NumChannels = Sensor->GetNumChannels();
	
	// All 4 channels should be valid
	TestEqual(TEXT("NumChannels should be 4 for RGBA with SceneColorSceneDepth"), NumChannels, 4);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorNumChannels_RGB_RGBA8_FinalColorLDR_Test, "Schola.Sensors.CameraSensor.NumChannels.RGB_RGBA8_FinalColorLDR", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorNumChannels_RGB_RGBA8_FinalColorLDR_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_FinalColorLDR, // Invalidates A
		15 // All channels enabled (R|G|B|A)
	);
	
	int NumChannels = Sensor->GetNumChannels();
	
	// Only RGB channels should be valid (A is invalid for FinalColorLDR)
	TestEqual(TEXT("NumChannels should be 3 for RGB with FinalColorLDR"), NumChannels, 3);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorNumChannels_R_Only_SceneDepth_Test, "Schola.Sensors.CameraSensor.NumChannels.R_Only_SceneDepth", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorNumChannels_R_Only_SceneDepth_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_SceneDepth, // Only R is valid
		15 // All channels enabled (R|G|B|A)
	);
	
	int NumChannels = Sensor->GetNumChannels();
	
	// Only R channel should be valid
	TestEqual(TEXT("NumChannels should be 1 for SceneDepth"), NumChannels, 1);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorNumChannels_RG_Only_RG8_Test, "Schola.Sensors.CameraSensor.NumChannels.RG_Only_RG8", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorNumChannels_RG_Only_RG8_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RG8, // Only R and G are valid
		ESceneCaptureSource::SCS_SceneColorSceneDepth, // Allows all channels
		15 // All channels enabled (R|G|B|A)
	);
	
	int NumChannels = Sensor->GetNumChannels();
	
	// Only R and G channels should be valid
	TestEqual(TEXT("NumChannels should be 2 for RTF_RG8"), NumChannels, 2);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorNumChannels_Selective_Enable_Test, "Schola.Sensors.CameraSensor.NumChannels.SelectiveEnable", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorNumChannels_Selective_Enable_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_SceneColorSceneDepth, // Allows all channels
		static_cast<uint8>(EChannels::R) | static_cast<uint8>(EChannels::B) // Only R and B enabled
	);
	
	int NumChannels = Sensor->GetNumChannels();
	
	// Only R and B channels are enabled
	TestEqual(TEXT("NumChannels should be 2 for R|B enabled"), NumChannels, 2);
	
	return true;
}

// Test GetObservationSpace() dimensions

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorObservationSpace_128x128_RGB_Test, "Schola.Sensors.CameraSensor.ObservationSpace.128x128_RGB", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorObservationSpace_128x128_RGB_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128, 128,
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_FinalColorLDR, // RGB only (no alpha)
		15 // All channels enabled
	);
	
	FInstancedStruct ObservationSpace;
	Sensor->GetObservationSpace_Implementation(ObservationSpace);
	
	TestTrue(TEXT("ObservationSpace should be a BoxSpace"), ObservationSpace.GetScriptStruct() == FBoxSpace::StaticStruct());
	
	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();
	
	// Shape should be [3, 128, 128] for row-major CHW format (3 channels, height, width)
	TestEqual(TEXT("Shape should have 3 dimensions"), Space.Shape.Num(), 3);
	TestEqual(TEXT("Shape[0] should be 3 (channels)"), Space.Shape[0], 3);
	TestEqual(TEXT("Shape[1] should be 128 (height)"), Space.Shape[1], 128);
	TestEqual(TEXT("Shape[2] should be 128 (width)"), Space.Shape[2], 128);
	
	// Total dimensions should be 3 * 128 * 128 = 49152
	TestEqual(TEXT("Total dimensions should be 49152"), Space.Dimensions.Num(), 49152);
	
	// All dimensions should be [0.0, 1.0] for normalized pixel values
	for (int i = 0; i < Space.Dimensions.Num(); i++)
	{
		TestEqual(TEXT("Dimension low should be 0.0"), Space.Dimensions[i].Low, 0.0f);
		TestEqual(TEXT("Dimension high should be 1.0"), Space.Dimensions[i].High, 1.0f);
	}
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorObservationSpace_256x256_RGBA_Test, "Schola.Sensors.CameraSensor.ObservationSpace.256x256_RGBA", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorObservationSpace_256x256_RGBA_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		256, 256,
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_SceneColorSceneDepth, // All channels valid
		15 // All channels enabled
	);
	
	FInstancedStruct ObservationSpace;
	Sensor->GetObservationSpace_Implementation(ObservationSpace);
	
	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();
	
	// Shape should be [4, 256, 256] for row-major CHW format
	TestEqual(TEXT("Shape should have 3 dimensions"), Space.Shape.Num(), 3);
	TestEqual(TEXT("Shape[0] should be 4 (channels)"), Space.Shape[0], 4);
	TestEqual(TEXT("Shape[1] should be 256 (height)"), Space.Shape[1], 256);
	TestEqual(TEXT("Shape[2] should be 256 (width)"), Space.Shape[2], 256);
	
	// Total dimensions should be 4 * 256 * 256 = 262144
	TestEqual(TEXT("Total dimensions should be 262144"), Space.Dimensions.Num(), 262144);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorObservationSpace_64x64_R_Only_Test, "Schola.Sensors.CameraSensor.ObservationSpace.64x64_R_Only", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorObservationSpace_64x64_R_Only_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		64, 64,
		ETextureRenderTargetFormat::RTF_R8, // Only R channel
		ESceneCaptureSource::SCS_SceneDepth, // Only R channel
		15 // All channels enabled
	);
	
	FInstancedStruct ObservationSpace;
	Sensor->GetObservationSpace_Implementation(ObservationSpace);
	
	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();
	
	// Shape should be [1, 64, 64] for row-major CHW format (1 channel)
	TestEqual(TEXT("Shape should have 3 dimensions"), Space.Shape.Num(), 3);
	TestEqual(TEXT("Shape[0] should be 1 (channel)"), Space.Shape[0], 1);
	TestEqual(TEXT("Shape[1] should be 64 (height)"), Space.Shape[1], 64);
	TestEqual(TEXT("Shape[2] should be 64 (width)"), Space.Shape[2], 64);
	
	// Total dimensions should be 1 * 64 * 64 = 4096
	TestEqual(TEXT("Total dimensions should be 4096"), Space.Dimensions.Num(), 4096);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorObservationSpace_512x512_RG_Test, "Schola.Sensors.CameraSensor.ObservationSpace.512x512_RG", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorObservationSpace_512x512_RG_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		512, 512,
		ETextureRenderTargetFormat::RTF_RG8, // Only R and G channels
		ESceneCaptureSource::SCS_SceneColorSceneDepth, // Allows all channels
		15 // All channels enabled
	);
	
	FInstancedStruct ObservationSpace;
	Sensor->GetObservationSpace_Implementation(ObservationSpace);
	
	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();
	
	// Shape should be [2, 512, 512] for row-major CHW format (2 channels)
	TestEqual(TEXT("Shape should have 3 dimensions"), Space.Shape.Num(), 3);
	TestEqual(TEXT("Shape[0] should be 2 (channels)"), Space.Shape[0], 2);
	TestEqual(TEXT("Shape[1] should be 512 (height)"), Space.Shape[1], 512);
	TestEqual(TEXT("Shape[2] should be 512 (width)"), Space.Shape[2], 512);
	
	// Total dimensions should be 2 * 512 * 512 = 524288
	TestEqual(TEXT("Total dimensions should be 524288"), Space.Dimensions.Num(), 524288);
	
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCameraSensorObservationSpace_NonSquare_Test, "Schola.Sensors.CameraSensor.ObservationSpace.NonSquare", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorObservationSpace_NonSquare_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		320, 240, // Non-square dimensions
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_FinalColorLDR, // RGB only
		15 // All channels enabled
	);
	
	FInstancedStruct ObservationSpace;
	Sensor->GetObservationSpace_Implementation(ObservationSpace);
	
	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();
	
	// Shape should be [3, 240, 320] for row-major CHW format
	TestEqual(TEXT("Shape should have 3 dimensions"), Space.Shape.Num(), 3);
	TestEqual(TEXT("Shape[0] should be 3 (channels)"), Space.Shape[0], 3);
	TestEqual(TEXT("Shape[1] should be 240 (height)"), Space.Shape[1], 240);
	TestEqual(TEXT("Shape[2] should be 320 (width)"), Space.Shape[2], 320);
	
	// Total dimensions should be 3 * 240 * 320 = 230400
	TestEqual(TEXT("Total dimensions should be 230400"), Space.Dimensions.Num(), 230400);
	
	return true;
}

// Verify bitmap conversion preserves width/height orientation (no transpose)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorUtils_ConvertBitmap_NoTranspose_Test,
	"Schola.Sensors.CameraSensorUtils.ConvertBitmap.NoTranspose",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FCameraSensorUtils_ConvertBitmap_NoTranspose_Test::RunTest(const FString& Parameters)
{
	constexpr int32 Width = 4;
	constexpr int32 Height = 3;
	const int32 ChannelStride = Width * Height;

	TArray<FColor> Bitmap;
	Bitmap.SetNum(ChannelStride);

	for (int32 H = 0; H < Height; ++H)
	{
		for (int32 W = 0; W < Width; ++W)
		{
			const int32 PixelIndex = H * Width + W;
			// R encodes (w, h); G encodes transposed (h, w) so a width/height swap is detectable.
			Bitmap[PixelIndex] = FColor(
				static_cast<uint8>(W * Height + H),
				static_cast<uint8>(H * Width + W),
				0,
				255);
		}
	}

	FBoxPoint BoxPoint;
	TestTrue(
		TEXT("ConvertBitmapToBoxPoint succeeds"),
		CameraSensorUtils::ConvertBitmapToBoxPoint(
			Bitmap,
			Width,
			Height,
			static_cast<uint8>(EChannels::R) | static_cast<uint8>(EChannels::G),
			BoxPoint));

	TestEqual(TEXT("Shape should have 3 dimensions"), BoxPoint.Shape.Num(), 3);
	TestEqual(TEXT("Shape[0] should be 2 (channels)"), BoxPoint.Shape[0], 2);
	TestEqual(TEXT("Shape[1] should be height"), BoxPoint.Shape[1], Height);
	TestEqual(TEXT("Shape[2] should be width"), BoxPoint.Shape[2], Width);
	TestEqual(TEXT("Values length should match channels * height * width"), BoxPoint.Values.Num(), ChannelStride * 2);

	for (int32 H = 0; H < Height; ++H)
	{
		for (int32 W = 0; W < Width; ++W)
		{
			const int32 PixelIndex = H * Width + W;
			const float ExpectedR = static_cast<float>(W * Height + H) / 255.0f;
			const float ExpectedG = static_cast<float>(H * Width + W) / 255.0f;

			TestEqual(
				FString::Printf(TEXT("R channel at (%d, %d) should not be transposed"), W, H),
				BoxPoint.Values[0 * ChannelStride + PixelIndex],
				ExpectedR);

			TestEqual(
				FString::Printf(TEXT("G channel at (%d, %d) should not be transposed"), W, H),
				BoxPoint.Values[1 * ChannelStride + PixelIndex],
				ExpectedG);
		}
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorUtils_ConvertBitmap_AllChannelMasks_Test,
	"Schola.Sensors.CameraSensorUtils.ConvertBitmap.AllChannelMasks",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FCameraSensorUtils_ConvertBitmap_AllChannelMasks_Test::RunTest(const FString& Parameters)
{
	constexpr int32 Width = 2;
	constexpr int32 Height = 2;
	TArray<FColor> Bitmap;
	Bitmap.Init(FColor(10, 20, 30, 40), Width * Height);

	const uint8 ChannelMasks[] = {
		static_cast<uint8>(EChannels::R),
		static_cast<uint8>(EChannels::G),
		static_cast<uint8>(EChannels::B),
		static_cast<uint8>(EChannels::A),
		static_cast<uint8>(EChannels::R) | static_cast<uint8>(EChannels::G) | static_cast<uint8>(EChannels::B) | static_cast<uint8>(EChannels::A),
	};

	const float ExpectedValues[4] = { 10.0f / 255.0f, 20.0f / 255.0f, 30.0f / 255.0f, 40.0f / 255.0f };

	for (uint8 Mask : ChannelMasks)
	{
		FBoxPoint BoxPoint;
		TestTrue(
			FString::Printf(TEXT("ConvertBitmapToBoxPoint succeeds for mask %d"), Mask),
			CameraSensorUtils::ConvertBitmapToBoxPoint(Bitmap, Width, Height, Mask, BoxPoint));

		int32 ExpectedChannels = 0;
		if (Mask & static_cast<uint8>(EChannels::R)) { ++ExpectedChannels; }
		if (Mask & static_cast<uint8>(EChannels::G)) { ++ExpectedChannels; }
		if (Mask & static_cast<uint8>(EChannels::B)) { ++ExpectedChannels; }
		if (Mask & static_cast<uint8>(EChannels::A)) { ++ExpectedChannels; }

		TestEqual(FString::Printf(TEXT("Channel count for mask %d"), Mask), BoxPoint.Shape[0], ExpectedChannels);
		TestEqual(FString::Printf(TEXT("Value count for mask %d"), Mask), BoxPoint.Values.Num(), ExpectedChannels * Width * Height);

		int32 ChannelIndex = 0;
		if (Mask & static_cast<uint8>(EChannels::R))
		{
			TestEqual(FString::Printf(TEXT("R for mask %d"), Mask), BoxPoint.Values[ChannelIndex * Width * Height], ExpectedValues[0]);
			++ChannelIndex;
		}
		if (Mask & static_cast<uint8>(EChannels::G))
		{
			TestEqual(FString::Printf(TEXT("G for mask %d"), Mask), BoxPoint.Values[ChannelIndex * Width * Height], ExpectedValues[1]);
			++ChannelIndex;
		}
		if (Mask & static_cast<uint8>(EChannels::B))
		{
			TestEqual(FString::Printf(TEXT("B for mask %d"), Mask), BoxPoint.Values[ChannelIndex * Width * Height], ExpectedValues[2]);
			++ChannelIndex;
		}
		if (Mask & static_cast<uint8>(EChannels::A))
		{
			TestEqual(FString::Printf(TEXT("A for mask %d"), Mask), BoxPoint.Values[ChannelIndex * Width * Height], ExpectedValues[3]);
		}
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorUtils_ConvertBitmap_Normalization_Test,
	"Schola.Sensors.CameraSensorUtils.ConvertBitmap.Normalization",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FCameraSensorUtils_ConvertBitmap_Normalization_Test::RunTest(const FString& Parameters)
{
	TArray<FColor> Bitmap = { FColor(0, 127, 255, 255) };

	FBoxPoint BoxPoint;
	TestTrue(
		TEXT("ConvertBitmapToBoxPoint succeeds"),
		CameraSensorUtils::ConvertBitmapToBoxPoint(
			Bitmap,
			1,
			1,
			static_cast<uint8>(EChannels::R) | static_cast<uint8>(EChannels::G) | static_cast<uint8>(EChannels::B),
			BoxPoint));

	TestEqual(TEXT("Zero normalizes to 0"), BoxPoint.Values[0], 0.0f);
	TestEqual(TEXT("127 normalizes correctly"), BoxPoint.Values[1], 127.0f / 255.0f);
	TestEqual(TEXT("255 normalizes to 1"), BoxPoint.Values[2], 1.0f);

	return true;
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorUtils_ConvertBitmap_SizeMismatch_Test,
	"Schola.Sensors.CameraSensorUtils.ConvertBitmap.SizeMismatch",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FCameraSensorUtils_ConvertBitmap_SizeMismatch_Test::RunTest(const FString& Parameters)
{
	AddExpectedError(
		TEXT("Bitmap size"),
		EAutomationExpectedErrorFlags::Contains,
		1);

	TArray<FColor> Bitmap = { FColor(255, 0, 0, 255), FColor(0, 255, 0, 255) };

	FBoxPoint BoxPoint;
	TestFalse(
		TEXT("ConvertBitmapToBoxPoint fails on size mismatch"),
		CameraSensorUtils::ConvertBitmapToBoxPoint(
			Bitmap,
			1,
			1,
			static_cast<uint8>(EChannels::R),
			BoxPoint));

	TestEqual(TEXT("Mismatch leaves values unallocated"), BoxPoint.Values.Num(), 0);
	TestEqual(TEXT("Mismatch leaves shape unallocated"), BoxPoint.Shape.Num(), 0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensor_GenerateId_FinalColorLDR_Test,
	"Schola.Sensors.CameraSensor.GenerateId.FinalColorLDR",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FCameraSensor_GenerateId_FinalColorLDR_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		128,
		128,
		ETextureRenderTargetFormat::RTF_RGBA8,
		ESceneCaptureSource::SCS_FinalColorLDR,
		15);

	const FString Id = Sensor->GenerateId();
	TestTrue(TEXT("Id contains Camera prefix"), Id.Contains(TEXT("Camera")));
	TestTrue(TEXT("Id contains capture source"), Id.Contains(TEXT("SCS_FinalColorLDR")));
	TestTrue(TEXT("Id contains RGB channel suffix"), Id.Contains(TEXT("RGB_W128")));
	TestTrue(TEXT("Id contains height"), Id.Contains(TEXT("H128")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensor_GenerateId_SceneDepthR8_Test,
	"Schola.Sensors.CameraSensor.GenerateId.SceneDepthR8",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FCameraSensor_GenerateId_SceneDepthR8_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = CreateCameraSensorWithRenderTarget(
		64,
		64,
		ETextureRenderTargetFormat::RTF_R8,
		ESceneCaptureSource::SCS_SceneDepth,
		15);

	const FString Id = Sensor->GenerateId();
	TestTrue(TEXT("Id contains R channel suffix before width"), Id.Contains(TEXT("R_W64")));
	TestFalse(TEXT("Id excludes G channel suffix"), Id.Contains(TEXT("G_W64")));
	TestFalse(TEXT("Id excludes B channel suffix"), Id.Contains(TEXT("B_W64")));
	TestFalse(TEXT("Id excludes A channel suffix before width"), Id.Contains(TEXT("A_W64")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensor_InitSensor_ExistingRenderTarget_Test,
	"Schola.Sensors.CameraSensor.InitSensor.ExistingRenderTarget",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FCameraSensor_InitSensor_ExistingRenderTarget_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = NewObject<UCameraSensor>();
	UTextureRenderTarget2D* ExistingTarget = NewObject<UTextureRenderTarget2D>();
	ExistingTarget->SizeX = 256;
	ExistingTarget->SizeY = 128;
	ExistingTarget->bNoFastClear = 1;
	ExistingTarget->bHDR_DEPRECATED = 1;
	Sensor->TextureTarget = ExistingTarget;

	Sensor->InitSensor_Implementation();

	TestEqual(TEXT("Existing render target preserved"), Sensor->TextureTarget.Get(), ExistingTarget);
	TestEqual(TEXT("bNoFastClear cleared"), Sensor->TextureTarget->bNoFastClear, static_cast<uint8>(0));
	TestEqual(TEXT("bHDR_DEPRECATED cleared"), Sensor->TextureTarget->bHDR_DEPRECATED, static_cast<uint8>(0));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensor_GetObservationSpace_NullTarget_Test,
	"Schola.Sensors.CameraSensor.ObservationSpace.NullTarget",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FCameraSensor_GetObservationSpace_NullTarget_Test::RunTest(const FString& Parameters)
{
	AddExpectedError(TEXT("RenderTarget not found. Returning empty observation space."), EAutomationExpectedErrorFlags::Contains, 1);

	UCameraSensor* Sensor = NewObject<UCameraSensor>();
	Sensor->TextureTarget = nullptr;

	FInstancedStruct ObservationSpace;
	Sensor->GetObservationSpace_Implementation(ObservationSpace);

	TestTrue(TEXT("Observation space is a BoxSpace"), ObservationSpace.GetScriptStruct() == FBoxSpace::StaticStruct());

	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();
	TestEqual(TEXT("Empty space has no dimensions"), Space.Dimensions.Num(), 0);
	TestEqual(TEXT("Empty space has no shape"), Space.Shape.Num(), 0);

	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS

