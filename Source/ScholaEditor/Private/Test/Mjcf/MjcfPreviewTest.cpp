// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "AssetCompilingManager.h"
#include "AssetImportTask.h"
#include "AssetToolsModule.h"
#include "Components/DirectionalLightComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "ContentStreaming.h"
#include "Engine/Blueprint.h"
#include "Engine/SceneCapture2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "ImageUtils.h"
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "MjcfConversion.h"
#include "PreviewScene.h"
#include "ShaderCompiler.h"
#include "TextureResource.h"
#include "UObject/StrongObjectPtr.h"

#if WITH_DEV_AUTOMATION_TESTS

/**
 * Renders MJCF models for comparing with MuJoCo's renderer. SCHOLA_MJCF_PREVIEW names a file with one model per line:
 * "name;model file;lookat x y z;distance;azimuth;elevation", describing MuJoCo's free camera in MuJoCo's units and axes.
 * Each model is imported and rendered to SCHOLA_MJCF_PREVIEW_OUT/name.png with the same camera and MuJoCo's 45 degree
 * vertical field of view. Needs a renderer, so run it without -nullrhi.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfPreviewTest, "Schola.Mjcf.Preview", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FMjcfPreviewTest::RunTest(const FString& Parameters)
{
	const FString SpecFile = FPlatformMisc::GetEnvironmentVariable(TEXT("SCHOLA_MJCF_PREVIEW"));
	const FString OutDir = FPlatformMisc::GetEnvironmentVariable(TEXT("SCHOLA_MJCF_PREVIEW_OUT"));
	TArray<FString> Lines;
	if (SpecFile.IsEmpty() || OutDir.IsEmpty() || !FFileHelper::LoadFileToStringArray(Lines, *SpecFile))
	{
		AddInfo(TEXT("Set SCHOLA_MJCF_PREVIEW and SCHOLA_MJCF_PREVIEW_OUT to run this test"));
		return true;
	}

	constexpr int32 Width = 1280;
	constexpr int32 Height = 720;
	for (const FString& Line : Lines)
	{
		TArray<FString> Fields;
		Line.ParseIntoArray(Fields, TEXT(";"));
		TArray<FString> LookAtNumbers;
		if (Fields.Num() == 6)
		{
			Fields[2].ParseIntoArrayWS(LookAtNumbers);
		}
		if (LookAtNumbers.Num() != 3)
		{
			AddError(FString::Printf(TEXT("Malformed line: %s"), *Line));
			continue;
		}
		const FVector LookAt(FCString::Atod(*LookAtNumbers[0]), FCString::Atod(*LookAtNumbers[1]), FCString::Atod(*LookAtNumbers[2]));
		const FString& Name = Fields[0];
		const double Distance = FCString::Atod(*Fields[3]);
		const double Azimuth = FMath::DegreesToRadians(FCString::Atod(*Fields[4]));
		const double Elevation = FMath::DegreesToRadians(FCString::Atod(*Fields[5]));

		TStrongObjectPtr<UAssetImportTask> Task(NewObject<UAssetImportTask>());
		Task->Filename = Fields[1];
		Task->DestinationPath = TEXT("/Game/ScholaMjcfPreview") / Name;
		Task->bAutomated = true;
		Task->bReplaceExisting = true;
		Task->bSave = false;
		FAssetToolsModule::GetModule().Get().ImportAssetTasks({ Task.Get() });
		const TArray<UObject*> Imported = Task->GetObjects();
		UBlueprint* Blueprint = Imported.IsEmpty() ? nullptr : Cast<UBlueprint>(Imported[0]);
		if (!Blueprint)
		{
			AddError(FString::Printf(TEXT("%s: nothing was imported from %s"), *Name, *Fields[1]));
			continue;
		}

		// MuJoCo's free camera looks at LookAt from Distance away, turned Azimuth about Z and tilted Elevation above the horizon.
		const FVector Forward(FMath::Cos(Elevation) * FMath::Cos(Azimuth), FMath::Cos(Elevation) * FMath::Sin(Azimuth), FMath::Sin(Elevation));
		const FVector Position = FMjcfConversion::ConvertPosition(LookAt - Distance * Forward);
		const FVector UnrealForward(Forward.X, -Forward.Y, Forward.Z);

		// Most Menagerie scenes light from straight above and add MuJoCo's shadowless headlight, which keeps everything in view lit.
		FPreviewScene Scene(FPreviewScene::ConstructionValues()
				.SetCreatePhysicsScene(false)
				.ShouldSimulatePhysics(false)
				.SetForceMipsResident(true)
				.SetLightBrightness(1.2f));
		Scene.SetLightDirection(FRotator(-89, 0, 0));
		UDirectionalLightComponent* Headlight = NewObject<UDirectionalLightComponent>(GetTransientPackage());
		Headlight->Intensity = 0.8f;
		Headlight->CastShadows = false;
		Headlight->Mobility = EComponentMobility::Movable;
		Scene.AddComponent(Headlight, FTransform(UnrealForward.Rotation()));
		Scene.GetWorld()->SpawnActor<AActor>(Blueprint->GeneratedClass, FTransform::Identity);
		ASceneCapture2D* Capture = Scene.GetWorld()->SpawnActor<ASceneCapture2D>(Position, UnrealForward.Rotation());
		USceneCaptureComponent2D* CaptureComponent = Capture->GetCaptureComponent2D();
		UTextureRenderTarget2D* Target = NewObject<UTextureRenderTarget2D>();
		Target->InitCustomFormat(Width, Height, PF_B8G8R8A8, /*bInForceLinearGamma=*/false);
		Target->UpdateResourceImmediate(true);
		CaptureComponent->TextureTarget = Target;
		CaptureComponent->CaptureSource = SCS_FinalColorLDR;
		CaptureComponent->bCaptureEveryFrame = false;
		CaptureComponent->bCaptureOnMovement = false;
		CaptureComponent->FOVAngle = FMath::RadiansToDegrees(2 * FMath::Atan(FMath::Tan(FMath::DegreesToRadians(22.5)) * Width / Height));
		// One capture has no history for Lumen to accumulate, so use direct light only, at a fixed exposure.
		FPostProcessSettings& PostProcess = CaptureComponent->PostProcessSettings;
		PostProcess.bOverride_DynamicGlobalIlluminationMethod = true;
		PostProcess.DynamicGlobalIlluminationMethod = EDynamicGlobalIlluminationMethod::None;
		PostProcess.bOverride_ReflectionMethod = true;
		PostProcess.ReflectionMethod = EReflectionMethod::ScreenSpace;
		PostProcess.bOverride_AutoExposureMethod = true;
		PostProcess.AutoExposureMethod = AEM_Manual;
		PostProcess.bOverride_AutoExposureApplyPhysicalCameraExposure = true;
		PostProcess.AutoExposureApplyPhysicalCameraExposure = false;
		PostProcess.bOverride_AutoExposureBias = true;
		PostProcess.AutoExposureBias = 0;

		FAssetCompilingManager::Get().FinishAllCompilation();
		if (GShaderCompilingManager)
		{
			GShaderCompilingManager->FinishAllCompilation();
		}
		IStreamingManager::Get().StreamAllResources(5.0f);
		for (int32 Pass = 0; Pass < 4; ++Pass)
		{
			CaptureComponent->CaptureScene();
			FlushRenderingCommands();
		}

		TArray<FColor> Pixels;
		Target->GameThread_GetRenderTargetResource()->ReadPixels(Pixels);
		for (FColor& Pixel : Pixels)
		{
			Pixel.A = 255;
		}
		const FString OutFile = OutDir / Name + TEXT(".png");
		TestTrue(FString::Printf(TEXT("Saved %s"), *OutFile), Pixels.Num() == Width * Height && FImageUtils::SaveImageByExtension(*OutFile, FImageView(Pixels.GetData(), Width, Height)));
	}
	return true;
}

#endif
