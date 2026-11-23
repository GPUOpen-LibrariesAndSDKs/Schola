// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Mjcf/MjcfMaterialBuilder.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Engine/Texture2D.h"
#include "LogScholaEditor.h"
#include "Materials/MaterialInstanceConstant.h"
#include "Misc/Paths.h"
#include "Mjcf/MjcfAssetUtils.h"
#include "MjcfModel.h"

namespace
{
	const TCHAR* const OpaqueParent = TEXT("/InterchangeAssets/Materials/PBRSurfaceMaterial_MR.PBRSurfaceMaterial_MR");
	const TCHAR* const TranslucentParent = TEXT("/InterchangeAssets/Materials/ThinTranslucentMaterial_MR.ThinTranslucentMaterial_MR");

	float SrgbToLinear(double Value)
	{
		Value = FMath::Clamp(Value, 0.0, 1.0);
		return float(Value <= 0.04045 ? Value / 12.92 : FMath::Pow((Value + 0.055) / 1.055, 2.4));
	}

	FLinearColor ToLinearColor(const FVector4& Rgba)
	{
		return FLinearColor(SrgbToLinear(Rgba.X), SrgbToLinear(Rgba.Y), SrgbToLinear(Rgba.Z), float(FMath::Clamp(Rgba.W, 0.0, 1.0)));
	}

	FColor ToSrgbColor(const FVector& Rgb)
	{
		const auto Channel = [](double Value) { return uint8(FMath::RoundToInt32(FMath::Clamp(Value, 0.0, 1.0) * 255)); };
		return FColor(Channel(Rgb.X), Channel(Rgb.Y), Channel(Rgb.Z));
	}

	/** Names a color in asset names, for example "FF8000FF". */
	FString ToHex(const FVector4& Rgba)
	{
		FColor Color = ToSrgbColor(FVector(Rgba));
		Color.A = uint8(FMath::RoundToInt32(FMath::Clamp(Rgba.W, 0.0, 1.0) * 255));
		return Color.ToHex();
	}

	void SetScalar(UMaterialInstanceConstant& Instance, const TCHAR* Name, double Value)
	{
		Instance.SetScalarParameterValueEditorOnly(FMaterialParameterInfo(FName(Name)), float(Value));
	}
} // namespace

FMjcfMaterialBuilder::FMjcfMaterialBuilder(const FMjcfModel& InModel, const FString& Folder, TArray<UObject*>& OutAssets)
	: Model(InModel)
	, MaterialFolder(Folder / TEXT("Materials"))
	, TextureFolder(Folder / TEXT("Textures"))
	, Assets(OutAssets)
{
}

FVector4 FMjcfMaterialBuilder::GeomColor(const FMjcfModel& Model, const FMjcfGeom& Geom)
{
	if (Geom.Rgba != FMjcfGeom::DefaultRgba)
	{
		return Geom.Rgba;
	}
	if (const FMjcfMaterial* Material = Geom.Material.IsEmpty() ? nullptr : Model.FindMaterial(Geom.Material))
	{
		return Material->Rgba;
	}
	return Geom.Rgba;
}

UMaterialInterface* FMjcfMaterialBuilder::GetMaterial(const FMjcfGeom& Geom)
{
	const FMjcfMaterial* Material = Geom.Material.IsEmpty() ? nullptr : Model.FindMaterial(Geom.Material);
	const FVector4 Rgba = GeomColor(Model, Geom);
	const bool bOwnColor = !Material || Geom.Rgba != FMjcfGeom::DefaultRgba;
	const FString Key = (Material ? Material->Name : FString()) + TEXT("\n") + (bOwnColor ? ToHex(Rgba) : FString());
	if (UMaterialInterface* const* Found = Materials.Find(Key))
	{
		return *Found;
	}

	FString Name = Material ? TEXT("M_") + MjcfAssetUtils::SanitizeName(Material->Name) : FString(TEXT("M_Color"));
	if (bOwnColor)
	{
		Name += TEXT("_") + ToHex(Rgba);
	}
	UMaterialInterface* Created = CreateMaterial(MjcfAssetUtils::ClaimName(Name, UsedNames), Material, Rgba);
	Materials.Add(Key, Created);
	return Created;
}

UMaterialInterface* FMjcfMaterialBuilder::CreateMaterial(const FString& Name, const FMjcfMaterial* Material, const FVector4& Rgba)
{
	const bool bTranslucent = Rgba.W < 1;
	UMaterialInterface* Parent = LoadObject<UMaterialInterface>(nullptr, bTranslucent ? TranslucentParent : OpaqueParent);
	if (!Parent)
	{
		UE_LOGFMT(LogScholaEditor, Error, "Could not create material '{0}' because '{1}' is missing; is the Interchange plugin enabled?", Name, bTranslucent ? TranslucentParent : OpaqueParent);
		return nullptr;
	}
	bool bCreated = false;
	UMaterialInstanceConstant* Instance = MjcfAssetUtils::FindOrCreateAsset<UMaterialInstanceConstant>(MaterialFolder, Name, bCreated);
	if (!Instance)
	{
		return nullptr;
	}

	// MuJoCo's lighting model is Phong-like; these are approximations of its specular and shininess in Unreal's PBR terms.
	const FMjcfMaterial Defaults;
	const FMjcfMaterial& Source = Material ? *Material : Defaults;
	const FLinearColor Color = ToLinearColor(Rgba);
	const double Roughness = FMath::Lerp(1.0 - 0.8 * Source.Shininess, 0.1, FMath::Clamp(Source.Reflectance, 0.0, 1.0));

	Instance->PreEditChange(nullptr);
	Instance->SetParentEditorOnly(Parent);
	Instance->ClearParameterValuesEditorOnly();
	Instance->SetVectorParameterValueEditorOnly(FMaterialParameterInfo(TEXT("BaseColor")), Color);
	SetScalar(*Instance, TEXT("Metallic"), 0);
	SetScalar(*Instance, TEXT("Roughness"), Roughness);
	SetScalar(*Instance, TEXT("Specular"), FMath::Clamp(Source.Specular, 0.0, 1.0));
	if (Source.Emission > 0)
	{
		Instance->SetVectorParameterValueEditorOnly(FMaterialParameterInfo(TEXT("EmissiveColor")), Color * float(Source.Emission));
	}
	if (bTranslucent)
	{
		SetScalar(*Instance, TEXT("Opacity"), Color.A);
	}
	if (UTexture2D* Texture = Source.Texture.IsEmpty() ? nullptr : GetTexture(Source.Texture))
	{
		Instance->SetTextureParameterValueEditorOnly(FMaterialParameterInfo(TEXT("BaseColorMap")), Texture);
		SetScalar(*Instance, TEXT("BaseColorMapWeight"), 1);
	}
	Instance->PostEditChange();
	Instance->MarkPackageDirty();
	if (bCreated)
	{
		FAssetRegistryModule::AssetCreated(Instance);
	}
	Assets.AddUnique(Instance);
	return Instance;
}

UTexture2D* FMjcfMaterialBuilder::GetTexture(const FString& Name)
{
	if (UTexture2D* const* Found = Textures.Find(Name))
	{
		return *Found;
	}

	UTexture2D* Texture = nullptr;
	if (const FMjcfTexture* Source = Model.FindTexture(Name))
	{
		const FString AssetName = MjcfAssetUtils::ClaimName(TEXT("T_") + MjcfAssetUtils::SanitizeName(Name), UsedNames);
		if (Source->Builtin != EMjcfTextureBuiltin::None)
		{
			Texture = CreateBuiltinTexture(*Source, AssetName);
		}
		else if (Source->Type == EMjcfTextureType::TwoD)
		{
			Texture = ImportTextureFile(*Source, AssetName);
		}
		else
		{
			UE_LOGFMT(LogScholaEditor, Warning, "Texture \"{0}\" is a cube or skybox texture read from a file, which isn't supported yet; materials using it are drawn without it", Name);
		}
	}
	Textures.Add(Name, Texture);
	if (Texture)
	{
		Assets.AddUnique(Texture);
	}
	return Texture;
}

UTexture2D* FMjcfMaterialBuilder::CreateBuiltinTexture(const FMjcfTexture& Source, const FString& AssetName)
{
	// Unreal only generates mipmaps for power-of-two textures, and repeated patterns shimmer in the distance without them.
	const int32 Requested = FMath::Max3(Source.Width, Source.Height, 1);
	const int32 Size = FMath::Clamp(int32(FMath::RoundUpToPowerOfTwo(uint32(Requested))), 64, 2048);
	const int32 MarkWidth = FMath::Max(1, FMath::RoundToInt32(float(Size) / Requested));
	const FColor Mark = ToSrgbColor(Source.MarkRgb);

	TArray<FColor> Pixels;
	Pixels.SetNumUninitialized(Size * Size);
	for (int32 Y = 0; Y < Size; ++Y)
	{
		for (int32 X = 0; X < Size; ++X)
		{
			FVector Rgb = Source.Rgb1;
			if (Source.Builtin == EMjcfTextureBuiltin::Checker)
			{
				Rgb = ((X < Size / 2) == (Y < Size / 2)) ? Source.Rgb1 : Source.Rgb2;
			}
			else if (Source.Builtin == EMjcfTextureBuiltin::Gradient)
			{
				const double Distance = FVector2D(X + 0.5 - Size / 2.0, Y + 0.5 - Size / 2.0).Size() / (Size / 2.0);
				Rgb = FMath::Lerp(Source.Rgb1, Source.Rgb2, FMath::Min(Distance, 1.0));
			}
			const bool bEdge = X < MarkWidth || Y < MarkWidth || X >= Size - MarkWidth || Y >= Size - MarkWidth;
			const bool bCross = FMath::Abs(2 * X + 1 - Size) < 2 * MarkWidth || FMath::Abs(2 * Y + 1 - Size) < 2 * MarkWidth;
			const bool bMarked = (Source.Mark == EMjcfTextureMark::Edge && bEdge) || (Source.Mark == EMjcfTextureMark::Cross && bCross);
			Pixels[Y * Size + X] = bMarked ? Mark : ToSrgbColor(Rgb);
		}
	}

	bool bCreated = false;
	UTexture2D* Texture = MjcfAssetUtils::FindOrCreateAsset<UTexture2D>(TextureFolder, AssetName, bCreated);
	if (!Texture)
	{
		return nullptr;
	}
	Texture->PreEditChange(nullptr);
	Texture->Source.Init(Size, Size, 1, 1, TSF_BGRA8, reinterpret_cast<const uint8*>(Pixels.GetData()));
	Texture->SRGB = true;
	Texture->LODGroup = TEXTUREGROUP_World;
	Texture->PostEditChange();
	Texture->MarkPackageDirty();
	if (bCreated)
	{
		FAssetRegistryModule::AssetCreated(Texture);
	}
	return Texture;
}

UTexture2D* FMjcfMaterialBuilder::ImportTextureFile(const FMjcfTexture& Source, const FString& AssetName)
{
	if (!FPaths::FileExists(Source.File))
	{
		UE_LOGFMT(LogScholaEditor, Error, "Could not import texture \"{0}\" because '{1}' does not exist", Source.Name, Source.File);
		return nullptr;
	}
	UTexture2D* Texture = MjcfAssetUtils::ImportWithInterchange<UTexture2D>(Source.File, TextureFolder, AssetName);
	if (!Texture)
	{
		UE_LOGFMT(LogScholaEditor, Error, "Could not import texture \"{0}\" because Interchange created no texture from '{1}'", Source.Name, Source.File);
	}
	return Texture;
}
