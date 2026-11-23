// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Mjcf/MjcfBlueprintBuilder.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/Blueprint.h"
#include "Engine/BlueprintGeneratedClass.h"
#include "Engine/CollisionProfile.h"
#include "Engine/SCS_Node.h"
#include "Engine/SimpleConstructionScript.h"
#include "Engine/StaticMesh.h"
#include "GameFramework/Actor.h"
#include "Kismet2/BlueprintEditorUtils.h"
#include "Kismet2/KismetEditorUtilities.h"
#include "LogScholaEditor.h"
#include "Mjcf/MjcfAssetUtils.h"
#include "Mjcf/MjcfMaterialBuilder.h"
#include "Mjcf/MjcfMeshBuilder.h"
#include "MjcfBodyComponent.h"
#include "MjcfConversion.h"
#include "MjcfModel.h"

namespace
{
	/** How far infinite planes (size 0) extend each way, in meters. */
	constexpr double InfinitePlaneHalfSize = 50.0;

	const TCHAR* TypeName(EMjcfGeomType Type)
	{
		static const TCHAR* const Names[] = { TEXT("plane"), TEXT("hfield"), TEXT("sphere"), TEXT("capsule"), TEXT("ellipsoid"), TEXT("cylinder"), TEXT("box"), TEXT("mesh"), TEXT("sdf") };
		return Names[int32(Type)];
	}

	/** Returns Name with every character that isn't a letter, digit, or underscore replaced, so it works as a component variable name. */
	FString VariableName(const FString& Name)
	{
		FString Result = Name;
		for (TCHAR& Char : Result)
		{
			if (!FChar::IsAlnum(Char) && Char != TEXT('_'))
			{
				Char = TEXT('_');
			}
		}
		return Result;
	}

	FTransform ToUnreal(const FVector& Position, const FQuat& Rotation)
	{
		return FTransform(FMjcfConversion::ConvertRotation(Rotation), FMjcfConversion::ConvertPosition(Position));
	}
} // namespace

FMjcfBlueprintBuilder::FMjcfBlueprintBuilder(const FMjcfModel& InModel, const TMap<FString, UStaticMesh*>& InMeshes, FMjcfMaterialBuilder& InMaterials,
	const FString& InFolder, TSet<FString>& InUsedNames, TArray<UObject*>& OutAssets)
	: Model(InModel)
	, Meshes(InMeshes)
	, Materials(InMaterials)
	, Folder(InFolder)
	, UsedNames(InUsedNames)
	, Assets(OutAssets)
{
	Cube = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
	Sphere = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	Cylinder = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
}

UBlueprint* FMjcfBlueprintBuilder::Build(UObject* Package, FName Name)
{
	Blueprint = FindObject<UBlueprint>(Package, *Name.ToString());
	const bool bCreated = Blueprint == nullptr;
	if (Blueprint)
	{
		// Children were added after their parents, so removing in reverse removes each child before its parent.
		USimpleConstructionScript* ConstructionScript = Blueprint->SimpleConstructionScript;
		TArray<USCS_Node*> OldNodes = ConstructionScript->GetAllNodes();
		for (int32 Index = OldNodes.Num() - 1; Index >= 0; --Index)
		{
			if (USCS_Node* Parent = ConstructionScript->FindParentNode(OldNodes[Index]))
			{
				Parent->RemoveChildNode(OldNodes[Index]);
			}
			else
			{
				ConstructionScript->RemoveNode(OldNodes[Index], /*bValidateSceneRootNodes=*/false);
			}
		}
		// Refreshes the skeleton class, which otherwise keeps the removed components' names taken.
		FBlueprintEditorUtils::MarkBlueprintAsStructurallyModified(Blueprint);
	}
	else
	{
		if (UObject* Existing = StaticFindObject(UObject::StaticClass(), Package, *Name.ToString()))
		{
			UE_LOGFMT(LogScholaEditor, Error, "Could not create Blueprint '{0}' because a {1} already has that name", Existing->GetPathName(), Existing->GetClass()->GetName());
			return nullptr;
		}
		Blueprint = FKismetEditorUtilities::CreateBlueprint(AActor::StaticClass(), Package, Name, BPTYPE_Normal, UBlueprint::StaticClass(), UBlueprintGeneratedClass::StaticClass());
	}

	AddBody(Model.WorldBody, nullptr);
	Blueprint->SimpleConstructionScript->ValidateSceneRootNodes();
	FKismetEditorUtilities::CompileBlueprint(Blueprint);
	Blueprint->MarkPackageDirty();
	if (bCreated)
	{
		FAssetRegistryModule::AssetCreated(Blueprint);
	}
	return Blueprint;
}

void FMjcfBlueprintBuilder::AddBody(const FMjcfBody& Body, USCS_Node* ParentNode)
{
	const bool bWorld = ParentNode == nullptr;
	const FString Label = bWorld ? FString(TEXT("WorldBody")) : !Body.Name.IsEmpty() ? Body.Name : FString::Printf(TEXT("Body%d"), ++UnnamedBodies);
	USCS_Node* Node = AddNode(UMjcfBodyComponent::StaticClass(), Label, ParentNode, ToUnreal(Body.Position, Body.Rotation));
	CastChecked<UMjcfBodyComponent>(Node->ComponentTemplate)->BodyName = Body.Name;

	for (const FMjcfGeom& Geom : Body.Geoms)
	{
		AddGeom(Geom, Label, *Node);
	}
	for (const FMjcfBody& Child : Body.Bodies)
	{
		AddBody(Child, Node);
	}
}

void FMjcfBlueprintBuilder::AddGeom(const FMjcfGeom& Geom, const FString& BodyLabel, USCS_Node& BodyNode)
{
	// MuJoCo only draws groups 0 to 2 by default; collision-only geoms usually go in group 3.
	if (Geom.Group < 0 || Geom.Group > 2 || FMjcfMaterialBuilder::GeomColor(Model, Geom).W <= 0)
	{
		return;
	}
	const FString Label = !Geom.Name.IsEmpty() ? Geom.Name
		: BodyLabel + TEXT("_") + (Geom.Type == EMjcfGeomType::Mesh ? Geom.Mesh : FString(TypeName(Geom.Type)));
	if (Geom.Type != EMjcfGeomType::Mesh && !Geom.Mesh.IsEmpty())
	{
		UE_LOGFMT(LogScholaEditor, Warning, "Skipped geom \"{0}\": a {1} fitted to mesh \"{2}\" isn't supported yet", Label, TypeName(Geom.Type), Geom.Mesh);
		return;
	}

	const FTransform Placement = ToUnreal(Geom.Position, Geom.Rotation);
	const FVector Size = Geom.Size * FMjcfConversion::CentimetersPerMeter;
	switch (Geom.Type)
	{
		case EMjcfGeomType::Mesh:
			if (UStaticMesh* const* Mesh = Meshes.Find(Geom.Mesh))
			{
				AddShape(BodyNode, Label, *Mesh, Placement, Materials.GetMaterial(Geom));
			}
			else
			{
				UE_LOGFMT(LogScholaEditor, Warning, "Skipped geom \"{0}\" because its mesh \"{1}\" wasn't imported", Label, Geom.Mesh);
			}
			break;
		case EMjcfGeomType::Sphere:
			AddFittedShape(BodyNode, Label, Sphere, FVector(Size.X), Placement, Materials.GetMaterial(Geom));
			break;
		case EMjcfGeomType::Ellipsoid:
			AddFittedShape(BodyNode, Label, Sphere, Size, Placement, Materials.GetMaterial(Geom));
			break;
		case EMjcfGeomType::Box:
			AddFittedShape(BodyNode, Label, Cube, Size, Placement, Materials.GetMaterial(Geom));
			break;
		case EMjcfGeomType::Cylinder:
			AddFittedShape(BodyNode, Label, Cylinder, FVector(Size.X, Size.X, Size.Y), Placement, Materials.GetMaterial(Geom));
			break;
		case EMjcfGeomType::Capsule:
		{
			UMaterialInterface* Material = Materials.GetMaterial(Geom);
			USCS_Node* Capsule = AddNode(USceneComponent::StaticClass(), Label, &BodyNode, Placement);
			AddFittedShape(*Capsule, Label + TEXT("_Cylinder"), Cylinder, FVector(Size.X, Size.X, Size.Y), FTransform::Identity, Material);
			AddFittedShape(*Capsule, Label + TEXT("_Top"), Sphere, FVector(Size.X), FTransform(FVector(0, 0, Size.Y)), Material);
			AddFittedShape(*Capsule, Label + TEXT("_Bottom"), Sphere, FVector(Size.X), FTransform(FVector(0, 0, -Size.Y)), Material);
			break;
		}
		case EMjcfGeomType::Plane:
			if (UStaticMesh* Plane = GetPlaneMesh(Geom, Label))
			{
				AddShape(BodyNode, Label, Plane, Placement, Materials.GetMaterial(Geom));
			}
			break;
		default:
			if (!ReportedTypes.Contains(int32(Geom.Type)))
			{
				ReportedTypes.Add(int32(Geom.Type));
				UE_LOGFMT(LogScholaEditor, Warning, "Skipped {0} geoms such as \"{1}\": they aren't supported yet (reported once)", TypeName(Geom.Type), Label);
			}
			break;
	}
}

USCS_Node* FMjcfBlueprintBuilder::AddNode(UClass* Class, const FString& Name, USCS_Node* ParentNode, const FTransform& Transform)
{
	USimpleConstructionScript* ConstructionScript = Blueprint->SimpleConstructionScript;
	USCS_Node* Node = ConstructionScript->CreateNode(Class, FName(*VariableName(Name)));
	USceneComponent* Template = CastChecked<USceneComponent>(Node->ComponentTemplate);
	Template->SetRelativeLocation_Direct(Transform.GetLocation());
	Template->SetRelativeRotation_Direct(Transform.Rotator());
	Template->SetRelativeScale3D_Direct(Transform.GetScale3D());
	Template->Mobility = EComponentMobility::Movable;
	if (ParentNode)
	{
		ParentNode->AddChildNode(Node);
	}
	else
	{
		ConstructionScript->AddNode(Node);
	}
	return Node;
}

void FMjcfBlueprintBuilder::AddShape(USCS_Node& ParentNode, const FString& Name, UStaticMesh* Mesh, const FTransform& Transform, UMaterialInterface* Material)
{
	USCS_Node* Node = AddNode(UStaticMeshComponent::StaticClass(), Name, &ParentNode, Transform);
	UStaticMeshComponent* Template = CastChecked<UStaticMeshComponent>(Node->ComponentTemplate);
	Template->SetStaticMesh(Mesh);
	if (Material)
	{
		for (int32 Slot = 0; Slot < Mesh->GetStaticMaterials().Num(); ++Slot)
		{
			Template->SetMaterial(Slot, Material);
		}
	}
	Template->SetCollisionProfileName(UCollisionProfile::NoCollision_ProfileName);
	Template->SetGenerateOverlapEvents(false);
	Template->SetCanEverAffectNavigation(false);
}

void FMjcfBlueprintBuilder::AddFittedShape(USCS_Node& ParentNode, const FString& Name, UStaticMesh* Shape, const FVector& HalfExtent, const FTransform& Transform,
	UMaterialInterface* Material)
{
	if (!Shape)
	{
		UE_LOGFMT(LogScholaEditor, Error, "Skipped geom \"{0}\" because the engine's basic shapes are missing", Name);
		return;
	}
	const FBox Bounds = Shape->GetBoundingBox();
	const FVector Scale = 2 * HalfExtent / Bounds.GetSize();
	const FTransform Fit(FQuat::Identity, -Bounds.GetCenter() * Scale, Scale);
	AddShape(ParentNode, Name, Shape, Fit * Transform, Material);
}

UStaticMesh* FMjcfBlueprintBuilder::GetPlaneMesh(const FMjcfGeom& Geom, const FString& Label)
{
	const FVector2D HalfSize(Geom.Size.X > 0 ? Geom.Size.X : InfinitePlaneHalfSize, Geom.Size.Y > 0 ? Geom.Size.Y : InfinitePlaneHalfSize);
	FVector2D UVRepeat(1, 1);
	if (const FMjcfMaterial* Material = Geom.Material.IsEmpty() ? nullptr : Model.FindMaterial(Geom.Material))
	{
		// MuJoCo repeats uniform textures TexRepeat times per two meters, as measured against its renderer.
		UVRepeat = Material->bTexUniform ? Material->TexRepeat * HalfSize : Material->TexRepeat;
	}

	const FString Key = FString::Printf(TEXT("%.9g %.9g %.9g %.9g"), HalfSize.X, HalfSize.Y, UVRepeat.X, UVRepeat.Y);
	if (UStaticMesh* const* Found = Planes.Find(Key))
	{
		return *Found;
	}
	const FString Name = MjcfAssetUtils::ClaimName(TEXT("Plane_") + MjcfAssetUtils::SanitizeName(Label), UsedNames);
	bool bCreated = false;
	UStaticMesh* Plane = MjcfMeshBuilder::SavePlane(HalfSize, UVRepeat, Folder, Name, bCreated);
	Planes.Add(Key, Plane);
	if (Plane)
	{
		Assets.Add(Plane);
	}
	return Plane;
}
