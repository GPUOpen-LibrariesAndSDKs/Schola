// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "AssetImportTask.h"
#include "AssetRegistry/IAssetRegistry.h"
#include "AssetToolsModule.h"
#include "Components/StaticMeshComponent.h"
#include "Editor.h"
#include "Engine/Blueprint.h"
#include "Engine/SCS_Node.h"
#include "Engine/SimpleConstructionScript.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInstanceConstant.h"
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "MjcfBodyComponent.h"
#include "MjcfConversion.h"
#include "ObjectTools.h"
#include "PhysicsEngine/BodySetup.h"
#include "Serialization/MemoryWriter.h"
#include "StaticMeshCompiler.h"
#include "StaticMeshResources.h"
#include "UObject/StrongObjectPtr.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace MjcfFactoryTest
{
	const TCHAR* const DestinationPath = TEXT("/Game/ScholaMjcfFactoryTest");

	/** A unit cube with one quad per face. */
	const TCHAR* const CubeObj = TEXT(R"(v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
)");

	/** A triangular prism. */
	const TCHAR* const WedgeObj = TEXT(R"(v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
v 1 0 1
v 0 1 1
f 1 3 2
f 4 5 6
f 1 2 5 4
f 2 3 6 5
f 3 1 4 6
)");

	/** Two groups in one file, which the OBJ importer reads as separate meshes but must still become a single static mesh. */
	const TCHAR* const TwoPartsObj = TEXT(R"(g PartA
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
g PartB
v 2 0 0
v 3 0 0
v 2 1 0
f 4 5 6
)");

	/** A 1 x 2 x 3 box with a corner at the origin and faces wound counterclockwise seen from outside, as OBJ expects. */
	const TCHAR* const BoxObj = TEXT(R"(v 0 0 0
v 1 0 0
v 1 2 0
v 0 2 0
v 0 0 3
v 1 0 3
v 1 2 3
v 0 2 3
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
)");

	/** Returns how many of Mesh's rendered vertex normals point into its bounding box instead of away from it. */
	int32 CountInwardNormals(const UStaticMesh& Mesh)
	{
		const FVector Center = Mesh.GetBoundingBox().GetCenter();
		const FStaticMeshVertexBuffers& Buffers = Mesh.GetRenderData()->LODResources[0].VertexBuffers;
		int32 Inward = 0;
		for (uint32 Vertex = 0; Vertex < Buffers.PositionVertexBuffer.GetNumVertices(); ++Vertex)
		{
			const FVector Position(Buffers.PositionVertexBuffer.VertexPosition(Vertex));
			const FVector Normal(Buffers.StaticMeshVertexBuffer.VertexTangentZ(Vertex));
			if (FVector::DotProduct(Normal, Position - Center) <= 0)
			{
				++Inward;
			}
		}
		return Inward;
	}

	FString WriteFixture(const TCHAR* RelativePath, const TCHAR* Contents)
	{
		const FString Path = FPaths::ConvertRelativePathToFull(FPaths::ProjectIntermediateDir() / TEXT("MjcfFactoryTest") / RelativePath);
		FFileHelper::SaveStringToFile(Contents, *Path);
		return Path;
	}

	FString WriteFixture(const TCHAR* RelativePath, const TArray<uint8>& Contents)
	{
		const FString Path = FPaths::ConvertRelativePathToFull(FPaths::ProjectIntermediateDir() / TEXT("MjcfFactoryTest") / RelativePath);
		FFileHelper::SaveArrayToFile(Contents, *Path);
		return Path;
	}

	/** Imports Filename the way a Content Browser drop does, so the factory is chosen by extension and FactoryCanImport. */
	TArray<UObject*> Import(const FString& Filename)
	{
		TStrongObjectPtr<UAssetImportTask> Task(NewObject<UAssetImportTask>());
		Task->Filename = Filename;
		Task->DestinationPath = DestinationPath;
		Task->bAutomated = true;
		Task->bReplaceExisting = true;
		Task->bSave = false;
		FAssetToolsModule::GetModule().Get().ImportAssetTasks({ Task.Get() });
		FStaticMeshCompilingManager::Get().FinishAllCompilation();
		return Task->GetObjects();
	}

	/** Returns Obj's faces as a binary STL file, splitting each face into a fan of triangles that keeps its winding. */
	TArray<uint8> ObjToStl(const TCHAR* Obj)
	{
		TArray<FString> Lines;
		FString(Obj).ParseIntoArrayLines(Lines);
		TArray<FVector3f> Vertices;
		TArray<FVector3f> Corners;
		for (const FString& Line : Lines)
		{
			TArray<FString> Words;
			Line.ParseIntoArrayWS(Words);
			if (Words.Num() == 4 && Words[0] == TEXT("v"))
			{
				Vertices.Emplace(FCString::Atof(*Words[1]), FCString::Atof(*Words[2]), FCString::Atof(*Words[3]));
			}
			else if (Words.Num() >= 4 && Words[0] == TEXT("f"))
			{
				for (int32 Corner = 2; Corner + 1 < Words.Num(); ++Corner)
				{
					Corners.Add(Vertices[FCString::Atoi(*Words[1]) - 1]);
					Corners.Add(Vertices[FCString::Atoi(*Words[Corner]) - 1]);
					Corners.Add(Vertices[FCString::Atoi(*Words[Corner + 1]) - 1]);
				}
			}
		}

		TArray<uint8> Stl;
		Stl.SetNumZeroed(80);
		FMemoryWriter Ar(Stl);
		Ar.Seek(80);
		uint32 NumTriangles = Corners.Num() / 3;
		Ar << NumTriangles;
		for (int32 Triangle = 0; Triangle < Corners.Num() / 3; ++Triangle)
		{
			FVector3f Normal = FVector3f::ZeroVector;
			uint16 Attributes = 0;
			Ar << Normal << Corners[3 * Triangle] << Corners[3 * Triangle + 1] << Corners[3 * Triangle + 2] << Attributes;
		}
		return Stl;
	}

	USCS_Node* FindNode(const UBlueprint& Blueprint, const TCHAR* Name)
	{
		for (USCS_Node* Node : Blueprint.SimpleConstructionScript->GetAllNodes())
		{
			if (Node->GetVariableName() == FName(Name))
			{
				return Node;
			}
		}
		return nullptr;
	}

	TArray<FString> ChildNames(const USCS_Node& Node)
	{
		TArray<FString> Names;
		for (const USCS_Node* Child : Node.GetChildNodes())
		{
			Names.Add(Child->GetVariableName().ToString());
		}
		return Names;
	}

	template <typename ComponentType>
	ComponentType* Template(const UBlueprint& Blueprint, const TCHAR* Name)
	{
		const USCS_Node* Node = FindNode(Blueprint, Name);
		return Node ? Cast<ComponentType>(Node->ComponentTemplate) : nullptr;
	}

	FLinearColor VectorParameter(const UMaterialInstanceConstant& Material, const TCHAR* Name)
	{
		FLinearColor Value = FLinearColor::Transparent;
		Material.GetVectorParameterValue(FHashedMaterialParameterInfo(FName(Name)), Value);
		return Value;
	}

	float ScalarParameter(const UMaterialInstanceConstant& Material, const TCHAR* Name)
	{
		float Value = -1;
		Material.GetScalarParameterValue(FHashedMaterialParameterInfo(FName(Name)), Value);
		return Value;
	}
} // namespace MjcfFactoryTest

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfFactoryImportsObjMeshesTest, "Schola.Mjcf.Factory.ImportsObjMeshes", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FMjcfFactoryImportsObjMeshesTest::RunTest(const FString& Parameters)
{
	using namespace MjcfFactoryTest;

	WriteFixture(TEXT("meshes/cube.obj"), CubeObj);
	WriteFixture(TEXT("meshes/wedge.obj"), WedgeObj);
	WriteFixture(TEXT("meshes/two_parts.obj"), TwoPartsObj);
	const FString ModelFile = WriteFixture(TEXT("robot.xml"), TEXT(R"(<mujoco model="robot">
  <compiler meshdir="meshes"/>
  <asset>
    <mesh name="cube" file="cube.obj"/>
    <mesh name="wedge" file="wedge.obj"/>
    <mesh name="cube_again" file="cube.obj"/>
    <mesh name="two_parts" file="two_parts.obj"/>
    <mesh name="part.1" file="wedge.obj"/>
    <mesh name="part_1" file="cube.obj"/>
    <mesh name="missing" file="missing.obj"/>
    <mesh name="arm" file="arm.stl"/>
  </asset>
</mujoco>)"));

	AddExpectedMessage(TEXT("Imported mesh \"part_1\" as \"part_1_2\""), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
	AddExpectedMessage(TEXT("Could not import mesh \"missing\""), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 1, false);
	AddExpectedMessage(TEXT("Could not import mesh \"arm\""), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 1, false);

	const TArray<UObject*> Imported = Import(ModelFile);
	if (!TestTrue(TEXT("Something is imported"), !Imported.IsEmpty()))
	{
		return false;
	}
	TestEqual(TEXT("The Blueprint comes first, named after the file, where it was dropped"), Imported[0]->GetPathName(), FString(TEXT("/Game/ScholaMjcfFactoryTest/robot.robot")));
	TestTrue(TEXT("It is a Blueprint"), Imported[0]->IsA<UBlueprint>());
	TArray<FString> ImportedPaths;
	for (const UObject* Object : TArrayView<UObject* const>(Imported).RightChop(1))
	{
		const UStaticMesh* Mesh = Cast<UStaticMesh>(Object);
		if (TestNotNull(TEXT("Every other imported object is a static mesh"), Mesh))
		{
			TestTrue(FString::Printf(TEXT("%s has triangles"), *Mesh->GetName()), Mesh->GetNumTriangles(0) > 0);
			ImportedPaths.Add(Mesh->GetPathName());
		}
	}
	const TArray<FString> ExpectedPaths = {
		TEXT("/Game/ScholaMjcfFactoryTest/robot/cube.cube"),
		TEXT("/Game/ScholaMjcfFactoryTest/robot/wedge.wedge"),
		TEXT("/Game/ScholaMjcfFactoryTest/robot/cube_again.cube_again"),
		TEXT("/Game/ScholaMjcfFactoryTest/robot/two_parts.two_parts"),
		TEXT("/Game/ScholaMjcfFactoryTest/robot/part_1.part_1"),
		TEXT("/Game/ScholaMjcfFactoryTest/robot/part_1_2.part_1_2"),
	};
	TestEqual(TEXT("One asset per OBJ mesh, in declaration order, in a folder named after the file"),
		FString::Join(ImportedPaths, TEXT("\n")), FString::Join(ExpectedPaths, TEXT("\n")));

	TArray<FAssetData> Assets;
	IAssetRegistry::GetChecked().GetAssetsByPath(FName(DestinationPath), Assets, /*bRecursive=*/true, /*bIncludeOnlyOnDiskAssets=*/false);
	TArray<FString> AssetPaths;
	for (const FAssetData& Asset : Assets)
	{
		AssetPaths.Add(Asset.GetObjectPathString());
	}
	TestEqual(TEXT("No other assets are created: ") + FString::Join(AssetPaths, TEXT(", ")), Assets.Num(), ExpectedPaths.Num() + 1);

	ObjectTools::ForceDeleteObjects(Imported, false);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfFactoryConvertsMeshesTest, "Schola.Mjcf.Factory.ConvertsMeshes", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FMjcfFactoryConvertsMeshesTest::RunTest(const FString& Parameters)
{
	using namespace MjcfFactoryTest;

	WriteFixture(TEXT("meshes/box.obj"), BoxObj);
	WriteFixture(TEXT("meshes/box.stl"), ObjToStl(BoxObj));
	const FString ModelFile = WriteFixture(TEXT("converted.xml"), TEXT(R"(<mujoco model="converted">
  <compiler meshdir="meshes"/>
  <asset>
    <mesh name="box" file="box.obj"/>
    <mesh name="scaled" file="box.obj" scale="0.5 1 2"/>
    <mesh name="mirrored" file="box.obj" scale="-1 1 1"/>
    <mesh name="stl_box" file="box.stl"/>
    <mesh name="stl_scaled" file="box.stl" scale="0.5 1 2"/>
    <mesh name="stl_mirrored" file="box.stl" scale="-1 1 1"/>
  </asset>
</mujoco>)"));

	// MuJoCo meters become centimeters, MuJoCo +Y becomes Unreal -Y, and the mesh scale applies along MuJoCo's axes. STL and
	// OBJ files of the same shape give the same mesh.
	const TMap<FString, FBox> ExpectedBounds = {
		{ TEXT("box"), FBox(FVector(0, -200, 0), FVector(100, 0, 300)) },
		{ TEXT("scaled"), FBox(FVector(0, -200, 0), FVector(50, 0, 600)) },
		{ TEXT("mirrored"), FBox(FVector(-100, -200, 0), FVector(0, 0, 300)) },
		{ TEXT("stl_box"), FBox(FVector(0, -200, 0), FVector(100, 0, 300)) },
		{ TEXT("stl_scaled"), FBox(FVector(0, -200, 0), FVector(50, 0, 600)) },
		{ TEXT("stl_mirrored"), FBox(FVector(-100, -200, 0), FVector(0, 0, 300)) },
	};

	const TArray<UObject*> Imported = Import(ModelFile);
	TestEqual(TEXT("Every mesh is imported, after the Blueprint"), Imported.Num(), ExpectedBounds.Num() + 1);
	for (const UObject* Object : Imported)
	{
		const UStaticMesh* Mesh = Cast<UStaticMesh>(Object);
		if (!Mesh)
		{
			continue;
		}
		const FBox* Expected = ExpectedBounds.Find(Mesh->GetName());
		if (!TestNotNull(TEXT("Imported mesh is one of the expected meshes"), Expected))
		{
			continue;
		}
		TestEqual(FString::Printf(TEXT("%s has 12 triangles"), *Mesh->GetName()), Mesh->GetNumTriangles(0), 12);
		const FBox Bounds = Mesh->GetBoundingBox();
		TestEqual(FString::Printf(TEXT("%s min corner"), *Mesh->GetName()), Bounds.Min, Expected->Min, 0.01f);
		TestEqual(FString::Printf(TEXT("%s max corner"), *Mesh->GetName()), Bounds.Max, Expected->Max, 0.01f);
		TestEqual(FString::Printf(TEXT("%s normals all point outward, so faces aren't inside out"), *Mesh->GetName()), CountInwardNormals(*Mesh), 0);
		TestEqual(FString::Printf(TEXT("%s has no generated collision"), *Mesh->GetName()), Mesh->GetBodySetup() ? Mesh->GetBodySetup()->AggGeom.GetElementCount() : 0, 0);
	}

	ObjectTools::ForceDeleteObjects(Imported, false);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfFactoryBuildsBlueprintTest, "Schola.Mjcf.Factory.BuildsBlueprint", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FMjcfFactoryBuildsBlueprintTest::RunTest(const FString& Parameters)
{
	using namespace MjcfFactoryTest;

	WriteFixture(TEXT("meshes/box.obj"), BoxObj);
	const FString ModelFile = WriteFixture(TEXT("assembled.xml"), TEXT(R"(<mujoco model="assembled">
  <compiler meshdir="meshes"/>
  <asset>
    <mesh name="box" file="box.obj"/>
    <texture name="grid" type="2d" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="1 1 1" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="5 5" texuniform="true"/>
    <material name="red" rgba="1 0 0 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    <body name="base" pos="1 2 3" quat="0.70710678 0 0 0.70710678">
      <geom name="base_box" type="mesh" mesh="box" material="red"/>
      <geom name="collision" type="box" size="0.1 0.1 0.1" group="3"/>
      <geom name="hidden" type="sphere" size="0.1" rgba="1 1 1 0"/>
      <body name="arm" pos="0 0 1">
        <geom name="ball" type="sphere" size="0.25" pos="0 1 0" rgba="0 1 0 1"/>
        <geom name="rod" type="capsule" fromto="0 0 0 0 0 1" size="0.05" material="red"/>
        <geom type="box" size="0.1 0.2 0.3" material="red" rgba="0 0 1 0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>)"));

	const TArray<UObject*> Imported = Import(ModelFile);
	UBlueprint* Blueprint = Imported.IsEmpty() ? nullptr : Cast<UBlueprint>(Imported[0]);
	if (!TestNotNull(TEXT("The first imported object is the Blueprint"), Blueprint))
	{
		ObjectTools::ForceDeleteObjects(Imported, false);
		return false;
	}

	const TArray<USCS_Node*>& Roots = Blueprint->SimpleConstructionScript->GetRootNodes();
	if (TestEqual(TEXT("One root"), Roots.Num(), 1))
	{
		TestEqual(TEXT("The root is the world body"), Roots[0]->GetVariableName(), FName(TEXT("WorldBody")));
		TestTrue(TEXT("Bodies are body components"), Roots[0]->ComponentTemplate->IsA<UMjcfBodyComponent>());
		TestEqual(TEXT("World children"), FString::Join(ChildNames(*Roots[0]), TEXT(",")), FString(TEXT("floor,base")));
	}
	const USCS_Node* Base = FindNode(*Blueprint, TEXT("base"));
	const USCS_Node* Arm = FindNode(*Blueprint, TEXT("arm"));
	const USCS_Node* Rod = FindNode(*Blueprint, TEXT("rod"));
	if (!TestNotNull(TEXT("base"), Base) || !TestNotNull(TEXT("arm"), Arm) || !TestNotNull(TEXT("rod"), Rod))
	{
		ObjectTools::ForceDeleteObjects(Imported, false);
		return false;
	}
	TestEqual(TEXT("Hidden and collision-only geoms are left out"), FString::Join(ChildNames(*Base), TEXT(",")), FString(TEXT("base_box,arm")));
	TestEqual(TEXT("Unnamed geoms are named after their body and type"), FString::Join(ChildNames(*Arm), TEXT(",")), FString(TEXT("ball,rod,arm_box")));
	TestEqual(TEXT("Capsules are a cylinder and two spheres"), FString::Join(ChildNames(*Rod), TEXT(",")), FString(TEXT("rod_Cylinder,rod_Top,rod_Bottom")));
	TestEqual(TEXT("Body name"), CastChecked<UMjcfBodyComponent>(Base->ComponentTemplate)->BodyName, FString(TEXT("base")));

	const USceneComponent* BaseTemplate = CastChecked<USceneComponent>(Base->ComponentTemplate);
	TestEqual(TEXT("Body position is converted"), BaseTemplate->GetRelativeLocation(), FVector(100, -200, 300), 1e-3f);
	TestTrue(TEXT("Body rotation is converted"), BaseTemplate->GetRelativeRotation().Quaternion().Equals(FMjcfConversion::ConvertRotation(FQuat(0, 0, 0.70710678, 0.70710678)), 1e-5));

	const UStaticMeshComponent* Ball = Template<UStaticMeshComponent>(*Blueprint, TEXT("ball"));
	if (TestNotNull(TEXT("ball"), Ball) && TestNotNull(TEXT("ball mesh"), Ball->GetStaticMesh().Get()))
	{
		TestEqual(TEXT("Spheres are 2 * radius across"), Ball->GetStaticMesh()->GetBoundingBox().GetSize() * Ball->GetRelativeScale3D(), FVector(50), 0.01f);
		TestTrue(TEXT("Visual geoms have no collision"), Ball->GetCollisionEnabled() == ECollisionEnabled::NoCollision);
		const UMaterialInstanceConstant* Green = Cast<UMaterialInstanceConstant>(Ball->GetMaterial(0));
		if (TestNotNull(TEXT("Ball material"), Green))
		{
			TestEqual(TEXT("Colors without a material get their own"), Green->GetName(), FString(TEXT("M_Color_00FF00FF")));
			TestEqual(TEXT("Color"), VectorParameter(*Green, TEXT("BaseColor")), FLinearColor(0, 1, 0, 1));
		}
	}
	const UStaticMeshComponent* Cylinder = Template<UStaticMeshComponent>(*Blueprint, TEXT("rod_Cylinder"));
	if (TestNotNull(TEXT("rod_Cylinder"), Cylinder) && TestNotNull(TEXT("rod_Cylinder mesh"), Cylinder->GetStaticMesh().Get()))
	{
		TestEqual(TEXT("Capsule cylinders are 2 * radius across and 2 * half-length long"),
			Cylinder->GetStaticMesh()->GetBoundingBox().GetSize() * Cylinder->GetRelativeScale3D(), FVector(10, 10, 100), 0.01f);
	}
	const USceneComponent* Top = Template<USceneComponent>(*Blueprint, TEXT("rod_Top"));
	if (TestNotNull(TEXT("rod_Top"), Top))
	{
		TestTrue(TEXT("Capsule caps sit at the cylinder's ends"), FMath::IsNearlyEqual(Top->GetRelativeLocation().Z, 50.0, 0.5));
	}

	const UMaterialInstanceConstant* Red = Cast<UMaterialInstanceConstant>(Template<UStaticMeshComponent>(*Blueprint, TEXT("base_box"))->GetMaterial(0));
	if (TestNotNull(TEXT("base_box material"), Red))
	{
		TestEqual(TEXT("MJCF materials are named after themselves"), Red->GetName(), FString(TEXT("M_red")));
		TestEqual(TEXT("Material color"), VectorParameter(*Red, TEXT("BaseColor")), FLinearColor(1, 0, 0, 1));
	}
	const UMaterialInstanceConstant* Blue = Cast<UMaterialInstanceConstant>(Template<UStaticMeshComponent>(*Blueprint, TEXT("arm_box"))->GetMaterial(0));
	if (TestNotNull(TEXT("arm_box material"), Blue))
	{
		TestEqual(TEXT("A geom color overrides its material's"), Blue->GetName(), FString(TEXT("M_red_0000FF80")));
		TestEqual(TEXT("Transparent colors use the translucent parent"), Blue->Parent ? Blue->Parent->GetName() : FString(), FString(TEXT("ThinTranslucentMaterial_MR")));
		TestEqual(TEXT("Opacity"), ScalarParameter(*Blue, TEXT("Opacity")), 0.5f, 0.01f);
	}

	const UStaticMeshComponent* Floor = Template<UStaticMeshComponent>(*Blueprint, TEXT("floor"));
	if (TestNotNull(TEXT("floor"), Floor) && TestNotNull(TEXT("floor mesh"), Floor->GetStaticMesh().Get()))
	{
		TestEqual(TEXT("Infinite planes become large plane meshes"), Floor->GetStaticMesh()->GetPathName(), FString(TEXT("/Game/ScholaMjcfFactoryTest/assembled/Plane_floor.Plane_floor")));
		const FBox Bounds = Floor->GetStaticMesh()->GetBoundingBox();
		TestTrue(TEXT("Plane extent"), FMath::IsNearlyEqual(Bounds.Max.X, 5000.0, 0.01));
		TestTrue(TEXT("Plane is flat"), FMath::IsNearlyZero(Bounds.GetSize().Z, 0.01));
		TestEqual(TEXT("Plane normals face up"), FVector(Floor->GetStaticMesh()->GetRenderData()->LODResources[0].VertexBuffers.StaticMeshVertexBuffer.VertexTangentZ(0)), FVector::UpVector, 1e-3f);
		const UMaterialInstanceConstant* FloorMaterial = Cast<UMaterialInstanceConstant>(Floor->GetMaterial(0));
		UTexture* Grid = nullptr;
		if (TestNotNull(TEXT("Floor material"), FloorMaterial) && FloorMaterial->GetTextureParameterValue(FHashedMaterialParameterInfo(TEXT("BaseColorMap")), Grid))
		{
			const UTexture2D* GridTexture = Cast<UTexture2D>(Grid);
			TestTrue(TEXT("Builtin textures are generated with a power-of-two size"), GridTexture && GridTexture->Source.GetSizeX() == 512 && GridTexture->Source.GetSizeY() == 512);
			TestEqual(TEXT("The texture is used"), ScalarParameter(*FloorMaterial, TEXT("BaseColorMapWeight")), 1.0f);
		}
		else
		{
			AddError(TEXT("The floor material has no texture"));
		}
	}

	// Spawning the Blueprint checks the whole chain of transforms: the ball is at (0, 1, 0) in arm, which is at (0, 0, 1) in
	// base, which is at (1, 2, 3) turned 90 degrees about Z, so MuJoCo puts it at (0, 2, 4).
	UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
	if (TestNotNull(TEXT("Editor world"), World))
	{
		AActor* Actor = World->SpawnActor<AActor>(Blueprint->GeneratedClass, FTransform::Identity);
		if (TestNotNull(TEXT("Spawned actor"), Actor))
		{
			TInlineComponentArray<UStaticMeshComponent*> Components(Actor);
			const UStaticMeshComponent* const* SpawnedBall = Components.FindByPredicate([](const UStaticMeshComponent* Component) { return Component->GetFName() == TEXT("ball"); });
			if (TestNotNull(TEXT("Spawned ball"), SpawnedBall))
			{
				TestEqual(TEXT("Ball world position"), (*SpawnedBall)->Bounds.Origin, FMjcfConversion::ConvertPosition(FVector(0, 2, 4)), 0.01f);
			}
			World->DestroyActor(Actor);
		}
	}

	// Importing again updates the same Blueprint and assets instead of adding new ones.
	TArray<FAssetData> AssetsBefore;
	IAssetRegistry::GetChecked().GetAssetsByPath(FName(DestinationPath), AssetsBefore, /*bRecursive=*/true, /*bIncludeOnlyOnDiskAssets=*/false);
	const TArray<UObject*> Reimported = Import(ModelFile);
	TestTrue(TEXT("The same Blueprint is updated"), !Reimported.IsEmpty() && Reimported[0] == Blueprint);
	TestEqual(TEXT("Component names stay the same"), FString::Join(ChildNames(*FindNode(*Blueprint, TEXT("arm"))), TEXT(",")), FString(TEXT("ball,rod,arm_box")));
	TestEqual(TEXT("Same number of components"), Blueprint->SimpleConstructionScript->GetAllNodes().Num(), 11);
	TArray<FAssetData> AssetsAfter;
	IAssetRegistry::GetChecked().GetAssetsByPath(FName(DestinationPath), AssetsAfter, /*bRecursive=*/true, /*bIncludeOnlyOnDiskAssets=*/false);
	TestEqual(TEXT("No assets are added"), AssetsAfter.Num(), AssetsBefore.Num());

	TArray<UObject*> ToDelete = Imported;
	for (UObject* Object : Reimported)
	{
		ToDelete.AddUnique(Object);
	}
	ObjectTools::ForceDeleteObjects(ToDelete, false);
	return true;
}

#endif
