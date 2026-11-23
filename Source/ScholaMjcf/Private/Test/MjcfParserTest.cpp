// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "HAL/FileManager.h"
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "MjcfParser.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace MjcfParserTest
{
	const FString& ModelDir()
	{
		static const FString Dir = FPaths::ConvertRelativePathToFull(FPaths::Combine(FPaths::ProjectIntermediateDir(), TEXT("MjcfParserTest")));
		return Dir;
	}

	FString ModelFile()
	{
		return ModelDir() / TEXT("robot.xml");
	}

	/** Parses Xml as ModelFile(), serving included files from Includes (keyed by absolute path) instead of disk. */
	TOptional<FMjcfModel> Parse(const FString& Xml, TArray<FMjcfDiagnostic>& OutDiagnostics, TMap<FString, FString> Includes = TMap<FString, FString>())
	{
		FMjcfFileReader Reader = [Includes = MoveTemp(Includes)](const FString& Path, FString& OutContents) {
			if (const FString* Contents = Includes.Find(Path))
			{
				OutContents = *Contents;
				return true;
			}
			return false;
		};
		return FMjcfParser::ParseString(Xml, ModelFile(), OutDiagnostics, MoveTemp(Reader));
	}

	FString Describe(const TArray<FMjcfDiagnostic>& Diagnostics)
	{
		TArray<FString> Lines;
		for (const FMjcfDiagnostic& Diagnostic : Diagnostics)
		{
			Lines.Add(Diagnostic.ToString());
		}
		return FString::Join(Lines, TEXT("\n"));
	}

	bool HasDiagnostic(const TArray<FMjcfDiagnostic>& Diagnostics, EMjcfSeverity Severity, const TCHAR* Fragment)
	{
		return Diagnostics.ContainsByPredicate([Severity, Fragment](const FMjcfDiagnostic& Diagnostic) {
			return Diagnostic.Severity == Severity && Diagnostic.Message.Contains(Fragment);
		});
	}

	const FMjcfMesh* FindMesh(const FMjcfModel& Model, const TCHAR* Name)
	{
		return Model.Meshes.FindByPredicate([Name](const FMjcfMesh& Mesh) { return Mesh.Name.Equals(Name, ESearchCase::CaseSensitive); });
	}

	/** Parses Xml and returns the file of its only mesh, or an empty string after recording a test error. */
	FString OnlyMeshFile(FAutomationTestBase& Test, const FString& Xml)
	{
		TArray<FMjcfDiagnostic> Diagnostics;
		const TOptional<FMjcfModel> Model = Parse(Xml, Diagnostics);
		if (!Model || Model->Meshes.Num() != 1)
		{
			Test.AddError(FString::Printf(TEXT("Expected one mesh from:\n%s\n%s"), *Xml, *Describe(Diagnostics)));
			return FString();
		}
		return Model->Meshes[0].File;
	}

	void ExpectError(FAutomationTestBase& Test, const TCHAR* What, const FString& Xml, const TCHAR* Fragment, TMap<FString, FString> Includes = TMap<FString, FString>())
	{
		TArray<FMjcfDiagnostic> Diagnostics;
		const TOptional<FMjcfModel> Model = Parse(Xml, Diagnostics, MoveTemp(Includes));
		Test.TestFalse(FString::Printf(TEXT("%s: parse fails"), What), Model.IsSet());
		Test.TestTrue(FString::Printf(TEXT("%s: reports an error containing '%s', got:\n%s"), What, Fragment, *Describe(Diagnostics)),
			HasDiagnostic(Diagnostics, EMjcfSeverity::Error, Fragment));
	}
} // namespace MjcfParserTest

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserReadsMeshesTest, "Schola.Mjcf.Parser.ReadsMeshes", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserReadsMeshesTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	TArray<FMjcfDiagnostic> Diagnostics;
	const TOptional<FMjcfModel> Model = Parse(TEXT(R"(<mujoco model="robot">
  <asset>
    <mesh name="base" file="base.obj"/>
    <mesh file="parts/link.stl" scale="1e-3  0.002 3E-3"/>
  </asset>
</mujoco>)"), Diagnostics);
	if (!TestTrue(TEXT("Parses: ") + Describe(Diagnostics), Model.IsSet()))
	{
		return false;
	}

	TestEqual(TEXT("Mesh count"), Model->Meshes.Num(), 2);
	TestEqual(TEXT("Source file"), Model->SourceFile, ModelFile());

	const FMjcfMesh* Base = FindMesh(*Model, TEXT("base"));
	const FMjcfMesh* Link = FindMesh(*Model, TEXT("link"));
	if (!TestNotNull(TEXT("base"), Base) || !TestNotNull(TEXT("link, named after its file"), Link))
	{
		return false;
	}
	TestEqual(TEXT("base file"), Base->File, ModelDir() / TEXT("base.obj"));
	TestEqual(TEXT("base scale"), Base->Scale, FVector::OneVector);
	TestEqual(TEXT("base element path"), Base->ElementPath, FString(TEXT("mujoco/asset[#0]/mesh[base]")));
	TestEqual(TEXT("link file"), Link->File, ModelDir() / TEXT("parts/link.stl"));
	TestEqual(TEXT("link scale"), Link->Scale, FVector(0.001, 0.002, 0.003));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserResolvesPathsTest, "Schola.Mjcf.Parser.ResolvesPaths", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserResolvesPathsTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	TestEqual(TEXT("meshdir is relative to the model folder"),
		OnlyMeshFile(*this, TEXT(R"(<mujoco><compiler meshdir="assets"/><asset><mesh file="a.obj"/></asset></mujoco>)")),
		ModelDir() / TEXT("assets/a.obj"));

	TestEqual(TEXT("meshdir can leave the model folder"),
		OnlyMeshFile(*this, TEXT(R"(<mujoco><compiler meshdir="../shared"/><asset><mesh file="a.obj"/></asset></mujoco>)")),
		FPaths::GetPath(ModelDir()) / TEXT("shared/a.obj"));

	const FString AbsoluteFile = ModelDir() / TEXT("elsewhere/absolute.obj");
	TestEqual(TEXT("Absolute mesh paths ignore meshdir"),
		OnlyMeshFile(*this, FString::Printf(TEXT(R"(<mujoco><compiler meshdir="assets"/><asset><mesh file="%s"/></asset></mujoco>)"), *AbsoluteFile)),
		AbsoluteFile);

	TestEqual(TEXT("assetdir applies when meshdir is unset"),
		OnlyMeshFile(*this, TEXT(R"(<mujoco><compiler assetdir="all"/><asset><mesh file="a.obj"/></asset></mujoco>)")),
		ModelDir() / TEXT("all/a.obj"));

	TestEqual(TEXT("meshdir wins over assetdir, whatever their order"),
		OnlyMeshFile(*this, TEXT(R"(<mujoco><compiler meshdir="meshes"/><compiler assetdir="all"/><asset><mesh file="a.obj"/></asset></mujoco>)")),
		ModelDir() / TEXT("meshes/a.obj"));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserDefaultScaleTest, "Schola.Mjcf.Parser.DefaultScale", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserDefaultScaleTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	// The main class's <mesh> comes after its nested classes on purpose: they must still inherit it.
	TArray<FMjcfDiagnostic> Diagnostics;
	const TOptional<FMjcfModel> Model = Parse(TEXT(R"(<mujoco>
  <default>
    <default class="big">
      <mesh scale="2 2 2"/>
      <default class="bigger"/>
    </default>
    <default class="inherits"/>
    <mesh scale="0.001 0.001 0.001"/>
  </default>
  <asset>
    <mesh name="plain" file="a.obj"/>
    <mesh name="classed" class="big" file="b.obj"/>
    <mesh name="nested" class="bigger" file="c.obj"/>
    <mesh name="inherited" class="inherits" file="d.obj"/>
    <mesh name="own" class="big" file="e.obj" scale="3 3 3"/>
  </asset>
</mujoco>)"), Diagnostics);
	if (!TestTrue(TEXT("Parses: ") + Describe(Diagnostics), Model.IsSet()))
	{
		return false;
	}

	auto ScaleOf = [this, &Model](const TCHAR* Name) {
		const FMjcfMesh* Mesh = FindMesh(*Model, Name);
		TestNotNull(Name, Mesh);
		return Mesh ? Mesh->Scale : FVector::ZeroVector;
	};
	TestEqual(TEXT("Main class scale"), ScaleOf(TEXT("plain")), FVector(0.001));
	TestEqual(TEXT("Named class scale"), ScaleOf(TEXT("classed")), FVector(2.0));
	TestEqual(TEXT("Nested class inherits its parent"), ScaleOf(TEXT("nested")), FVector(2.0));
	TestEqual(TEXT("Class without a mesh setting inherits main"), ScaleOf(TEXT("inherited")), FVector(0.001));
	TestEqual(TEXT("The mesh's own scale wins"), ScaleOf(TEXT("own")), FVector(3.0));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserIncludesTest, "Schola.Mjcf.Parser.Includes", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserIncludesTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	TMap<FString, FString> Includes;
	Includes.Add(ModelDir() / TEXT("sub/robot.xml"), TEXT(R"(<mujoco>
  <asset><mesh name="arm" file="arm.stl"/></asset>
  <include file="sub/parts.xml"/>
</mujoco>)"));
	Includes.Add(ModelDir() / TEXT("sub/parts.xml"), TEXT(R"(<mujoco><asset><mesh name="gripper" file="gripper.obj"/></asset></mujoco>)"));

	TArray<FMjcfDiagnostic> Diagnostics;
	const TOptional<FMjcfModel> Model = Parse(TEXT(R"(<mujoco>
  <compiler meshdir="assets"/>
  <asset><mesh name="floor" file="floor.obj"/></asset>
  <include file="sub/robot.xml"/>
</mujoco>)"), Diagnostics, MoveTemp(Includes));
	if (!TestTrue(TEXT("Parses: ") + Describe(Diagnostics), Model.IsSet()) || !TestEqual(TEXT("Mesh count"), Model->Meshes.Num(), 3))
	{
		return false;
	}

	TestEqual(TEXT("Main file mesh first"), Model->Meshes[0].Name, FString(TEXT("floor")));
	TestEqual(TEXT("Included mesh next"), Model->Meshes[1].Name, FString(TEXT("arm")));
	TestEqual(TEXT("Nested include last"), Model->Meshes[2].Name, FString(TEXT("gripper")));
	TestEqual(TEXT("meshdir from the main file applies to included meshes"), Model->Meshes[1].File, ModelDir() / TEXT("assets/arm.stl"));
	TestEqual(TEXT("Nested include paths are relative to the main file"), Model->Meshes[2].File, ModelDir() / TEXT("assets/gripper.obj"));

	// MuJoCo 3.13 prefers the main file's folder and falls back to the including file's folder.
	TMap<FString, FString> Fallback;
	Fallback.Add(ModelDir() / TEXT("sub/a.xml"), TEXT(R"(<mujoco><include file="b.xml"/><include file="c.xml"/></mujoco>)"));
	Fallback.Add(ModelDir() / TEXT("b.xml"), TEXT(R"(<mujoco><body name="b_next_to_main"/></mujoco>)"));
	Fallback.Add(ModelDir() / TEXT("sub/b.xml"), TEXT(R"(<mujoco><body name="b_next_to_includer"/></mujoco>)"));
	Fallback.Add(ModelDir() / TEXT("sub/c.xml"), TEXT(R"(<mujoco><body name="c_next_to_includer"/></mujoco>)"));
	TArray<FMjcfDiagnostic> FallbackDiagnostics;
	const TOptional<FMjcfModel> FallbackModel = Parse(TEXT(R"(<mujoco><worldbody><include file="sub/a.xml"/></worldbody></mujoco>)"), FallbackDiagnostics, MoveTemp(Fallback));
	if (TestTrue(TEXT("Fallback parses: ") + Describe(FallbackDiagnostics), FallbackModel.IsSet())
		&& TestEqual(TEXT("Fallback bodies"), FallbackModel->WorldBody.Bodies.Num(), 2))
	{
		TestEqual(TEXT("The main file's folder wins"), FallbackModel->WorldBody.Bodies[0].Name, FString(TEXT("b_next_to_main")));
		TestEqual(TEXT("The including file's folder is the fallback"), FallbackModel->WorldBody.Bodies[1].Name, FString(TEXT("c_next_to_includer")));
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserErrorsTest, "Schola.Mjcf.Parser.Errors", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserErrorsTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	ExpectError(*this, TEXT("Invalid XML"), TEXT("<mujoco><asset>"), TEXT("Not valid XML"));
	ExpectError(*this, TEXT("Wrong root"), TEXT("<robot></robot>"), TEXT("root element must be <mujoco>"));
	ExpectError(*this, TEXT("Duplicate names"),
		TEXT(R"(<mujoco><asset><mesh file="a/arm.obj"/><mesh file="b/arm.stl"/></asset></mujoco>)"), TEXT("already defined"));
	ExpectError(*this, TEXT("Malformed scale"),
		TEXT(R"(<mujoco><asset><mesh file="a.obj" scale="1 2"/></asset></mujoco>)"), TEXT("scale must be three numbers"));
	ExpectError(*this, TEXT("Scale with a unit suffix"),
		TEXT(R"(<mujoco><asset><mesh file="a.obj" scale="1.5cm 1 1"/></asset></mujoco>)"), TEXT("scale must be three numbers"));
	ExpectError(*this, TEXT("Comma-separated scale"),
		TEXT(R"(<mujoco><asset><mesh file="a.obj" scale="1,1,1"/></asset></mujoco>)"), TEXT("scale must be three numbers"));
	ExpectError(*this, TEXT("Non-finite scale"),
		TEXT(R"(<mujoco><asset><mesh file="a.obj" scale="nan 1 1"/></asset></mujoco>)"), TEXT("scale must be three numbers"));
	ExpectError(*this, TEXT("Unknown class"),
		TEXT(R"(<mujoco><asset><mesh file="a.obj" class="missing"/></asset></mujoco>)"), TEXT("Unknown default class"));
	ExpectError(*this, TEXT("Nested default without a class"),
		TEXT(R"(<mujoco><default><default><mesh scale="1 1 1"/></default></default></mujoco>)"), TEXT("need a class"));
	ExpectError(*this, TEXT("Duplicate class"),
		TEXT(R"(<mujoco><default><default class="a"/><default class="a"/></default></mujoco>)"), TEXT("defined more than once"));
	ExpectError(*this, TEXT("Include without a file"), TEXT(R"(<mujoco><include/></mujoco>)"), TEXT("needs a file"));
	ExpectError(*this, TEXT("Missing include"), TEXT(R"(<mujoco><include file="missing.xml"/></mujoco>)"), TEXT("Could not read"));

	TMap<FString, FString> Part;
	Part.Add(ModelDir() / TEXT("part.xml"), TEXT("<mujoco></mujoco>"));
	ExpectError(*this, TEXT("Included twice"),
		TEXT(R"(<mujoco><include file="part.xml"/><include file="part.xml"/></mujoco>)"), TEXT("more than once"), Part);

	TMap<FString, FString> WrongRoot;
	WrongRoot.Add(ModelDir() / TEXT("part.xml"), TEXT("<robot></robot>"));
	ExpectError(*this, TEXT("Included file with the wrong root"),
		TEXT(R"(<mujoco><include file="part.xml"/></mujoco>)"), TEXT("root element must be <mujoco>"), WrongRoot);

	ExpectError(*this, TEXT("Unknown mesh"), TEXT(R"(<mujoco><worldbody><geom type="mesh" mesh="nope"/></worldbody></mujoco>)"), TEXT("Unknown mesh \"nope\""));
	ExpectError(*this, TEXT("Mesh geom without a mesh"), TEXT(R"(<mujoco><worldbody><geom type="mesh"/></worldbody></mujoco>)"), TEXT("need a mesh attribute"));
	ExpectError(*this, TEXT("Unknown material"), TEXT(R"(<mujoco><worldbody><geom size="1" material="nope"/></worldbody></mujoco>)"), TEXT("Unknown material"));
	ExpectError(*this, TEXT("Unknown texture"), TEXT(R"(<mujoco><asset><material name="m" texture="nope"/></asset></mujoco>)"), TEXT("which is not defined"));
	ExpectError(*this, TEXT("Material without a name"), TEXT(R"(<mujoco><asset><material rgba="1 0 0 1"/></asset></mujoco>)"), TEXT("Materials need a name"));
	ExpectError(*this, TEXT("Unknown geom type"), TEXT(R"(<mujoco><worldbody><geom type="cube" size="1"/></worldbody></mujoco>)"), TEXT("type must be one of"));
	ExpectError(*this, TEXT("Two orientations"), TEXT(R"(<mujoco><worldbody><body quat="1 0 0 0" euler="0 0 0"/></worldbody></mujoco>)"), TEXT("Only one of quat"));
	ExpectError(*this, TEXT("Zero quaternion"), TEXT(R"(<mujoco><worldbody><body quat="0 0 0 0"/></worldbody></mujoco>)"), TEXT("must not be zero"));
	ExpectError(*this, TEXT("Parallel xyaxes"), TEXT(R"(<mujoco><worldbody><body xyaxes="1 0 0 2 0 0"/></worldbody></mujoco>)"), TEXT("non-parallel"));
	ExpectError(*this, TEXT("Unknown childclass"), TEXT(R"(<mujoco><worldbody><body childclass="nope"/></worldbody></mujoco>)"), TEXT("Unknown default class"));
	ExpectError(*this, TEXT("Bad angle unit"), TEXT(R"(<mujoco><compiler angle="grad"/></mujoco>)"), TEXT("angle must be"));
	ExpectError(*this, TEXT("Bad eulerseq"), TEXT(R"(<mujoco><compiler eulerseq="xyw"/></mujoco>)"), TEXT("eulerseq must be"));
	ExpectError(*this, TEXT("Malformed rgba"), TEXT(R"(<mujoco><worldbody><geom size="1" rgba="1 0 0"/></worldbody></mujoco>)"), TEXT("rgba must be four numbers"));
	ExpectError(*this, TEXT("Too many sizes"), TEXT(R"(<mujoco><worldbody><geom size="1 2 3 4"/></worldbody></mujoco>)"), TEXT("size must be one to three numbers"));
	ExpectError(*this, TEXT("Joint on the world"), TEXT(R"(<mujoco><worldbody><joint/></worldbody></mujoco>)"), TEXT("world body can't have joints"));
	ExpectError(*this, TEXT("Nested free joint"), TEXT(R"(<mujoco><worldbody><body><body><freejoint/></body></body></worldbody></mujoco>)"), TEXT("directly in <worldbody>"));
	ExpectError(*this, TEXT("Zero joint axis"), TEXT(R"(<mujoco><worldbody><body><joint axis="0 0 0"/></body></worldbody></mujoco>)"), TEXT("must not be zero"));
	ExpectError(*this, TEXT("Range without autolimits"),
		TEXT(R"(<mujoco><compiler autolimits="false"/><worldbody><body><joint range="0 1"/></body></worldbody></mujoco>)"), TEXT("limited is 'auto'"));
	ExpectError(*this, TEXT("Reversed range"), TEXT(R"(<mujoco><worldbody><body><joint range="1 0"/></body></worldbody></mujoco>)"), TEXT("lower limit followed"));
	ExpectError(*this, TEXT("Duplicate joint names"),
		TEXT(R"(<mujoco><worldbody><body><joint name="j"/></body><body><joint name="j"/></body></worldbody></mujoco>)"), TEXT("joint named \"j\" is already defined"));
	ExpectError(*this, TEXT("Nested mocap body"), TEXT(R"(<mujoco><worldbody><body><body mocap="true"/></body></worldbody></mujoco>)"), TEXT("Mocap bodies"));
	ExpectError(*this, TEXT("Inertial without a mass"), TEXT(R"(<mujoco><worldbody><body><inertial diaginertia="1 1 1"/></body></worldbody></mujoco>)"), TEXT("needs a mass"));
	ExpectError(*this, TEXT("Inertial with two inertias"),
		TEXT(R"(<mujoco><worldbody><body><inertial mass="1" diaginertia="1 1 1" fullinertia="1 1 1 0 0 0"/></body></worldbody></mujoco>)"), TEXT("Only one of diaginertia"));
	ExpectError(*this, TEXT("fullinertia with an orientation"),
		TEXT(R"(<mujoco><worldbody><body><inertial mass="1" fullinertia="1 1 1 0 0 0" quat="1 0 0 0"/></body></worldbody></mujoco>)"), TEXT("can't be combined"));
	ExpectError(*this, TEXT("Two inertials"),
		TEXT(R"(<mujoco><worldbody><body><inertial mass="1" diaginertia="1 1 1"/><inertial mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>)"), TEXT("only one <inertial>"));
	ExpectError(*this, TEXT("Actuator on an unknown joint"), TEXT(R"(<mujoco><actuator><motor joint="nope"/></actuator></mujoco>)"), TEXT("Unknown joint \"nope\""));
	ExpectError(*this, TEXT("Actuator without a target"), TEXT(R"(<mujoco><actuator><motor/></actuator></mujoco>)"), TEXT("need a joint"));
	ExpectError(*this, TEXT("inheritrange on a joint without a range"),
		TEXT(R"(<mujoco><worldbody><body><joint name="j"/></body></worldbody><actuator><position joint="j" inheritrange="1"/></actuator></mujoco>)"), TEXT("inheritrange needs"));
	ExpectError(*this, TEXT("Exclude with an unknown body"),
		TEXT(R"(<mujoco><worldbody><body name="a"/></worldbody><contact><exclude body1="a" body2="nope"/></contact></mujoco>)"), TEXT("Unknown body \"nope\""));
	ExpectError(*this, TEXT("Zero timestep"), TEXT(R"(<mujoco><option timestep="0"/></mujoco>)"), TEXT("timestep must be positive"));
	return true;
}

namespace MjcfParserTest
{
	/** Builds a rotation from a quaternion written in MJCF's w x y z order. */
	FQuat MjcfQuat(double W, double X, double Y, double Z)
	{
		return FQuat(X, Y, Z, W);
	}

	/** Returns whether A and B are the same rotation; a quaternion and its negation are. */
	bool SameRotation(const FQuat& A, const FQuat& B)
	{
		return FMath::Abs(A | B) > 1 - 1e-7;
	}

	/** Parses Xml, which must succeed, and returns its world body. */
	FMjcfBody ParseWorld(FAutomationTestBase& Test, const FString& Xml)
	{
		TArray<FMjcfDiagnostic> Diagnostics;
		const TOptional<FMjcfModel> Model = Parse(Xml, Diagnostics);
		Test.TestTrue(TEXT("Parses: ") + Describe(Diagnostics), Model.IsSet());
		return Model ? Model->WorldBody : FMjcfBody();
	}
} // namespace MjcfParserTest

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserBodiesTest, "Schola.Mjcf.Parser.Bodies", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserBodiesTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	TMap<FString, FString> Includes;
	Includes.Add(ModelDir() / TEXT("arm.xml"), TEXT(R"(<mujoco><worldbody><body name="arm" pos="0 0 1"/></worldbody></mujoco>)"));

	TArray<FMjcfDiagnostic> Diagnostics;
	const TOptional<FMjcfModel> Model = Parse(TEXT(R"(<mujoco model="robot">
  <default>
    <geom rgba="1 0 0 1"/>
    <default class="visual">
      <geom type="mesh" group="2"/>
    </default>
    <default class="shell">
      <geom type="box" size="0.1 0.2 0.3" quat="0 1 0 0"/>
    </default>
  </default>
  <asset>
    <mesh name="link" file="link.obj"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="base" pos="1 2 3" childclass="visual">
      <geom name="base_visual" mesh="link"/>
      <geom name="base_shell" class="shell" euler="0 0 90"/>
      <frame pos="0 1 0" euler="0 0 90">
        <body name="child" pos="1 0 0" childclass="shell">
          <geom name="child_box" pos="0 0 1"/>
        </body>
        <geom name="framed" type="sphere" size="0.5" pos="1 0 0" group="1"/>
      </frame>
    </body>
  </worldbody>
  <include file="arm.xml"/>
</mujoco>)"), Diagnostics, MoveTemp(Includes));
	if (!TestTrue(TEXT("Parses: ") + Describe(Diagnostics), Model.IsSet()))
	{
		return false;
	}
	TestEqual(TEXT("Model name"), Model->ModelName, FString(TEXT("robot")));

	const FMjcfBody& World = Model->WorldBody;
	if (!TestEqual(TEXT("World geoms"), World.Geoms.Num(), 1) || !TestEqual(TEXT("Bodies from every <worldbody>, in order"), World.Bodies.Num(), 2))
	{
		return false;
	}
	TestEqual(TEXT("Floor type"), (int32)World.Geoms[0].Type, (int32)EMjcfGeomType::Plane);
	TestEqual(TEXT("Floor size"), World.Geoms[0].Size, FVector(0, 0, 0.05));
	TestEqual(TEXT("Main class geom default"), World.Geoms[0].Rgba.X, 1.0);
	TestEqual(TEXT("Included body comes second"), World.Bodies[1].Name, FString(TEXT("arm")));

	const FMjcfBody& Base = World.Bodies[0];
	TestEqual(TEXT("Body position"), Base.Position, FVector(1, 2, 3));
	if (!TestEqual(TEXT("Base geoms, including the one inside the frame"), Base.Geoms.Num(), 3) || !TestEqual(TEXT("Base child bodies"), Base.Bodies.Num(), 1))
	{
		return false;
	}

	const FMjcfGeom& Visual = Base.Geoms[0];
	TestEqual(TEXT("childclass sets the type"), (int32)Visual.Type, (int32)EMjcfGeomType::Mesh);
	TestEqual(TEXT("childclass sets the group"), Visual.Group, 2);
	TestEqual(TEXT("Mesh name"), Visual.Mesh, FString(TEXT("link")));
	TestEqual(TEXT("Classes inherit main's rgba"), Visual.Rgba.X, 1.0);

	const FMjcfGeom& Shell = Base.Geoms[1];
	TestEqual(TEXT("class wins over childclass"), (int32)Shell.Type, (int32)EMjcfGeomType::Box);
	TestEqual(TEXT("Class size"), Shell.Size, FVector(0.1, 0.2, 0.3));
	TestTrue(TEXT("The geom's own euler replaces the class's quat"), SameRotation(Shell.Rotation, MjcfQuat(0.70710678, 0, 0, 0.70710678)));

	const FMjcfGeom& Framed = Base.Geoms[2];
	TestEqual(TEXT("Frame position applies to geoms"), Framed.Position, FVector(0, 2, 0), 1e-9f);
	TestTrue(TEXT("Frame rotation applies to geoms"), SameRotation(Framed.Rotation, MjcfQuat(0.70710678, 0, 0, 0.70710678)));
	TestEqual(TEXT("Geom group"), Framed.Group, 1);

	const FMjcfBody& Child = Base.Bodies[0];
	TestEqual(TEXT("Frame position applies to bodies, as in MuJoCo"), Child.Position, FVector(0, 2, 0), 1e-9f);
	TestTrue(TEXT("Frame rotation applies to bodies"), SameRotation(Child.Rotation, MjcfQuat(0.70710678, 0, 0, 0.70710678)));
	if (TestEqual(TEXT("Child geoms"), Child.Geoms.Num(), 1))
	{
		TestEqual(TEXT("Child's childclass replaces its parent's"), (int32)Child.Geoms[0].Type, (int32)EMjcfGeomType::Box);
		TestEqual(TEXT("Geom position is relative to its body"), Child.Geoms[0].Position, FVector(0, 0, 1));
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserOrientationsTest, "Schola.Mjcf.Parser.Orientations", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserOrientationsTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	// Expected values are body_quat and geom_* from MuJoCo 3.13 for the same XML.
	const auto BodyRotation = [this](const TCHAR* Compiler, const TCHAR* Orientation) {
		const FMjcfBody World = ParseWorld(*this, FString::Printf(TEXT("<mujoco>%s<worldbody><body %s/></worldbody></mujoco>"), Compiler, Orientation));
		return World.Bodies.IsEmpty() ? FQuat(ForceInit) : World.Bodies[0].Rotation;
	};
	const struct
	{
		const TCHAR* Compiler;
		const TCHAR* Orientation;
		FQuat Expected;
	} Cases[] = {
		{ TEXT(""), TEXT(R"(quat="2 0 0 0")"), MjcfQuat(1, 0, 0, 0) },
		{ TEXT(""), TEXT(R"(euler="90 0 0")"), MjcfQuat(0.70710678, 0.70710678, 0, 0) },
		{ TEXT(""), TEXT(R"(euler="30 40 50")"), MjcfQuat(0.78522072, 0.36004217, 0.19662823, 0.46382691) },
		{ TEXT(R"(<compiler eulerseq="XYZ"/>)"), TEXT(R"(euler="30 40 50")"), MjcfQuat(0.86004217, 0.08080469, 0.40219849, 0.30337177) },
		{ TEXT(R"(<compiler eulerseq="zyx"/>)"), TEXT(R"(euler="30 40 50")"), MjcfQuat(0.86004217, 0.30337177, 0.40219849, 0.08080469) },
		{ TEXT(R"(<compiler angle="radian"/>)"), TEXT(R"(euler="1.5707963267948966 0 0")"), MjcfQuat(0.70710678, 0.70710678, 0, 0) },
		{ TEXT(""), TEXT(R"(axisangle="0 0 2 90")"), MjcfQuat(0.70710678, 0, 0, 0.70710678) },
		{ TEXT(""), TEXT(R"(xyaxes="0 1 0 -1 0 0")"), MjcfQuat(0.70710678, 0, 0, 0.70710678) },
		{ TEXT(""), TEXT(R"(xyaxes="1 1 0 0 1 0")"), MjcfQuat(0.92387953, 0, 0, 0.38268343) },
		{ TEXT(""), TEXT(R"(zaxis="1 0 0")"), MjcfQuat(0.70710678, 0, 0.70710678, 0) },
	};
	for (const auto& Case : Cases)
	{
		const FQuat Actual = BodyRotation(Case.Compiler, Case.Orientation);
		TestTrue(FString::Printf(TEXT("%s %s gives %s, got %s"), Case.Compiler, Case.Orientation, *Case.Expected.ToString(), *Actual.ToString()), SameRotation(Actual, Case.Expected));
	}

	const FMjcfBody World = ParseWorld(*this, TEXT(R"(<mujoco><worldbody>
  <geom type="box" fromto="0 0 0 0 0 2" size="0.1 0.2"/>
  <geom type="capsule" fromto="0 0 0 2 0 0" size="0.1" pos="5 5 5"/>
</worldbody></mujoco>)"));
	if (TestEqual(TEXT("fromto geoms"), World.Geoms.Num(), 2))
	{
		TestEqual(TEXT("fromto box uses size[0] across"), World.Geoms[0].Size, FVector(0.1, 0.1, 1));
		TestEqual(TEXT("fromto box position"), World.Geoms[0].Position, FVector(0, 0, 1));
		TestTrue(TEXT("fromto box rotation"), SameRotation(World.Geoms[0].Rotation, MjcfQuat(0, 1, 0, 0)));
		TestEqual(TEXT("fromto capsule half-length"), World.Geoms[1].Size, FVector(0.1, 1, 0));
		TestEqual(TEXT("fromto wins over pos"), World.Geoms[1].Position, FVector(1, 0, 0));
		TestTrue(TEXT("fromto capsule rotation"), SameRotation(World.Geoms[1].Rotation, MjcfQuat(0.70710678, 0, -0.70710678, 0)));
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserMaterialsTest, "Schola.Mjcf.Parser.Materials", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserMaterialsTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	TArray<FMjcfDiagnostic> Diagnostics;
	const TOptional<FMjcfModel> Model = Parse(TEXT(R"(<mujoco>
  <compiler texturedir="textures" meshdir="meshes"/>
  <default>
    <default class="shiny">
      <material specular="0.9" shininess="0.8"/>
    </default>
  </default>
  <asset>
    <material name="floor" texture="grid" texrepeat="5 6" texuniform="true" reflectance="0.2"/>
    <material name="red" class="shiny" rgba="1 0 0 0.5"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="200"/>
    <texture type="2d" file="wood.png"/>
    <texture type="skybox" builtin="gradient" width="512" height="512"/>
  </asset>
</mujoco>)"), Diagnostics);
	if (!TestTrue(TEXT("Parses: ") + Describe(Diagnostics), Model.IsSet()))
	{
		return false;
	}

	const FMjcfMaterial* Floor = Model->FindMaterial(TEXT("floor"));
	const FMjcfMaterial* Red = Model->FindMaterial(TEXT("red"));
	if (TestNotNull(TEXT("floor"), Floor))
	{
		TestEqual(TEXT("A material can use a texture declared after it"), Floor->Texture, FString(TEXT("grid")));
		TestEqual(TEXT("texrepeat"), Floor->TexRepeat, FVector2D(5, 6));
		TestTrue(TEXT("texuniform"), Floor->bTexUniform);
		TestEqual(TEXT("reflectance"), Floor->Reflectance, 0.2);
		TestEqual(TEXT("Default rgba is white"), Floor->Rgba.X, 1.0);
	}
	if (TestNotNull(TEXT("red"), Red))
	{
		TestEqual(TEXT("rgba alpha"), Red->Rgba.W, 0.5);
		TestEqual(TEXT("Class specular"), Red->Specular, 0.9);
		TestEqual(TEXT("Class shininess"), Red->Shininess, 0.8);
	}

	const FMjcfTexture* Grid = Model->FindTexture(TEXT("grid"));
	const FMjcfTexture* Wood = Model->FindTexture(TEXT("wood"));
	if (TestNotNull(TEXT("grid"), Grid))
	{
		TestEqual(TEXT("Type"), (int32)Grid->Type, (int32)EMjcfTextureType::TwoD);
		TestEqual(TEXT("Builtin"), (int32)Grid->Builtin, (int32)EMjcfTextureBuiltin::Checker);
		TestEqual(TEXT("Mark"), (int32)Grid->Mark, (int32)EMjcfTextureMark::Edge);
		TestEqual(TEXT("rgb1"), Grid->Rgb1, FVector(0.2, 0.3, 0.4));
		TestEqual(TEXT("markrgb"), Grid->MarkRgb, FVector(0.8));
		TestEqual(TEXT("Width"), Grid->Width, 300);
		TestEqual(TEXT("Height"), Grid->Height, 200);
	}
	if (TestNotNull(TEXT("wood, named after its file"), Wood))
	{
		TestEqual(TEXT("Texture files use texturedir, not meshdir"), Wood->File, ModelDir() / TEXT("textures/wood.png"));
	}
	TestEqual(TEXT("Unnamed builtin textures are kept"), Model->Textures.Num(), 3);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserWarningsTest, "Schola.Mjcf.Parser.Warnings", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserWarningsTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	TArray<FMjcfDiagnostic> Diagnostics;
	const TOptional<FMjcfModel> Model = Parse(TEXT(R"(<mujoco>
  <asset>
    <mesh name="inline" vertex="0 0 0 1 0 0 0 1 0"/>
    <mesh name="shifted" file="s.obj" refpos="0 0 1"/>
    <hfield name="terrain" nrow="2" ncol="2" size="1 1 1 1"/>
    <texture name="blank" type="2d"/>
    <material name="uses_blank" texture="blank"/>
  </asset>
  <worldbody>
    <light pos="0 0 1"/>
    <replicate count="2"><geom size="1"/></replicate>
    <geom type="mesh" mesh="shifted"/>
    <geom type="mesh" mesh="inline"/>
    <geom type="sphere" size="1" fromto="0 0 0 1 0 0"/>
    <body name="b"><joint name="j" springdamper="0.1 1"/><site name="s"/></body>
  </worldbody>
  <sensor/>
  <equality/>
  <tendon/>
  <contact><pair geom1="a" geom2="b"/></contact>
  <actuator>
    <muscle joint="j"/>
    <motor site="s"/>
    <general joint="j" gaintype="muscle"/>
    <position joint="j"/>
  </actuator>
</mujoco>)"), Diagnostics);
	if (!TestTrue(TEXT("Warnings don't fail the parse: ") + Describe(Diagnostics), Model.IsSet()))
	{
		return false;
	}

	TestEqual(TEXT("Only the mesh with a file is kept"), Model->Meshes.Num(), 1);
	TestTrue(TEXT("Mesh without a file is reported"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("without a file")));
	TestTrue(TEXT("Ignored attribute is reported"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("refpos")));
	TestTrue(TEXT("Other assets are noted"), HasDiagnostic(Diagnostics, EMjcfSeverity::Info, TEXT("<hfield>")));
	TestTrue(TEXT("Texture without a file or pattern is reported"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("need a file or a builtin")));
	TestTrue(TEXT("Material using a skipped texture is reported"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("which was skipped; the material")));
	TestTrue(TEXT("Geom using a skipped mesh is reported"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("which was skipped; the geom")));
	TestTrue(TEXT("fromto on a sphere is reported"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("fromto only applies")));
	TestTrue(TEXT("Body contents that aren't imported are noted"), HasDiagnostic(Diagnostics, EMjcfSeverity::Info, TEXT("<light>")));
	TestTrue(TEXT("Elements that create bodies or geoms are warned about"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("<replicate> is not supported")));
	TestTrue(TEXT("Sections that aren't imported are noted"), HasDiagnostic(Diagnostics, EMjcfSeverity::Info, TEXT("<sensor>")));
	TestTrue(TEXT("Equality constraints are warned about"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("<equality>")));
	TestTrue(TEXT("Tendons are warned about"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("<tendon>")));
	TestTrue(TEXT("Contact pairs are warned about"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("<pair>")));
	TestTrue(TEXT("Unsupported actuator types are warned about"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("<muscle> actuators")));
	TestTrue(TEXT("Actuators on sites are warned about"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("drive a site")));
	TestTrue(TEXT("Muscle gains are warned about"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("Muscle and user-defined")));
	TestTrue(TEXT("springdamper is warned about"), HasDiagnostic(Diagnostics, EMjcfSeverity::Warning, TEXT("springdamper")));
	TestEqual(TEXT("Only the supported actuator is kept"), Model->Actuators.Num(), 1);
	TestEqual(TEXT("The geom with a skipped mesh is dropped"), Model->WorldBody.Geoms.Num(), 2);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserPhysicsTest, "Schola.Mjcf.Parser.Physics", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserPhysicsTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	TArray<FMjcfDiagnostic> Diagnostics;
	const TOptional<FMjcfModel> Model = Parse(TEXT(R"(<mujoco>
  <compiler angle="degree" inertiafromgeom="true"/>
  <option gravity="0 0 -5" timestep="0.001"/>
  <default>
    <joint damping="0.5"/>
    <geom friction="0.8" contype="0"/>
    <default class="arm">
      <joint armature="0.1" range="-90 90"/>
      <motor ctrlrange="-1 1"/>
      <position kp="50"/>
    </default>
    <default class="lift">
      <general gainprm="400" biasprm="0 -200 -100"/>
    </default>
  </default>
  <worldbody>
    <body name="target" mocap="true"/>
    <body name="cart" pos="0 0 1" gravcomp="1">
      <freejoint name="free"/>
      <inertial pos="0 0 0.1" mass="2" diaginertia="0.1 0.2 0.3"/>
      <geom type="box" size="0.1 0.1 0.1" mass="3" conaffinity="2" friction="0.5 0.01"/>
      <body name="arm" childclass="arm">
        <frame pos="0 0 0.2" euler="0 0 90">
          <joint name="shoulder" pos="0.1 0 0" axis="1 0 0" ref="30" frictionloss="2"/>
        </frame>
        <geom type="capsule" size="0.02 0.1" density="500"/>
        <body name="slider">
          <joint name="slide" class="main" type="slide" axis="0 0 2" range="0 0.5"/>
          <inertial mass="1" fullinertia="2 2 4 1 0 0"/>
        </body>
      </body>
      <body name="wrist"><joint name="ball" type="ball" range="0 45"/></body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m" joint="shoulder" class="arm" gear="10"/>
    <position name="p" joint="shoulder" class="arm" timeconst="0.1"/>
    <velocity name="v" joint="slide" kv="20" forcerange="-5 5"/>
    <general name="g" joint="slide" gainprm="7" biastype="affine" biasprm="0 -7 -1" dyntype="filter" dynprm="0.05"/>
    <position name="lift" joint="slide" class="lift"/>
    <position name="inherit" joint="shoulder" inheritrange="0.5" dampratio="1"/>
  </actuator>
  <contact><exclude body1="cart" body2="arm"/></contact>
</mujoco>)"), Diagnostics);
	if (!TestTrue(TEXT("Parses: ") + Describe(Diagnostics), Model.IsSet()))
	{
		return false;
	}

	TestEqual(TEXT("Gravity"), Model->Gravity, FVector(0, 0, -5));
	TestEqual(TEXT("Timestep"), Model->Timestep, 0.001);
	TestEqual(TEXT("inertiafromgeom"), (int32)Model->InertiaFromGeom, (int32)EMjcfInertiaFromGeom::True);

	const FMjcfBody& World = Model->WorldBody;
	if (!TestEqual(TEXT("World bodies"), World.Bodies.Num(), 2))
	{
		return false;
	}
	TestTrue(TEXT("Mocap body"), World.Bodies[0].bMocap);
	TestTrue(TEXT("Mocap bodies need no joints"), World.Bodies[0].Joints.IsEmpty());

	const FMjcfBody& Cart = World.Bodies[1];
	TestEqual(TEXT("gravcomp"), Cart.GravComp, 1.0);
	if (TestEqual(TEXT("Cart joints"), Cart.Joints.Num(), 1))
	{
		TestEqual(TEXT("<freejoint> is a free joint"), (int32)Cart.Joints[0].Type, (int32)EMjcfJointType::Free);
		TestEqual(TEXT("<freejoint> takes no defaults"), Cart.Joints[0].Damping, 0.0);
	}
	if (TestTrue(TEXT("Cart inertial"), Cart.Inertial.IsSet()))
	{
		TestEqual(TEXT("Inertial position"), Cart.Inertial->Position, FVector(0, 0, 0.1));
		TestEqual(TEXT("Inertial mass"), Cart.Inertial->Mass, 2.0);
		TestEqual(TEXT("diaginertia"), Cart.Inertial->DiagInertia, FVector(0.1, 0.2, 0.3));
	}
	if (TestEqual(TEXT("Cart geoms"), Cart.Geoms.Num(), 1))
	{
		const FMjcfGeom& Box = Cart.Geoms[0];
		TestEqual(TEXT("contype from the default"), Box.ConType, 0);
		TestEqual(TEXT("conaffinity"), Box.ConAffinity, 2);
		TestEqual(TEXT("Partial friction keeps MuJoCo's other values"), Box.Friction, FVector(0.5, 0.01, 0.0001));
		TestTrue(TEXT("Geom mass"), Box.Mass.IsSet() && *Box.Mass == 3.0);
	}
	if (!TestEqual(TEXT("Cart child bodies"), Cart.Bodies.Num(), 2))
	{
		return false;
	}

	const FMjcfBody& Arm = Cart.Bodies[0];
	TestFalse(TEXT("No <inertial> means it comes from the geoms"), Arm.Inertial.IsSet());
	if (TestEqual(TEXT("Arm joints"), Arm.Joints.Num(), 1))
	{
		const FMjcfJoint& Shoulder = Arm.Joints[0];
		TestEqual(TEXT("Hinge by default"), (int32)Shoulder.Type, (int32)EMjcfJointType::Hinge);
		TestEqual(TEXT("Frame places the joint"), Shoulder.Position, FVector(0, 0.1, 0.2), 1e-9f);
		TestEqual(TEXT("Frame turns the axis"), Shoulder.Axis, FVector(0, 1, 0), 1e-9f);
		TestTrue(TEXT("Range given means limited"), Shoulder.bLimited);
		TestTrue(TEXT("Hinge range in radians"), Shoulder.Range.Equals(FVector2D(-UE_DOUBLE_HALF_PI, UE_DOUBLE_HALF_PI), 1e-9));
		TestEqual(TEXT("Hinge ref in radians"), Shoulder.Ref, FMath::DegreesToRadians(30.0), 1e-9);
		TestEqual(TEXT("Class armature"), Shoulder.Armature, 0.1);
		TestEqual(TEXT("Main damping is inherited"), Shoulder.Damping, 0.5);
		TestEqual(TEXT("frictionloss"), Shoulder.FrictionLoss, 2.0);
	}
	if (TestEqual(TEXT("Arm geoms"), Arm.Geoms.Num(), 1))
	{
		TestEqual(TEXT("Density"), Arm.Geoms[0].Density, 500.0);
		TestFalse(TEXT("No mass when density is used"), Arm.Geoms[0].Mass.IsSet());
		TestEqual(TEXT("childclass inherits main's friction"), Arm.Geoms[0].Friction.X, 0.8);
	}
	if (TestEqual(TEXT("Arm child bodies"), Arm.Bodies.Num(), 1) && TestEqual(TEXT("Slider joints"), Arm.Bodies[0].Joints.Num(), 1))
	{
		const FMjcfBody& Slider = Arm.Bodies[0];
		const FMjcfJoint& Slide = Slider.Joints[0];
		TestEqual(TEXT("Slide joint"), (int32)Slide.Type, (int32)EMjcfJointType::Slide);
		TestEqual(TEXT("Axis is normalized"), Slide.Axis, FVector(0, 0, 1));
		TestEqual(TEXT("class=\"main\" skips the childclass"), Slide.Armature, 0.0);
		TestTrue(TEXT("Slide range is not converted"), Slide.bLimited && Slide.Range == FVector2D(0, 0.5));
		if (TestTrue(TEXT("Slider inertial"), Slider.Inertial.IsSet()))
		{
			// The principal moments of [[2 1 0] [1 2 0] [0 0 4]] are 1, 3, and 4; rotating them back must give the matrix.
			const FMjcfInertial& Inertial = *Slider.Inertial;
			const double Expected[3][3] = { { 2, 1, 0 }, { 1, 2, 0 }, { 0, 0, 4 } };
			double Rebuilt[3][3] = {};
			for (int32 Axis = 0; Axis < 3; ++Axis)
			{
				FVector Unit = FVector::ZeroVector;
				Unit[Axis] = 1;
				const FVector Direction = Inertial.Rotation.RotateVector(Unit);
				for (int32 Row = 0; Row < 3; ++Row)
				{
					for (int32 Column = 0; Column < 3; ++Column)
					{
						Rebuilt[Row][Column] += Inertial.DiagInertia[Axis] * Direction[Row] * Direction[Column];
					}
				}
			}
			for (int32 Row = 0; Row < 3; ++Row)
			{
				for (int32 Column = 0; Column < 3; ++Column)
				{
					TestEqual(FString::Printf(TEXT("fullinertia [%d][%d]"), Row, Column), Rebuilt[Row][Column], Expected[Row][Column], 1e-9);
				}
			}
			TestEqual(TEXT("Principal moments sum to the trace"), Inertial.DiagInertia.X + Inertial.DiagInertia.Y + Inertial.DiagInertia.Z, 8.0, 1e-9);
		}
	}
	if (TestEqual(TEXT("Wrist joints"), Cart.Bodies[1].Joints.Num(), 1))
	{
		const FMjcfJoint& Ball = Cart.Bodies[1].Joints[0];
		TestEqual(TEXT("Ball joint"), (int32)Ball.Type, (int32)EMjcfJointType::Ball);
		TestEqual(TEXT("Ball range in radians"), Ball.Range.Y, UE_DOUBLE_PI / 4, 1e-9);
	}

	if (!TestEqual(TEXT("Actuators"), Model->Actuators.Num(), 6))
	{
		return false;
	}
	const FMjcfActuator& Motor = Model->Actuators[0];
	TestEqual(TEXT("Motor joint"), Motor.Joint, FString(TEXT("shoulder")));
	TestEqual(TEXT("Motor gear"), Motor.Gear, 10.0);
	TestEqual(TEXT("Motor gain"), Motor.GainPrm, FVector(1, 0, 0));
	TestEqual(TEXT("Motor has no bias"), (int32)Motor.BiasType, (int32)EMjcfActuatorBias::None);
	TestTrue(TEXT("Motor ctrlrange from the class"), Motor.bCtrlLimited && Motor.CtrlRange == FVector2D(-1, 1));

	const FMjcfActuator& Position = Model->Actuators[1];
	TestEqual(TEXT("Position kind"), Position.Kind, FString(TEXT("position")));
	TestEqual(TEXT("Position gain is kp"), Position.GainPrm, FVector(50, 0, 0));
	TestEqual(TEXT("Position bias pulls toward the control"), Position.BiasPrm, FVector(0, -50, 0));
	TestEqual(TEXT("timeconst adds a filter"), (int32)Position.Dynamics, (int32)EMjcfActuatorDynamics::FilterExact);
	TestEqual(TEXT("Filter time constant"), Position.TimeConstant, 0.1);
	TestTrue(TEXT("<motor> defaults apply to <position> in the same class, as in MuJoCo"), Position.bCtrlLimited && Position.CtrlRange == FVector2D(-1, 1));

	const FMjcfActuator& Velocity = Model->Actuators[2];
	TestEqual(TEXT("Velocity gain is kv"), Velocity.GainPrm, FVector(20, 0, 0));
	TestEqual(TEXT("Velocity bias damps"), Velocity.BiasPrm, FVector(0, 0, -20));
	TestFalse(TEXT("No ctrlrange means unlimited"), Velocity.bCtrlLimited);
	TestTrue(TEXT("forcerange means force limited"), Velocity.bForceLimited && Velocity.ForceRange == FVector2D(-5, 5));

	const FMjcfActuator& General = Model->Actuators[3];
	TestEqual(TEXT("Partial gainprm is padded with zeros"), General.GainPrm, FVector(7, 0, 0));
	TestEqual(TEXT("General bias"), General.BiasPrm, FVector(0, -7, -1));
	TestEqual(TEXT("General dynamics"), (int32)General.Dynamics, (int32)EMjcfActuatorDynamics::Filter);
	TestEqual(TEXT("dynprm"), General.TimeConstant, 0.05);

	const FMjcfActuator& Lift = Model->Actuators[4];
	TestEqual(TEXT("kp defaults to the class's <general> gain, as in MuJoCo"), Lift.GainPrm, FVector(400, 0, 0));
	TestEqual(TEXT("kp replaces the position term and the class's velocity term stays"), Lift.BiasPrm, FVector(0, -400, -100));

	const FMjcfActuator& Inherit = Model->Actuators[5];
	TestTrue(TEXT("inheritrange limits the control"), Inherit.bCtrlLimited);
	TestTrue(TEXT("inheritrange takes that fraction of the joint's range, about its middle"),
		Inherit.CtrlRange.Equals(FVector2D(-UE_DOUBLE_PI / 4, UE_DOUBLE_PI / 4), 1e-9));
	TestEqual(TEXT("dampratio is kept for later"), Inherit.DampRatio, 1.0);
	TestEqual(TEXT("dampratio leaves the velocity term to be computed"), Inherit.BiasPrm, FVector(0, -1, 0));

	if (TestEqual(TEXT("Contact excludes"), Model->ContactExcludes.Num(), 1))
	{
		TestEqual(TEXT("Exclude body1"), Model->ContactExcludes[0].Body1, FString(TEXT("cart")));
		TestEqual(TEXT("Exclude body2"), Model->ContactExcludes[0].Body2, FString(TEXT("arm")));
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserRecognizesFilesTest, "Schola.Mjcf.Parser.RecognizesFiles", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserRecognizesFilesTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	const auto Write = [](const TCHAR* Name, const TCHAR* Contents) {
		const FString Path = ModelDir() / TEXT("Recognize") / Name;
		FFileHelper::SaveStringToFile(Contents, *Path);
		return Path;
	};

	TestTrue(TEXT("MJCF model"), FMjcfParser::IsMjcfFile(Write(TEXT("model.xml"), TEXT(R"(<?xml version="1.0"?>
<!-- A robot -->
<mujoco model="m"/>)"))));
	TestFalse(TEXT("Other XML"), FMjcfParser::IsMjcfFile(Write(TEXT("other.xml"), TEXT("<robot name=\"r\"/>"))));
	TestFalse(TEXT("Root tag is case-sensitive"), FMjcfParser::IsMjcfFile(Write(TEXT("upper.xml"), TEXT("<MuJoCo/>"))));
	TestFalse(TEXT("Invalid XML"), FMjcfParser::IsMjcfFile(Write(TEXT("broken.xml"), TEXT("<mujoco>"))));
	TestFalse(TEXT("Missing file"), FMjcfParser::IsMjcfFile(ModelDir() / TEXT("Recognize/missing.xml")));
	return true;
}

namespace MjcfParserTest
{
	FString JsonNumbers(std::initializer_list<double> Values)
	{
		TArray<FString> Parts;
		for (const double Value : Values)
		{
			Parts.Add(FString::Printf(TEXT("%.9g"), Value));
		}
		return TEXT("[") + FString::Join(Parts, TEXT(",")) + TEXT("]");
	}

	struct FCorpusDump
	{
		TArray<FString> Bodies;
		TArray<FString> Geoms;
		TArray<FString> Joints;
	};

	/** Appends Body and its descendants in MuJoCo's body order, with world placements, as JSON objects. */
	void DumpBody(const FMjcfBody& Body, const FVector& ParentPosition, const FQuat& ParentRotation, FCorpusDump& Out)
	{
		const FVector Position = ParentPosition + ParentRotation.RotateVector(Body.Position);
		const FQuat Rotation = ParentRotation * Body.Rotation;
		const int32 BodyIndex = Out.Bodies.Num();
		FString Inertial = TEXT("null");
		if (Body.Inertial)
		{
			const FMjcfInertial& I = *Body.Inertial;
			Inertial = FString::Printf(TEXT("{\"pos\":%s,\"quat\":%s,\"mass\":%.9g,\"diag\":%s}"), *JsonNumbers({ I.Position.X, I.Position.Y, I.Position.Z }),
				*JsonNumbers({ I.Rotation.W, I.Rotation.X, I.Rotation.Y, I.Rotation.Z }), I.Mass, *JsonNumbers({ I.DiagInertia.X, I.DiagInertia.Y, I.DiagInertia.Z }));
		}
		Out.Bodies.Add(FString::Printf(TEXT("{\"name\":\"%s\",\"pos\":%s,\"quat\":%s,\"mocap\":%d,\"gravcomp\":%.9g,\"inertial\":%s}"), *Body.Name.ReplaceCharWithEscapedChar(),
			*JsonNumbers({ Position.X, Position.Y, Position.Z }), *JsonNumbers({ Rotation.W, Rotation.X, Rotation.Y, Rotation.Z }), Body.bMocap ? 1 : 0, Body.GravComp,
			*Inertial));
		for (const FMjcfJoint& Joint : Body.Joints)
		{
			Out.Joints.Add(FString::Printf(TEXT("{\"name\":\"%s\",\"body\":%d,\"type\":%d,\"pos\":%s,\"axis\":%s,\"limited\":%d,\"range\":%s,\"ref\":%.9g,\"springref\":%.9g,")
				TEXT("\"stiffness\":%.9g,\"damping\":%.9g,\"armature\":%.9g,\"frictionloss\":%.9g}"),
				*Joint.Name.ReplaceCharWithEscapedChar(), BodyIndex, (int32)Joint.Type, *JsonNumbers({ Joint.Position.X, Joint.Position.Y, Joint.Position.Z }),
				*JsonNumbers({ Joint.Axis.X, Joint.Axis.Y, Joint.Axis.Z }), Joint.bLimited ? 1 : 0, *JsonNumbers({ Joint.Range.X, Joint.Range.Y }), Joint.Ref, Joint.SpringRef,
				Joint.Stiffness, Joint.Damping, Joint.Armature, Joint.FrictionLoss));
		}
		for (const FMjcfGeom& Geom : Body.Geoms)
		{
			const FVector GeomPosition = Position + Rotation.RotateVector(Geom.Position);
			const FQuat GeomRotation = Rotation * Geom.Rotation;
			Out.Geoms.Add(FString::Printf(TEXT("{\"body\":%d,\"type\":%d,\"pos\":%s,\"quat\":%s,\"size\":%s,\"group\":%d,\"rgba\":%s,\"material\":\"%s\",\"mesh\":\"%s\",")
				TEXT("\"contype\":%d,\"conaffinity\":%d,\"friction\":%s}"),
				BodyIndex, (int32)Geom.Type, *JsonNumbers({ GeomPosition.X, GeomPosition.Y, GeomPosition.Z }),
				*JsonNumbers({ GeomRotation.W, GeomRotation.X, GeomRotation.Y, GeomRotation.Z }), *JsonNumbers({ Geom.Size.X, Geom.Size.Y, Geom.Size.Z }), Geom.Group,
				*JsonNumbers({ Geom.Rgba.X, Geom.Rgba.Y, Geom.Rgba.Z, Geom.Rgba.W }), *Geom.Material.ReplaceCharWithEscapedChar(), *Geom.Mesh.ReplaceCharWithEscapedChar(),
				Geom.ConType, Geom.ConAffinity, *JsonNumbers({ Geom.Friction.X, Geom.Friction.Y, Geom.Friction.Z })));
		}
		for (const FMjcfBody& Child : Body.Bodies)
		{
			DumpBody(Child, Position, Rotation, Out);
		}
	}

	FString DumpActuator(const FMjcfActuator& Actuator)
	{
		return FString::Printf(TEXT("{\"name\":\"%s\",\"joint\":\"%s\",\"gear\":%.9g,\"dyntype\":%d,\"gaintype\":%d,\"biastype\":%d,\"timeconst\":%.9g,\"gainprm\":%s,")
			TEXT("\"biasprm\":%s,\"dampratio\":%.9g,\"ctrllimited\":%d,\"ctrlrange\":%s,\"forcelimited\":%d,\"forcerange\":%s}"),
			*Actuator.Name.ReplaceCharWithEscapedChar(), *Actuator.Joint.ReplaceCharWithEscapedChar(), Actuator.Gear, (int32)Actuator.Dynamics, (int32)Actuator.GainType,
			(int32)Actuator.BiasType, Actuator.TimeConstant, *JsonNumbers({ Actuator.GainPrm.X, Actuator.GainPrm.Y, Actuator.GainPrm.Z }),
			*JsonNumbers({ Actuator.BiasPrm.X, Actuator.BiasPrm.Y, Actuator.BiasPrm.Z }), Actuator.DampRatio, Actuator.bCtrlLimited ? 1 : 0,
			*JsonNumbers({ Actuator.CtrlRange.X, Actuator.CtrlRange.Y }), Actuator.bForceLimited ? 1 : 0, *JsonNumbers({ Actuator.ForceRange.X, Actuator.ForceRange.Y }));
	}
} // namespace MjcfParserTest

/**
 * Parses every MJCF file under the folder named by SCHOLA_MJCF_CORPUS, such as a MuJoCo Menagerie checkout. When
 * SCHOLA_MJCF_CORPUS_OUT names a folder, writes each model's world placements there for comparison with MuJoCo.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMjcfParserCorpusTest, "Schola.Mjcf.Parser.Corpus", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FMjcfParserCorpusTest::RunTest(const FString& Parameters)
{
	using namespace MjcfParserTest;

	const FString Corpus = FPlatformMisc::GetEnvironmentVariable(TEXT("SCHOLA_MJCF_CORPUS"));
	if (Corpus.IsEmpty())
	{
		AddInfo(TEXT("Set SCHOLA_MJCF_CORPUS to a folder of MJCF files to run this test"));
		return true;
	}

	const FString OutDir = FPlatformMisc::GetEnvironmentVariable(TEXT("SCHOLA_MJCF_CORPUS_OUT"));
	TArray<FString> Files;
	IFileManager::Get().FindFilesRecursive(Files, *Corpus, TEXT("*.xml"), /*Files=*/true, /*Directories=*/false);
	Files.Sort();
	int32 Parsed = 0;
	int32 Failed = 0;
	for (const FString& File : Files)
	{
		if (!FMjcfParser::IsMjcfFile(File))
		{
			continue;
		}
		++Parsed;
		TArray<FMjcfDiagnostic> Diagnostics;
		const TOptional<FMjcfModel> Model = FMjcfParser::ParseFile(File, Diagnostics);
		if (!Model)
		{
			++Failed;
			AddError(FString::Printf(TEXT("%s:\n%s"), *File, *Describe(Diagnostics)));
			continue;
		}
		if (!OutDir.IsEmpty())
		{
			FCorpusDump Dump;
			DumpBody(Model->WorldBody, FVector::ZeroVector, FQuat::Identity, Dump);
			TArray<FString> Actuators;
			for (const FMjcfActuator& Actuator : Model->Actuators)
			{
				Actuators.Add(DumpActuator(Actuator));
			}
			FString Relative = File;
			FPaths::MakePathRelativeTo(Relative, *(Corpus / TEXT("")));
			const FString Json = FString::Printf(
				TEXT("{\"file\":\"%s\",\"gravity\":%s,\"timestep\":%.9g,\"inertiafromgeom\":%d,\"bodies\":[%s],\"geoms\":[%s],\"joints\":[%s],\"actuators\":[%s]}"),
				*File.ReplaceCharWithEscapedChar(), *JsonNumbers({ Model->Gravity.X, Model->Gravity.Y, Model->Gravity.Z }), Model->Timestep, (int32)Model->InertiaFromGeom,
				*FString::Join(Dump.Bodies, TEXT(",")), *FString::Join(Dump.Geoms, TEXT(",")), *FString::Join(Dump.Joints, TEXT(",")), *FString::Join(Actuators, TEXT(",")));
			FFileHelper::SaveStringToFile(Json, *(OutDir / Relative.Replace(TEXT("/"), TEXT("__")) + TEXT(".json")));
		}
	}
	AddInfo(FString::Printf(TEXT("Parsed %d MJCF files, %d failed"), Parsed, Failed));
	return true;
}

#endif
