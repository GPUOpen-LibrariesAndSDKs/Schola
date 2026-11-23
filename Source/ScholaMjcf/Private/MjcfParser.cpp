// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "MjcfParser.h"
#include "Algo/Find.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "MjcfMath.h"
#include "XmlFile.h"

#include <cstdlib>

namespace
{
	const TCHAR* const MainClass = TEXT("main");

	const TCHAR* const OrientationAttributes[] = { TEXT("quat"), TEXT("axisangle"), TEXT("euler"), TEXT("xyaxes"), TEXT("zaxis") };

	using FAttributes = TMap<FString, FString>;

	/** An XML element copied out of FXmlFile, so that included files can be spliced into the tree. */
	struct FElement
	{
		FString Tag;
		FAttributes Attributes;
		TArray<FElement> Children;

		/** The file the element was read from, which differs from the main file for included elements. */
		FString SourceFile;

		/** The element's location within its own file, for example "mujoco/asset[#0]/mesh[arm]". */
		FString Path;

		bool Is(const TCHAR* InTag) const
		{
			return Tag.Equals(InTag, ESearchCase::CaseSensitive);
		}
	};

	enum class EAutoFlag : uint8
	{
		False,
		True,
		Auto,
	};

	/** An actuator while its defaults and attributes are being applied. */
	struct FActuatorState
	{
		FMjcfActuator Actuator;
		EAutoFlag CtrlLimited = EAutoFlag::Auto;
		EAutoFlag ForceLimited = EAutoFlag::Auto;
		bool bCtrlRangeGiven = false;
		bool bForceRangeGiven = false;
		double InheritRange = 0;
	};

	/** The actuator elements that are imported; the others are skipped with a warning. */
	const TCHAR* const ActuatorTags[] = { TEXT("general"), TEXT("motor"), TEXT("position"), TEXT("velocity") };

	bool IsActuatorTag(const FString& Tag)
	{
		return Algo::FindByPredicate(ActuatorTags, [&Tag](const TCHAR* Name) { return Tag.Equals(Name, ESearchCase::CaseSensitive); }) != nullptr;
	}

	/** The attributes a <default> class sets for each element type, including those inherited from its parent classes. */
	struct FDefaultClass
	{
		FString Name;
		TMap<FString, FAttributes> AttributesByTag;

		/**
		 * MuJoCo keeps one actuator default per class, in general form, which each actuator element in the class changes in
		 * turn, so a <position> after a <general> keeps the parameters kp doesn't set.
		 */
		FActuatorState Actuator;
	};

	/** A frame that positions what it contains, such as a body's own frame or a <frame> element within it. */
	struct FFramePlacement
	{
		FVector Position = FVector::ZeroVector;
		FQuat Rotation = FQuat::Identity;

		/** Returns a placement given relative to this one, relative to this one's parent instead. */
		FFramePlacement Compose(const FFramePlacement& Child) const
		{
			return { Position + Rotation.RotateVector(Child.Position), Rotation * Child.Rotation };
		}
	};

	FString NormalizePath(const FString& InPath)
	{
		FString Path = InPath;
		FPaths::NormalizeFilename(Path);
		FPaths::CollapseRelativeDirectories(Path);
		return FPaths::ConvertRelativePathToFull(Path);
	}

	/** Parses Token as a double, failing unless the whole token is a finite number. */
	bool TryParseDouble(const FString& Token, double& OutValue)
	{
		const auto Ansi = StringCast<ANSICHAR>(*Token);
		char* End = nullptr;
		OutValue = std::strtod(Ansi.Get(), &End);
		return End != Ansi.Get() && *End == '\0' && FMath::IsFinite(OutValue);
	}

	bool ParseDoubles(const FString& Text, TArray<double>& Out)
	{
		TArray<FString> Tokens;
		Text.ParseIntoArrayWS(Tokens);
		for (const FString& Token : Tokens)
		{
			double Value = 0.0;
			if (!TryParseDouble(Token, Value))
			{
				return false;
			}
			Out.Add(Value);
		}
		return true;
	}

	FString DescribeCount(int32 MinCount, int32 MaxCount)
	{
		static const TCHAR* const Words[] = { TEXT("zero"), TEXT("one"), TEXT("two"), TEXT("three"), TEXT("four"), TEXT("five"), TEXT("six") };
		const TCHAR* const MaxWord = Words[FMath::Min(MaxCount, 6)];
		if (MinCount == MaxCount)
		{
			return FString::Printf(TEXT("%s number%s"), MaxWord, MaxCount == 1 ? TEXT("") : TEXT("s"));
		}
		return FString::Printf(TEXT("%s to %s numbers"), Words[FMath::Min(MinCount, 6)], MaxWord);
	}

	void ConvertNode(const FXmlNode& Node, const FString& SourceFile, FElement& Out)
	{
		Out.Tag = Node.GetTag();
		Out.SourceFile = SourceFile;
		for (const FXmlAttribute& Attribute : Node.GetAttributes())
		{
			Out.Attributes.Add(Attribute.GetTag(), Attribute.GetValue());
		}
		for (const FXmlNode* Child : Node.GetChildrenNodes())
		{
			if (Child)
			{
				ConvertNode(*Child, SourceFile, Out.Children.AddDefaulted_GetRef());
			}
		}
	}

	void AssignPaths(FElement& Element)
	{
		TMap<FString, int32> UnnamedCounts;
		for (FElement& Child : Element.Children)
		{
			const FString* Name = Child.Attributes.Find(Child.Is(TEXT("default")) ? TEXT("class") : TEXT("name"));
			const FString Label = Name ? *Name : FString::Printf(TEXT("#%d"), UnnamedCounts.FindOrAdd(Child.Tag)++);
			Child.Path = FString::Printf(TEXT("%s/%s[%s]"), *Element.Path, *Child.Tag, *Label);
			AssignPaths(Child);
		}
	}

	bool HasOrientation(const FAttributes& Attributes)
	{
		for (const TCHAR* Name : OrientationAttributes)
		{
			if (Attributes.Contains(Name))
			{
				return true;
			}
		}
		return false;
	}

	void RemoveOrientation(FAttributes& Attributes)
	{
		for (const TCHAR* Name : OrientationAttributes)
		{
			Attributes.Remove(Name);
		}
	}

	template <typename EnumType>
	struct TEnumName
	{
		const TCHAR* Name;
		EnumType Value;
	};

	class FModelParser
	{
	public:
		FModelParser(const FString& InSourceFile, FMjcfFileReader InReader, TArray<FMjcfDiagnostic>& InDiagnostics)
			: SourceFile(NormalizePath(InSourceFile))
			, ModelDirectory(FPaths::GetPath(SourceFile))
			, Reader(MoveTemp(InReader))
			, Diagnostics(InDiagnostics)
		{
			Model.SourceFile = SourceFile;
		}

		TOptional<FMjcfModel> Parse(const FString& Xml)
		{
			FElement Root;
			if (!LoadDocument(Xml, SourceFile, Root))
			{
				return {};
			}
			IncludedFiles.Add(SourceFile);
			ExpandIncludes(Root);
			Model.ModelName = Root.Attributes.FindRef(TEXT("model"));

			// Compiler settings, options, and defaults apply to the whole model, wherever they appear in the file.
			for (const FElement& Section : Root.Children)
			{
				if (Section.Is(TEXT("compiler")))
				{
					ReadCompiler(Section);
				}
				else if (Section.Is(TEXT("option")))
				{
					ReadOption(Section);
				}
				else if (Section.Is(TEXT("default")))
				{
					ReadDefault(Section, INDEX_NONE);
				}
			}
			for (const FElement& Section : Root.Children)
			{
				if (Section.Is(TEXT("asset")))
				{
					ReadAssets(Section);
				}
			}
			CheckMaterialTextures();

			// Geoms refer to assets by name, so bodies are read once every asset is known.
			for (const FElement& Section : Root.Children)
			{
				if (Section.Is(TEXT("worldbody")))
				{
					if (Model.WorldBody.ElementPath.IsEmpty())
					{
						Model.WorldBody.ElementPath = Section.Path;
					}
					ReadBodyContents(Section, Model.WorldBody, MainClass, FFramePlacement());
				}
			}

			// Actuators and contact exclusions refer to joints and bodies by name, so they are read once every body is known.
			for (const FElement& Section : Root.Children)
			{
				if (Section.Is(TEXT("actuator")))
				{
					ReadActuators(Section);
				}
				else if (Section.Is(TEXT("contact")))
				{
					ReadContact(Section);
				}
				else if (Section.Is(TEXT("equality")))
				{
					ReportOnce(TEXT("section.equality"), EMjcfSeverity::Warning, Section,
						TEXT("<equality> constraints are not imported, so the joints and bodies they connect move independently (reported once)"));
				}
				else if (Section.Is(TEXT("tendon")))
				{
					ReportOnce(TEXT("section.tendon"), EMjcfSeverity::Warning, Section,
						TEXT("<tendon> is not imported, so tendons and anything that uses them are missing (reported once)"));
				}
				else if (!Section.Is(TEXT("compiler")) && !Section.Is(TEXT("option")) && !Section.Is(TEXT("default")) && !Section.Is(TEXT("asset"))
					&& !Section.Is(TEXT("worldbody")))
				{
					ReportOnce(TEXT("section.") + Section.Tag, EMjcfSeverity::Info, Section,
						FString::Printf(TEXT("<%s> is not imported (reported once)"), *Section.Tag));
				}
			}

			if (bHasErrors)
			{
				return {};
			}
			return Model;
		}

	private:
		bool LoadDocument(const FString& Contents, const FString& File, FElement& OutRoot)
		{
			FXmlFile Xml(Contents, EConstructMethod::ConstructFromBuffer);
			if (!Xml.IsValid() || !Xml.GetRootNode())
			{
				Report(EMjcfSeverity::Error, File, FString(), FString::Printf(TEXT("Not valid XML: %s"), *Xml.GetLastError()));
				return false;
			}
			ConvertNode(*Xml.GetRootNode(), File, OutRoot);
			OutRoot.Path = OutRoot.Tag;
			AssignPaths(OutRoot);
			if (!OutRoot.Is(TEXT("mujoco")))
			{
				Error(OutRoot, FString::Printf(TEXT("The root element must be <mujoco>, found <%s>"), *OutRoot.Tag));
				return false;
			}
			return true;
		}

		void ExpandIncludes(FElement& Element)
		{
			TArray<FElement> Expanded;
			Expanded.Reserve(Element.Children.Num());
			for (FElement& Child : Element.Children)
			{
				if (!Child.Is(TEXT("include")))
				{
					ExpandIncludes(Child);
					Expanded.Add(MoveTemp(Child));
					continue;
				}
				FElement Included;
				if (LoadInclude(Child, Included))
				{
					ExpandIncludes(Included);
					Expanded.Append(MoveTemp(Included.Children));
				}
			}
			Element.Children = MoveTemp(Expanded);
		}

		bool LoadInclude(const FElement& Include, FElement& OutRoot)
		{
			const FString* File = Include.Attributes.Find(TEXT("file"));
			if (!File || File->IsEmpty())
			{
				Error(Include, TEXT("<include> needs a file attribute"));
				return false;
			}

			// Like MuJoCo, look next to the main model file first and then next to the including file.
			FString Path = NormalizePath(FPaths::IsRelative(*File) ? FPaths::Combine(ModelDirectory, *File) : *File);
			FString Contents;
			bool bRead = Reader(Path, Contents);
			if (!bRead && FPaths::IsRelative(*File) && Include.SourceFile != SourceFile)
			{
				const FString NextToIncluder = NormalizePath(FPaths::Combine(FPaths::GetPath(Include.SourceFile), *File));
				bRead = Reader(NextToIncluder, Contents);
				if (bRead)
				{
					Path = NextToIncluder;
				}
			}
			if (!bRead)
			{
				Error(Include, FString::Printf(TEXT("Could not read included file '%s'"), *Path));
				return false;
			}

			bool bAlreadyIncluded = false;
			IncludedFiles.Add(Path, &bAlreadyIncluded);
			if (bAlreadyIncluded)
			{
				Error(Include, FString::Printf(TEXT("'%s' is included more than once"), **File));
				return false;
			}
			return LoadDocument(Contents, Path, OutRoot);
		}

		void ReadCompiler(const FElement& Compiler)
		{
			if (const FString* Value = Compiler.Attributes.Find(TEXT("meshdir")))
			{
				MeshDir = *Value;
			}
			if (const FString* Value = Compiler.Attributes.Find(TEXT("texturedir")))
			{
				TextureDir = *Value;
			}
			if (const FString* Value = Compiler.Attributes.Find(TEXT("assetdir")))
			{
				AssetDir = *Value;
			}
			if (const FString* Value = Compiler.Attributes.Find(TEXT("angle")))
			{
				if (Value->Equals(TEXT("degree"), ESearchCase::CaseSensitive) || Value->Equals(TEXT("radian"), ESearchCase::CaseSensitive))
				{
					bDegrees = Value->Equals(TEXT("degree"), ESearchCase::CaseSensitive);
				}
				else
				{
					Error(Compiler, FString::Printf(TEXT("angle must be 'degree' or 'radian', got '%s'"), **Value));
				}
			}
			if (const FString* Value = Compiler.Attributes.Find(TEXT("eulerseq")))
			{
				bool bValid = Value->Len() == 3;
				for (int32 Index = 0; bValid && Index < 3; ++Index)
				{
					bValid = FCString::Strchr(TEXT("xyzXYZ"), (*Value)[Index]) != nullptr;
				}
				if (bValid)
				{
					EulerSequence = *Value;
				}
				else
				{
					Error(Compiler, FString::Printf(TEXT("eulerseq must be three of the letters xyzXYZ, got '%s'"), **Value));
				}
			}
			static const TEnumName<EMjcfInertiaFromGeom> InertiaFromGeom[] = {
				{ TEXT("false"), EMjcfInertiaFromGeom::False }, { TEXT("true"), EMjcfInertiaFromGeom::True }, { TEXT("auto"), EMjcfInertiaFromGeom::Auto },
			};
			ReadBool(Compiler, Compiler.Attributes, TEXT("autolimits"), bAutoLimits);
			ReadEnum(Compiler, Compiler.Attributes, TEXT("inertiafromgeom"), InertiaFromGeom, Model.InertiaFromGeom);
		}

		void ReadOption(const FElement& Option)
		{
			ReadVector(Option, Option.Attributes, TEXT("gravity"), Model.Gravity);
			ReadDouble(Option, Option.Attributes, TEXT("timestep"), Model.Timestep);
			if (Model.Timestep <= 0)
			{
				Error(Option, TEXT("timestep must be positive"));
			}
		}

		void ReadDefault(const FElement& Element, int32 ParentIndex)
		{
			FString Name = MainClass;
			if (ParentIndex != INDEX_NONE)
			{
				const FString* ClassName = Element.Attributes.Find(TEXT("class"));
				if (!ClassName || ClassName->IsEmpty())
				{
					Error(Element, TEXT("Nested <default> elements need a class attribute"));
					return;
				}
				Name = *ClassName;
			}

			int32 Index = FindClass(Name);
			if (Index != INDEX_NONE && ParentIndex != INDEX_NONE)
			{
				Error(Element, FString::Printf(TEXT("Default class \"%s\" is defined more than once"), *Name));
				return;
			}
			if (Index == INDEX_NONE)
			{
				Index = Classes.Num();
				FDefaultClass& Class = Classes.AddDefaulted_GetRef();
				Class.Name = Name;
				if (ParentIndex != INDEX_NONE)
				{
					Class.AttributesByTag = Classes[ParentIndex].AttributesByTag;
					Class.Actuator = Classes[ParentIndex].Actuator;
				}
			}

			// A class's own settings are read before its nested classes, so children inherit them wherever they appear.
			for (const FElement& Child : Element.Children)
			{
				if (Child.Is(TEXT("default")))
				{
					continue;
				}
				if (IsActuatorTag(Child.Tag))
				{
					ApplyActuatorAttributes(Child, *Child.Tag, Child.Attributes, Classes[Index].Actuator);
					continue;
				}
				FAttributes& Attributes = Classes[Index].AttributesByTag.FindOrAdd(Child.Tag);
				if (HasOrientation(Child.Attributes))
				{
					RemoveOrientation(Attributes);
				}
				for (const TPair<FString, FString>& Pair : Child.Attributes)
				{
					if (!Pair.Key.Equals(TEXT("name"), ESearchCase::CaseSensitive))
					{
						Attributes.Add(Pair.Key, Pair.Value);
					}
				}
			}
			for (const FElement& Child : Element.Children)
			{
				if (Child.Is(TEXT("default")))
				{
					ReadDefault(Child, Index);
				}
			}
		}

		/**
		 * Returns the index of Element's default class, which is its class attribute or ChildClass when it has none. Returns
		 * INDEX_NONE for a main class that sets nothing, or nothing after reporting an unknown class.
		 */
		TOptional<int32> ResolveClass(const FElement& Element, const FString& ChildClass)
		{
			const FString* ClassAttribute = Element.Attributes.Find(TEXT("class"));
			const FString ClassName = ClassAttribute ? *ClassAttribute : ChildClass;
			const int32 ClassIndex = FindClass(ClassName);
			if (ClassIndex == INDEX_NONE && !ClassName.Equals(MainClass, ESearchCase::CaseSensitive))
			{
				Error(Element, FString::Printf(TEXT("Unknown default class \"%s\""), *ClassName));
				return {};
			}
			return ClassIndex;
		}

		/**
		 * Returns Element's attributes over the ones its default class sets for its tag, or nothing after reporting an unknown
		 * class. The class is Element's class attribute, or ChildClass when it has none.
		 */
		TOptional<FAttributes> ResolveAttributes(const FElement& Element, const FString& ChildClass)
		{
			const TOptional<int32> ClassIndex = ResolveClass(Element, ChildClass);
			if (!ClassIndex)
			{
				return {};
			}

			FAttributes Attributes;
			if (*ClassIndex != INDEX_NONE)
			{
				if (const FAttributes* Defaults = Classes[*ClassIndex].AttributesByTag.Find(Element.Tag))
				{
					Attributes = *Defaults;
				}
			}
			// An orientation on the element replaces the class's, even when it's written in another form.
			if (HasOrientation(Element.Attributes))
			{
				RemoveOrientation(Attributes);
			}
			for (const TPair<FString, FString>& Pair : Element.Attributes)
			{
				Attributes.Add(Pair.Key, Pair.Value);
			}
			return Attributes;
		}

		/** Returns the childclass Element sets for its contents, or ChildClass, or nothing after reporting an unknown class. */
		TOptional<FString> ResolveChildClass(const FElement& Element, const FString& ChildClass)
		{
			const FString* Value = Element.Attributes.Find(TEXT("childclass"));
			if (!Value)
			{
				return ChildClass;
			}
			if (FindClass(*Value) == INDEX_NONE && !Value->Equals(MainClass, ESearchCase::CaseSensitive))
			{
				Error(Element, FString::Printf(TEXT("Unknown default class \"%s\""), **Value));
				return {};
			}
			return *Value;
		}

		void ReadAssets(const FElement& Asset)
		{
			for (const FElement& Child : Asset.Children)
			{
				if (Child.Is(TEXT("mesh")))
				{
					ReadMesh(Child);
				}
				else if (Child.Is(TEXT("texture")))
				{
					ReadTexture(Child);
				}
				else if (Child.Is(TEXT("material")))
				{
					ReadMaterial(Child);
				}
				else
				{
					ReportOnce(TEXT("asset.") + Child.Tag, EMjcfSeverity::Info, Child,
						FString::Printf(TEXT("<%s> assets are not imported (reported once)"), *Child.Tag));
				}
			}
		}

		void ReadMesh(const FElement& Element)
		{
			const TOptional<FAttributes> Attributes = ResolveAttributes(Element, MainClass);
			if (!Attributes)
			{
				return;
			}

			const FString* FileAttribute = Attributes->Find(TEXT("file"));
			if (!FileAttribute || FileAttribute->IsEmpty())
			{
				Warning(Element, TEXT("Meshes without a file, such as those made from vertex data or a builtin shape, are not supported; the mesh was skipped"));
				if (const FString* NameAttribute = Attributes->Find(TEXT("name")))
				{
					SkippedMeshes.Add(*NameAttribute);
				}
				return;
			}
			const FString File = *FileAttribute;

			const FString* NameAttribute = Attributes->Find(TEXT("name"));
			const FString Name = NameAttribute ? *NameAttribute : FPaths::GetBaseFilename(File);

			FVector Scale = FVector::OneVector;
			if (!ReadVector(Element, *Attributes, TEXT("scale"), Scale))
			{
				return;
			}

			ReportIgnoredMeshAttributes(Element, *Attributes);

			if (Model.FindMesh(Name))
			{
				Error(Element, FString::Printf(TEXT("A mesh named \"%s\" is already defined"), *Name));
				return;
			}

			FMjcfMesh& Mesh = Model.Meshes.AddDefaulted_GetRef();
			Mesh.Name = Name;
			Mesh.File = ResolveAssetPath(File, MeshDir);
			Mesh.Scale = Scale;
			Mesh.ElementPath = Element.Path;
		}

		void ReportIgnoredMeshAttributes(const FElement& Element, const FAttributes& Attributes)
		{
			// The second group only affects MuJoCo's physics and rendering, not the geometry in the file.
			static const TCHAR* const Handled[] = {
				TEXT("name"), TEXT("class"), TEXT("file"), TEXT("scale"),
				TEXT("content_type"), TEXT("smoothnormal"), TEXT("maxhullvert"), TEXT("inertia"), TEXT("material"),
			};
			for (const TPair<FString, FString>& Pair : Attributes)
			{
				bool bHandled = false;
				for (const TCHAR* Name : Handled)
				{
					bHandled |= Pair.Key.Equals(Name, ESearchCase::CaseSensitive);
				}
				if (!bHandled)
				{
					ReportOnce(TEXT("mesh.") + Pair.Key, EMjcfSeverity::Warning, Element,
						FString::Printf(TEXT("Attribute '%s' on <mesh> is not supported and was ignored (reported once)"), *Pair.Key));
				}
			}
		}

		void ReadTexture(const FElement& Element)
		{
			const TOptional<FAttributes> Attributes = ResolveAttributes(Element, MainClass);
			if (!Attributes)
			{
				return;
			}

			FMjcfTexture Texture;
			Texture.ElementPath = Element.Path;
			static const TEnumName<EMjcfTextureType> Types[] = {
				{ TEXT("2d"), EMjcfTextureType::TwoD }, { TEXT("cube"), EMjcfTextureType::Cube }, { TEXT("skybox"), EMjcfTextureType::Skybox },
			};
			static const TEnumName<EMjcfTextureBuiltin> Builtins[] = {
				{ TEXT("none"), EMjcfTextureBuiltin::None }, { TEXT("gradient"), EMjcfTextureBuiltin::Gradient },
				{ TEXT("checker"), EMjcfTextureBuiltin::Checker }, { TEXT("flat"), EMjcfTextureBuiltin::Flat },
			};
			static const TEnumName<EMjcfTextureMark> Marks[] = {
				{ TEXT("none"), EMjcfTextureMark::None }, { TEXT("edge"), EMjcfTextureMark::Edge },
				{ TEXT("cross"), EMjcfTextureMark::Cross }, { TEXT("random"), EMjcfTextureMark::Random },
			};
			double Width = 0;
			double Height = 0;
			if (!ReadEnum(Element, *Attributes, TEXT("type"), Types, Texture.Type)
				|| !ReadEnum(Element, *Attributes, TEXT("builtin"), Builtins, Texture.Builtin)
				|| !ReadEnum(Element, *Attributes, TEXT("mark"), Marks, Texture.Mark)
				|| !ReadVector(Element, *Attributes, TEXT("rgb1"), Texture.Rgb1)
				|| !ReadVector(Element, *Attributes, TEXT("rgb2"), Texture.Rgb2)
				|| !ReadVector(Element, *Attributes, TEXT("markrgb"), Texture.MarkRgb)
				|| !ReadDouble(Element, *Attributes, TEXT("width"), Width)
				|| !ReadDouble(Element, *Attributes, TEXT("height"), Height))
			{
				return;
			}
			Texture.Width = FMath::RoundToInt32(Width);
			Texture.Height = FMath::RoundToInt32(Height);

			const FString File = Attributes->FindRef(TEXT("file"));
			if (!File.IsEmpty())
			{
				Texture.File = ResolveAssetPath(File, TextureDir);
			}
			const FString* NameAttribute = Attributes->Find(TEXT("name"));
			Texture.Name = NameAttribute ? *NameAttribute : FPaths::GetBaseFilename(File);

			if (Texture.Builtin == EMjcfTextureBuiltin::None && Texture.File.IsEmpty())
			{
				Warning(Element, TEXT("Textures need a file or a builtin pattern; the texture was skipped"));
				SkippedTextures.Add(Texture.Name);
				return;
			}
			if (!Texture.Name.IsEmpty() && Model.FindTexture(Texture.Name))
			{
				Error(Element, FString::Printf(TEXT("A texture named \"%s\" is already defined"), *Texture.Name));
				return;
			}
			Model.Textures.Add(MoveTemp(Texture));
		}

		void ReadMaterial(const FElement& Element)
		{
			const TOptional<FAttributes> Attributes = ResolveAttributes(Element, MainClass);
			if (!Attributes)
			{
				return;
			}

			FMjcfMaterial Material;
			Material.ElementPath = Element.Path;
			Material.Name = Attributes->FindRef(TEXT("name"));
			Material.Texture = Attributes->FindRef(TEXT("texture"));
			if (Material.Name.IsEmpty())
			{
				Error(Element, TEXT("Materials need a name"));
				return;
			}
			if (!ReadVector4(Element, *Attributes, TEXT("rgba"), Material.Rgba)
				|| !ReadVector2D(Element, *Attributes, TEXT("texrepeat"), Material.TexRepeat)
				|| !ReadBool(Element, *Attributes, TEXT("texuniform"), Material.bTexUniform)
				|| !ReadDouble(Element, *Attributes, TEXT("specular"), Material.Specular)
				|| !ReadDouble(Element, *Attributes, TEXT("shininess"), Material.Shininess)
				|| !ReadDouble(Element, *Attributes, TEXT("reflectance"), Material.Reflectance)
				|| !ReadDouble(Element, *Attributes, TEXT("emission"), Material.Emission))
			{
				return;
			}
			if (Model.FindMaterial(Material.Name))
			{
				Error(Element, FString::Printf(TEXT("A material named \"%s\" is already defined"), *Material.Name));
				return;
			}
			MaterialSourceFiles.Add(Material.Name, Element.SourceFile);
			Model.Materials.Add(MoveTemp(Material));
		}

		/** Materials can name textures declared after them, so their references are checked once every asset is read. */
		void CheckMaterialTextures()
		{
			for (FMjcfMaterial& Material : Model.Materials)
			{
				if (Material.Texture.IsEmpty() || Model.FindTexture(Material.Texture))
				{
					continue;
				}
				const FString Message = FString::Printf(TEXT("Material \"%s\" uses texture \"%s\""), *Material.Name, *Material.Texture);
				const FString& File = MaterialSourceFiles.FindRef(Material.Name);
				if (SkippedTextures.Contains(Material.Texture))
				{
					Report(EMjcfSeverity::Warning, File, Material.ElementPath, Message + TEXT(", which was skipped; the material is used without it"));
					Material.Texture.Empty();
				}
				else
				{
					Report(EMjcfSeverity::Error, File, Material.ElementPath, Message + TEXT(", which is not defined"));
				}
			}
		}

		/** Reads the bodies, geoms, and frames inside Element into Body, placing them with Frame. */
		void ReadBodyContents(const FElement& Element, FMjcfBody& Body, const FString& ChildClass, const FFramePlacement& Frame)
		{
			for (const FElement& Child : Element.Children)
			{
				if (Child.Is(TEXT("body")))
				{
					FMjcfBody ChildBody;
					if (ReadBody(Child, ChildClass, Frame, ChildBody))
					{
						Body.Bodies.Add(MoveTemp(ChildBody));
					}
				}
				else if (Child.Is(TEXT("geom")))
				{
					FMjcfGeom Geom;
					if (ReadGeom(Child, ChildClass, Frame, Geom))
					{
						Body.Geoms.Add(MoveTemp(Geom));
					}
				}
				else if (Child.Is(TEXT("joint")) || Child.Is(TEXT("freejoint")))
				{
					FMjcfJoint Joint;
					if (ReadJoint(Child, ChildClass, Frame, Joint))
					{
						Body.Joints.Add(MoveTemp(Joint));
					}
				}
				else if (Child.Is(TEXT("inertial")))
				{
					FMjcfInertial Inertial;
					if (Body.Inertial)
					{
						Error(Child, TEXT("A body can have only one <inertial>"));
					}
					else if (ReadInertial(Child, Frame, Inertial))
					{
						Body.Inertial = Inertial;
					}
				}
				else if (Child.Is(TEXT("frame")))
				{
					const TOptional<FString> FrameChildClass = ResolveChildClass(Child, ChildClass);
					FFramePlacement Placement;
					if (FrameChildClass && ReadPlacement(Child, Child.Attributes, Placement))
					{
						ReadBodyContents(Child, Body, *FrameChildClass, Frame.Compose(Placement));
					}
				}
				else if (Child.Is(TEXT("attach")) || Child.Is(TEXT("replicate")) || Child.Is(TEXT("composite")) || Child.Is(TEXT("flexcomp")))
				{
					Warning(Child, FString::Printf(TEXT("<%s> is not supported; the bodies and geoms it would create are missing"), *Child.Tag));
				}
				else
				{
					ReportOnce(TEXT("body.") + Child.Tag, EMjcfSeverity::Info, Child,
						FString::Printf(TEXT("<%s> elements are not imported (reported once)"), *Child.Tag));
				}
			}
		}

		bool ReadBody(const FElement& Element, const FString& ChildClass, const FFramePlacement& Frame, FMjcfBody& Out)
		{
			// Bodies have no default class of their own; childclass only applies to what they contain.
			const TOptional<FString> BodyChildClass = ResolveChildClass(Element, ChildClass);
			FFramePlacement Placement;
			if (!BodyChildClass || !ReadPlacement(Element, Element.Attributes, Placement))
			{
				return false;
			}
			const FFramePlacement Placed = Frame.Compose(Placement);
			Out.Name = Element.Attributes.FindRef(TEXT("name"));
			Out.Position = Placed.Position;
			Out.Rotation = Placed.Rotation;
			Out.ElementPath = Element.Path;
			if (!ReadBool(Element, Element.Attributes, TEXT("mocap"), Out.bMocap) || !ReadDouble(Element, Element.Attributes, TEXT("gravcomp"), Out.GravComp))
			{
				return false;
			}
			if (!Out.Name.IsEmpty())
			{
				BodyNames.Add(Out.Name);
			}
			++BodyDepth;
			ReadBodyContents(Element, Out, *BodyChildClass, FFramePlacement());
			--BodyDepth;
			if (Out.bMocap && (BodyDepth > 0 || !Out.Joints.IsEmpty()))
			{
				Error(Element, TEXT("Mocap bodies must be directly in <worldbody> and have no joints"));
				return false;
			}
			return true;
		}

		bool ReadJoint(const FElement& Element, const FString& ChildClass, const FFramePlacement& Frame, FMjcfJoint& Out)
		{
			// <freejoint> takes no defaults.
			const bool bFreeJoint = Element.Is(TEXT("freejoint"));
			const TOptional<FAttributes> Attributes = bFreeJoint ? TOptional<FAttributes>(Element.Attributes) : ResolveAttributes(Element, ChildClass);
			if (!Attributes)
			{
				return false;
			}

			static const TEnumName<EMjcfJointType> Types[] = {
				{ TEXT("free"), EMjcfJointType::Free }, { TEXT("ball"), EMjcfJointType::Ball },
				{ TEXT("slide"), EMjcfJointType::Slide }, { TEXT("hinge"), EMjcfJointType::Hinge },
			};
			Out.Name = Attributes->FindRef(TEXT("name"));
			Out.ElementPath = Element.Path;
			Out.Type = bFreeJoint ? EMjcfJointType::Free : Out.Type;
			FVector Axis = Out.Axis;
			EAutoFlag Limited = EAutoFlag::Auto;
			if ((!bFreeJoint && !ReadEnum(Element, *Attributes, TEXT("type"), Types, Out.Type))
				|| !ReadVector(Element, *Attributes, TEXT("pos"), Out.Position)
				|| !ReadVector(Element, *Attributes, TEXT("axis"), Axis)
				|| !ReadAutoFlag(Element, *Attributes, TEXT("limited"), Limited)
				|| !ReadVector2D(Element, *Attributes, TEXT("range"), Out.Range)
				|| !ReadDouble(Element, *Attributes, TEXT("ref"), Out.Ref)
				|| !ReadDouble(Element, *Attributes, TEXT("springref"), Out.SpringRef)
				|| !ReadDouble(Element, *Attributes, TEXT("stiffness"), Out.Stiffness)
				|| !ReadDouble(Element, *Attributes, TEXT("damping"), Out.Damping)
				|| !ReadDouble(Element, *Attributes, TEXT("armature"), Out.Armature)
				|| !ReadDouble(Element, *Attributes, TEXT("frictionloss"), Out.FrictionLoss))
			{
				return false;
			}

			if (BodyDepth == 0)
			{
				Error(Element, TEXT("The world body can't have joints"));
				return false;
			}
			if (Out.Type == EMjcfJointType::Free)
			{
				if (BodyDepth > 1)
				{
					Error(Element, TEXT("Free joints are only allowed on bodies directly in <worldbody>"));
					return false;
				}
				Out.Position = FVector::ZeroVector;
				Out.Axis = FVector::UpVector;
				Out.Range = FVector2D::ZeroVector;
				Out.bLimited = false;
			}
			else
			{
				if (!CheckNonZero(Element, TEXT("axis"), Axis)
					|| !ResolveLimited(Element, TEXT("limited"), TEXT("range"), Limited, Attributes->Contains(TEXT("range")), Out.bLimited))
				{
					return false;
				}
				Out.Position = Frame.Position + Frame.Rotation.RotateVector(Out.Position);
				Out.Axis = Frame.Rotation.RotateVector(Axis.GetSafeNormal());
				if (Out.Type != EMjcfJointType::Slide)
				{
					Out.Range = FVector2D(ToRadians(Out.Range.X), ToRadians(Out.Range.Y));
				}
				if (Out.Type == EMjcfJointType::Hinge)
				{
					Out.Ref = ToRadians(Out.Ref);
					Out.SpringRef = ToRadians(Out.SpringRef);
				}
				if (Out.bLimited && Out.Range.X > Out.Range.Y)
				{
					Error(Element, TEXT("range must be the lower limit followed by the upper limit"));
					return false;
				}
			}
			if (Attributes->Contains(TEXT("springdamper")))
			{
				ReportOnce(TEXT("joint.springdamper"), EMjcfSeverity::Warning, Element,
					TEXT("springdamper is not supported, so joints that use it have no spring or damping from it (reported once)"));
			}
			if (!AddUniqueName(Element, TEXT("joint"), Out.Name, JointNames))
			{
				return false;
			}
			if (Out.bLimited && !Out.Name.IsEmpty() && (Out.Type == EMjcfJointType::Hinge || Out.Type == EMjcfJointType::Slide))
			{
				LimitedJointRanges.Add(Out.Name, Out.Range);
			}
			return true;
		}

		bool ReadInertial(const FElement& Element, const FFramePlacement& Frame, FMjcfInertial& Out)
		{
			// <inertial> takes no defaults.
			const FAttributes& Attributes = Element.Attributes;
			FFramePlacement Placement;
			TArray<double> FullInertia;
			if (!ReadPlacement(Element, Attributes, Placement)
				|| !ReadDouble(Element, Attributes, TEXT("mass"), Out.Mass)
				|| !ReadVector(Element, Attributes, TEXT("diaginertia"), Out.DiagInertia)
				|| !ReadNumbers(Element, Attributes, TEXT("fullinertia"), 6, 6, FullInertia))
			{
				return false;
			}
			if (!Attributes.Contains(TEXT("mass")))
			{
				Error(Element, TEXT("<inertial> needs a mass"));
				return false;
			}
			if (Attributes.Contains(TEXT("diaginertia")) && !FullInertia.IsEmpty())
			{
				Error(Element, TEXT("Only one of diaginertia and fullinertia can be given"));
				return false;
			}
			if (!FullInertia.IsEmpty())
			{
				if (HasOrientation(Attributes))
				{
					Error(Element, TEXT("fullinertia sets the orientation itself, so it can't be combined with one"));
					return false;
				}
				// fullinertia is ixx iyy izz ixy ixz iyz.
				const double Matrix[3][3] = {
					{ FullInertia[0], FullInertia[3], FullInertia[4] },
					{ FullInertia[3], FullInertia[1], FullInertia[5] },
					{ FullInertia[4], FullInertia[5], FullInertia[2] },
				};
				FMjcfMath::DiagonalizeInertia(Matrix, Out.DiagInertia, Placement.Rotation);
			}
			const FFramePlacement Placed = Frame.Compose(Placement);
			Out.Position = Placed.Position;
			Out.Rotation = Placed.Rotation;
			return true;
		}

		void ReadActuators(const FElement& Section)
		{
			for (const FElement& Child : Section.Children)
			{
				if (IsActuatorTag(Child.Tag))
				{
					FMjcfActuator Actuator;
					if (ReadActuator(Child, Actuator))
					{
						Model.Actuators.Add(MoveTemp(Actuator));
					}
				}
				else
				{
					Warning(Child, FString::Printf(TEXT("<%s> actuators are not supported; the actuator was skipped"), *Child.Tag));
				}
			}
		}

		bool ReadActuator(const FElement& Element, FMjcfActuator& Out)
		{
			const TOptional<int32> ClassIndex = ResolveClass(Element, MainClass);
			if (!ClassIndex)
			{
				return false;
			}
			FActuatorState State = *ClassIndex != INDEX_NONE ? Classes[*ClassIndex].Actuator : FActuatorState();
			if (!ApplyActuatorAttributes(Element, *Element.Tag, Element.Attributes, State))
			{
				return false;
			}
			Out = State.Actuator;
			Out.Name = Element.Attributes.FindRef(TEXT("name"));
			Out.Kind = Element.Tag;
			Out.ElementPath = Element.Path;

			// MuJoCo reads a positive velocity term on a position-style actuator as a damping ratio, however it was written.
			if (Out.GainType == EMjcfActuatorGain::Fixed && Out.BiasType == EMjcfActuatorBias::Affine && Out.BiasPrm.Y == -Out.GainPrm.X && Out.BiasPrm.Z > 0)
			{
				Out.DampRatio = Out.BiasPrm.Z;
				Out.BiasPrm.Z = 0;
			}

			if (Out.Dynamics == EMjcfActuatorDynamics::Muscle || Out.Dynamics == EMjcfActuatorDynamics::User
				|| Out.GainType == EMjcfActuatorGain::Muscle || Out.GainType == EMjcfActuatorGain::User
				|| Out.BiasType == EMjcfActuatorBias::Muscle || Out.BiasType == EMjcfActuatorBias::User)
			{
				Warning(Element, TEXT("Muscle and user-defined actuator models are not supported; the actuator was skipped"));
				return false;
			}
			for (const TCHAR* Target : { TEXT("tendon"), TEXT("site"), TEXT("body"), TEXT("cranksite") })
			{
				if (Element.Attributes.Contains(Target))
				{
					Warning(Element, FString::Printf(TEXT("Actuators that drive a %s are not supported; the actuator was skipped"), Target));
					return false;
				}
			}
			Out.Joint = Element.Attributes.FindRef(TEXT("joint"));
			if (Out.Joint.IsEmpty())
			{
				Out.Joint = Element.Attributes.FindRef(TEXT("jointinparent"));
			}
			if (Out.Joint.IsEmpty())
			{
				Error(Element, TEXT("Actuators need a joint, tendon, site, or body to drive"));
				return false;
			}
			if (!JointNames.Contains(Out.Joint))
			{
				Error(Element, FString::Printf(TEXT("Unknown joint \"%s\""), *Out.Joint));
				return false;
			}

			if (State.InheritRange > 0)
			{
				const FVector2D* Range = LimitedJointRanges.Find(Out.Joint);
				if (!Range)
				{
					Error(Element, FString::Printf(TEXT("inheritrange needs joint \"%s\" to have a range"), *Out.Joint));
					return false;
				}
				const double Center = (Range->X + Range->Y) / 2;
				const double HalfWidth = (Range->Y - Range->X) / 2 * State.InheritRange;
				Out.CtrlRange = FVector2D(Center - HalfWidth, Center + HalfWidth);
				State.CtrlLimited = EAutoFlag::True;
			}
			if (!ResolveLimited(Element, TEXT("ctrllimited"), TEXT("ctrlrange"), State.CtrlLimited, State.bCtrlRangeGiven, Out.bCtrlLimited)
				|| !ResolveLimited(Element, TEXT("forcelimited"), TEXT("forcerange"), State.ForceLimited, State.bForceRangeGiven, Out.bForceLimited))
			{
				return false;
			}
			return AddUniqueName(Element, TEXT("actuator"), Out.Name, ActuatorNames);
		}

		/**
		 * Applies the attributes of one actuator element, or of its defaults, to State as MuJoCo does: <motor>, <position>, and
		 * <velocity> set the parameters they control, and kp and kv default to the gain already there.
		 */
		bool ApplyActuatorAttributes(const FElement& Element, const TCHAR* Tag, const FAttributes& Attributes, FActuatorState& State)
		{
			FMjcfActuator& Out = State.Actuator;
			TArray<double> Gear;
			if (!ReadNumbers(Element, Attributes, TEXT("gear"), 1, 6, Gear)
				|| !ReadAutoFlag(Element, Attributes, TEXT("ctrllimited"), State.CtrlLimited)
				|| !ReadVector2D(Element, Attributes, TEXT("ctrlrange"), Out.CtrlRange)
				|| !ReadAutoFlag(Element, Attributes, TEXT("forcelimited"), State.ForceLimited)
				|| !ReadVector2D(Element, Attributes, TEXT("forcerange"), Out.ForceRange))
			{
				return false;
			}
			Out.Gear = Gear.IsEmpty() ? Out.Gear : Gear[0];
			State.bCtrlRangeGiven |= Attributes.Contains(TEXT("ctrlrange"));
			State.bForceRangeGiven |= Attributes.Contains(TEXT("forcerange"));

			if (FCString::Strcmp(Tag, TEXT("motor")) == 0)
			{
				Out.Dynamics = EMjcfActuatorDynamics::None;
				Out.GainType = EMjcfActuatorGain::Fixed;
				Out.GainPrm = FVector(1, 0, 0);
				Out.BiasType = EMjcfActuatorBias::None;
				Out.BiasPrm = FVector::ZeroVector;
			}
			else if (FCString::Strcmp(Tag, TEXT("position")) == 0)
			{
				double Kp = Out.GainPrm.X;
				double Kv = 0;
				double DampRatio = 0;
				double TimeConstant = 0;
				if (!ReadDouble(Element, Attributes, TEXT("kp"), Kp) || !ReadDouble(Element, Attributes, TEXT("kv"), Kv)
					|| !ReadDouble(Element, Attributes, TEXT("dampratio"), DampRatio) || !ReadDouble(Element, Attributes, TEXT("timeconst"), TimeConstant)
					|| !ReadDouble(Element, Attributes, TEXT("inheritrange"), State.InheritRange))
				{
					return false;
				}
				Out.GainType = EMjcfActuatorGain::Fixed;
				Out.GainPrm.X = Kp;
				Out.BiasType = EMjcfActuatorBias::Affine;
				Out.BiasPrm.Y = -Kp;
				// Like MuJoCo, a positive velocity term stands for a damping ratio until it's resolved.
				if (Attributes.Contains(TEXT("kv")))
				{
					Out.BiasPrm.Z = -Kv;
				}
				if (Attributes.Contains(TEXT("dampratio")))
				{
					Out.BiasPrm.Z = DampRatio;
				}
				if (Attributes.Contains(TEXT("timeconst")))
				{
					Out.Dynamics = TimeConstant > 0 ? EMjcfActuatorDynamics::FilterExact : EMjcfActuatorDynamics::None;
					Out.TimeConstant = TimeConstant;
				}
			}
			else if (FCString::Strcmp(Tag, TEXT("velocity")) == 0)
			{
				double Kv = Out.GainPrm.X;
				if (!ReadDouble(Element, Attributes, TEXT("kv"), Kv))
				{
					return false;
				}
				Out.GainType = EMjcfActuatorGain::Fixed;
				Out.GainPrm.X = Kv;
				Out.BiasType = EMjcfActuatorBias::Affine;
				Out.BiasPrm = FVector(Out.BiasPrm.X, 0, -Kv);
			}
			else if (FCString::Strcmp(Tag, TEXT("general")) == 0)
			{
				static const TEnumName<EMjcfActuatorDynamics> DynamicsTypes[] = {
					{ TEXT("none"), EMjcfActuatorDynamics::None }, { TEXT("integrator"), EMjcfActuatorDynamics::Integrator },
					{ TEXT("filter"), EMjcfActuatorDynamics::Filter }, { TEXT("filterexact"), EMjcfActuatorDynamics::FilterExact },
					{ TEXT("muscle"), EMjcfActuatorDynamics::Muscle }, { TEXT("user"), EMjcfActuatorDynamics::User },
				};
				static const TEnumName<EMjcfActuatorGain> GainTypes[] = {
					{ TEXT("fixed"), EMjcfActuatorGain::Fixed }, { TEXT("affine"), EMjcfActuatorGain::Affine },
					{ TEXT("muscle"), EMjcfActuatorGain::Muscle }, { TEXT("user"), EMjcfActuatorGain::User },
				};
				static const TEnumName<EMjcfActuatorBias> BiasTypes[] = {
					{ TEXT("none"), EMjcfActuatorBias::None }, { TEXT("affine"), EMjcfActuatorBias::Affine },
					{ TEXT("muscle"), EMjcfActuatorBias::Muscle }, { TEXT("user"), EMjcfActuatorBias::User },
				};
				TArray<double> DynPrm;
				TArray<double> GainPrm;
				TArray<double> BiasPrm;
				if (!ReadEnum(Element, Attributes, TEXT("dyntype"), DynamicsTypes, Out.Dynamics)
					|| !ReadEnum(Element, Attributes, TEXT("gaintype"), GainTypes, Out.GainType)
					|| !ReadEnum(Element, Attributes, TEXT("biastype"), BiasTypes, Out.BiasType)
					|| !ReadNumbers(Element, Attributes, TEXT("dynprm"), 1, 10, DynPrm)
					|| !ReadNumbers(Element, Attributes, TEXT("gainprm"), 1, 10, GainPrm)
					|| !ReadNumbers(Element, Attributes, TEXT("biasprm"), 1, 10, BiasPrm))
				{
					return false;
				}
				// MuJoCo fills the parameters that aren't written with zeros.
				Out.TimeConstant = DynPrm.IsEmpty() ? Out.TimeConstant : DynPrm[0];
				if (!GainPrm.IsEmpty())
				{
					Out.GainPrm = FVector(GainPrm[0], GainPrm.IsValidIndex(1) ? GainPrm[1] : 0, GainPrm.IsValidIndex(2) ? GainPrm[2] : 0);
				}
				if (!BiasPrm.IsEmpty())
				{
					Out.BiasPrm = FVector(BiasPrm[0], BiasPrm.IsValidIndex(1) ? BiasPrm[1] : 0, BiasPrm.IsValidIndex(2) ? BiasPrm[2] : 0);
				}
			}
			return true;
		}

		void ReadContact(const FElement& Section)
		{
			for (const FElement& Child : Section.Children)
			{
				if (Child.Is(TEXT("exclude")))
				{
					FMjcfContactExclude Exclude;
					Exclude.Body1 = Child.Attributes.FindRef(TEXT("body1"));
					Exclude.Body2 = Child.Attributes.FindRef(TEXT("body2"));
					if (Exclude.Body1.IsEmpty() || Exclude.Body2.IsEmpty())
					{
						Error(Child, TEXT("<exclude> needs body1 and body2"));
						continue;
					}
					bool bValid = true;
					for (const FString* Body : { &Exclude.Body1, &Exclude.Body2 })
					{
						if (!BodyNames.Contains(*Body))
						{
							Error(Child, FString::Printf(TEXT("Unknown body \"%s\""), **Body));
							bValid = false;
						}
					}
					if (bValid)
					{
						Model.ContactExcludes.Add(MoveTemp(Exclude));
					}
				}
				else if (Child.Is(TEXT("pair")))
				{
					ReportOnce(TEXT("contact.pair"), EMjcfSeverity::Warning, Child,
						TEXT("<pair> contacts are not imported, so those geoms collide with their usual settings (reported once)"));
				}
				else
				{
					ReportOnce(TEXT("contact.") + Child.Tag, EMjcfSeverity::Info, Child,
						FString::Printf(TEXT("<%s> elements are not imported (reported once)"), *Child.Tag));
				}
			}
		}

		/** Reads limited="true|false|auto" and similar attributes, which are written as forcelimited and ctrllimited on actuators. */
		bool ReadAutoFlag(const FElement& Element, const FAttributes& Attributes, const TCHAR* Name, EAutoFlag& Out)
		{
			static const TEnumName<EAutoFlag> Values[] = { { TEXT("false"), EAutoFlag::False }, { TEXT("true"), EAutoFlag::True }, { TEXT("auto"), EAutoFlag::Auto } };
			return ReadEnum(Element, Attributes, Name, Values, Out);
		}

		/** Decides whether a range applies, as MuJoCo does: "auto" means yes when the range is given and <compiler autolimits> is on. */
		bool ResolveLimited(const FElement& Element, const TCHAR* FlagName, const TCHAR* RangeName, EAutoFlag Flag, bool bRangeGiven, bool& Out)
		{
			if (Flag != EAutoFlag::Auto)
			{
				Out = Flag == EAutoFlag::True;
				return true;
			}
			if (bRangeGiven && !bAutoLimits)
			{
				Error(Element, FString::Printf(TEXT("%s is given but %s is 'auto'; set %s, or <compiler autolimits=\"true\">"), RangeName, FlagName, FlagName));
				return false;
			}
			Out = bRangeGiven;
			return true;
		}

		/** Records Name among Names, failing if another element of Kind already has it. Unnamed elements always succeed. */
		bool AddUniqueName(const FElement& Element, const TCHAR* Kind, const FString& Name, TSet<FString>& Names)
		{
			bool bAlreadyDefined = false;
			if (!Name.IsEmpty())
			{
				Names.Add(Name, &bAlreadyDefined);
			}
			if (bAlreadyDefined)
			{
				Error(Element, FString::Printf(TEXT("A %s named \"%s\" is already defined"), Kind, *Name));
				return false;
			}
			return true;
		}

		bool ReadGeom(const FElement& Element, const FString& ChildClass, const FFramePlacement& Frame, FMjcfGeom& Out)
		{
			const TOptional<FAttributes> Attributes = ResolveAttributes(Element, ChildClass);
			if (!Attributes)
			{
				return false;
			}

			static const TEnumName<EMjcfGeomType> Types[] = {
				{ TEXT("plane"), EMjcfGeomType::Plane }, { TEXT("hfield"), EMjcfGeomType::HField },
				{ TEXT("sphere"), EMjcfGeomType::Sphere }, { TEXT("capsule"), EMjcfGeomType::Capsule },
				{ TEXT("ellipsoid"), EMjcfGeomType::Ellipsoid }, { TEXT("cylinder"), EMjcfGeomType::Cylinder },
				{ TEXT("box"), EMjcfGeomType::Box }, { TEXT("mesh"), EMjcfGeomType::Mesh }, { TEXT("sdf"), EMjcfGeomType::Sdf },
			};
			Out.Name = Attributes->FindRef(TEXT("name"));
			Out.Mesh = Attributes->FindRef(TEXT("mesh"));
			Out.Material = Attributes->FindRef(TEXT("material"));
			Out.ElementPath = Element.Path;
			TArray<double> Size;
			TArray<double> Friction;
			double Group = 0;
			double ConType = Out.ConType;
			double ConAffinity = Out.ConAffinity;
			double Mass = 0;
			FFramePlacement Placement;
			if (!ReadEnum(Element, *Attributes, TEXT("type"), Types, Out.Type)
				|| !ReadNumbers(Element, *Attributes, TEXT("size"), 1, 3, Size)
				|| !ReadVector4(Element, *Attributes, TEXT("rgba"), Out.Rgba)
				|| !ReadDouble(Element, *Attributes, TEXT("group"), Group)
				|| !ReadDouble(Element, *Attributes, TEXT("contype"), ConType)
				|| !ReadDouble(Element, *Attributes, TEXT("conaffinity"), ConAffinity)
				|| !ReadNumbers(Element, *Attributes, TEXT("friction"), 1, 3, Friction)
				|| !ReadDouble(Element, *Attributes, TEXT("mass"), Mass)
				|| !ReadDouble(Element, *Attributes, TEXT("density"), Out.Density)
				|| !ReadPlacement(Element, *Attributes, Placement))
			{
				return false;
			}
			for (int32 Index = 0; Index < Size.Num(); ++Index)
			{
				Out.Size[Index] = Size[Index];
			}
			for (int32 Index = 0; Index < Friction.Num(); ++Index)
			{
				Out.Friction[Index] = Friction[Index];
			}
			Out.Group = FMath::RoundToInt32(Group);
			Out.ConType = FMath::RoundToInt32(ConType);
			Out.ConAffinity = FMath::RoundToInt32(ConAffinity);
			if (Attributes->Contains(TEXT("mass")))
			{
				Out.Mass = Mass;
			}

			TArray<double> FromTo;
			if (!ReadNumbers(Element, *Attributes, TEXT("fromto"), 6, 6, FromTo))
			{
				return false;
			}
			if (!FromTo.IsEmpty())
			{
				ApplyFromTo(Element, FVector(FromTo[0], FromTo[1], FromTo[2]), FVector(FromTo[3], FromTo[4], FromTo[5]), Out, Placement);
			}
			const FFramePlacement Placed = Frame.Compose(Placement);
			Out.Position = Placed.Position;
			Out.Rotation = Placed.Rotation;

			if (Out.Type == EMjcfGeomType::Mesh && !CheckMeshReference(Element, Out))
			{
				return false;
			}
			if (!Out.Material.IsEmpty() && !Model.FindMaterial(Out.Material))
			{
				Error(Element, FString::Printf(TEXT("Unknown material \"%s\""), *Out.Material));
				return false;
			}
			return true;
		}

		/** Places a capsule, cylinder, box, or ellipsoid between two points, as MuJoCo's fromto does. */
		void ApplyFromTo(const FElement& Element, const FVector& From, const FVector& To, FMjcfGeom& Geom, FFramePlacement& OutPlacement)
		{
			const double HalfLength = FVector::Distance(From, To) / 2;
			switch (Geom.Type)
			{
				case EMjcfGeomType::Capsule:
				case EMjcfGeomType::Cylinder:
					Geom.Size.Y = HalfLength;
					break;
				case EMjcfGeomType::Box:
				case EMjcfGeomType::Ellipsoid:
					Geom.Size = FVector(Geom.Size.X, Geom.Size.X, HalfLength);
					break;
				default:
					Warning(Element, TEXT("fromto only applies to capsules, cylinders, boxes, and ellipsoids and was ignored"));
					return;
			}
			OutPlacement.Position = (From + To) / 2;
			OutPlacement.Rotation = FMjcfMath::RotationFromZ(From - To);
		}

		bool CheckMeshReference(const FElement& Element, const FMjcfGeom& Geom)
		{
			if (Geom.Mesh.IsEmpty())
			{
				Error(Element, TEXT("Mesh geoms need a mesh attribute"));
				return false;
			}
			if (Model.FindMesh(Geom.Mesh))
			{
				return true;
			}
			if (SkippedMeshes.Contains(Geom.Mesh))
			{
				Warning(Element, FString::Printf(TEXT("The geom uses mesh \"%s\", which was skipped; the geom was skipped too"), *Geom.Mesh));
			}
			else
			{
				Error(Element, FString::Printf(TEXT("Unknown mesh \"%s\""), *Geom.Mesh));
			}
			return false;
		}

		/** Reads pos and the orientation into Out. */
		bool ReadPlacement(const FElement& Element, const FAttributes& Attributes, FFramePlacement& Out)
		{
			Out = FFramePlacement();
			return ReadVector(Element, Attributes, TEXT("pos"), Out.Position) && ReadOrientation(Element, Attributes, Out.Rotation);
		}

		/** Reads whichever of MuJoCo's orientation forms Attributes uses into Out, leaving the identity when there is none. */
		bool ReadOrientation(const FElement& Element, const FAttributes& Attributes, FQuat& Out)
		{
			Out = FQuat::Identity;
			int32 Count = 0;
			for (const TCHAR* Name : OrientationAttributes)
			{
				Count += Attributes.Contains(Name) ? 1 : 0;
			}
			if (Count > 1)
			{
				Error(Element, TEXT("Only one of quat, axisangle, euler, xyaxes, and zaxis can be given"));
				return false;
			}

			TArray<double> Values;
			if (Attributes.Contains(TEXT("quat")))
			{
				if (!ReadNumbers(Element, Attributes, TEXT("quat"), 4, 4, Values))
				{
					return false;
				}
				// MJCF writes quaternions as w x y z.
				const FQuat Rotation(Values[1], Values[2], Values[3], Values[0]);
				if (Rotation.SizeSquared() < UE_DOUBLE_SMALL_NUMBER)
				{
					Error(Element, TEXT("quat must not be zero"));
					return false;
				}
				Out = Rotation.GetNormalized();
			}
			else if (Attributes.Contains(TEXT("axisangle")))
			{
				if (!ReadNumbers(Element, Attributes, TEXT("axisangle"), 4, 4, Values) || !CheckNonZero(Element, TEXT("axisangle"), FVector(Values[0], Values[1], Values[2])))
				{
					return false;
				}
				Out = FQuat(FVector(Values[0], Values[1], Values[2]).GetSafeNormal(), ToRadians(Values[3]));
			}
			else if (Attributes.Contains(TEXT("euler")))
			{
				if (!ReadNumbers(Element, Attributes, TEXT("euler"), 3, 3, Values))
				{
					return false;
				}
				Out = FMjcfMath::EulerToRotation(FVector(ToRadians(Values[0]), ToRadians(Values[1]), ToRadians(Values[2])), EulerSequence);
			}
			else if (Attributes.Contains(TEXT("xyaxes")))
			{
				if (!ReadNumbers(Element, Attributes, TEXT("xyaxes"), 6, 6, Values))
				{
					return false;
				}
				if (!FMjcfMath::RotationFromXYAxes(FVector(Values[0], Values[1], Values[2]), FVector(Values[3], Values[4], Values[5]), Out))
				{
					Error(Element, TEXT("xyaxes needs two non-zero, non-parallel axes"));
					return false;
				}
			}
			else if (Attributes.Contains(TEXT("zaxis")))
			{
				if (!ReadNumbers(Element, Attributes, TEXT("zaxis"), 3, 3, Values) || !CheckNonZero(Element, TEXT("zaxis"), FVector(Values[0], Values[1], Values[2])))
				{
					return false;
				}
				Out = FMjcfMath::RotationFromZ(FVector(Values[0], Values[1], Values[2]));
			}
			return true;
		}

		bool CheckNonZero(const FElement& Element, const TCHAR* Name, const FVector& Axis)
		{
			if (Axis.IsNearlyZero(UE_DOUBLE_SMALL_NUMBER))
			{
				Error(Element, FString::Printf(TEXT("The axis in %s must not be zero"), Name));
				return false;
			}
			return true;
		}

		double ToRadians(double Angle) const
		{
			return bDegrees ? FMath::DegreesToRadians(Angle) : Angle;
		}

		/**
		 * Reads attribute Name into Out, leaving Out empty when it isn't set. Returns false after reporting an error if it isn't
		 * between MinCount and MaxCount numbers.
		 */
		bool ReadNumbers(const FElement& Element, const FAttributes& Attributes, const TCHAR* Name, int32 MinCount, int32 MaxCount, TArray<double>& Out)
		{
			Out.Reset();
			const FString* Text = Attributes.Find(Name);
			if (!Text)
			{
				return true;
			}
			if (!ParseDoubles(*Text, Out) || Out.Num() < MinCount || Out.Num() > MaxCount)
			{
				Error(Element, FString::Printf(TEXT("%s must be %s, got '%s'"), Name, *DescribeCount(MinCount, MaxCount), **Text));
				Out.Reset();
				return false;
			}
			return true;
		}

		bool ReadDouble(const FElement& Element, const FAttributes& Attributes, const TCHAR* Name, double& Out)
		{
			TArray<double> Values;
			if (!ReadNumbers(Element, Attributes, Name, 1, 1, Values))
			{
				return false;
			}
			if (!Values.IsEmpty())
			{
				Out = Values[0];
			}
			return true;
		}

		bool ReadVector2D(const FElement& Element, const FAttributes& Attributes, const TCHAR* Name, FVector2D& Out)
		{
			TArray<double> Values;
			if (!ReadNumbers(Element, Attributes, Name, 2, 2, Values))
			{
				return false;
			}
			if (!Values.IsEmpty())
			{
				Out = FVector2D(Values[0], Values[1]);
			}
			return true;
		}

		bool ReadVector(const FElement& Element, const FAttributes& Attributes, const TCHAR* Name, FVector& Out)
		{
			TArray<double> Values;
			if (!ReadNumbers(Element, Attributes, Name, 3, 3, Values))
			{
				return false;
			}
			if (!Values.IsEmpty())
			{
				Out = FVector(Values[0], Values[1], Values[2]);
			}
			return true;
		}

		bool ReadVector4(const FElement& Element, const FAttributes& Attributes, const TCHAR* Name, FVector4& Out)
		{
			TArray<double> Values;
			if (!ReadNumbers(Element, Attributes, Name, 4, 4, Values))
			{
				return false;
			}
			if (!Values.IsEmpty())
			{
				Out = FVector4(Values[0], Values[1], Values[2], Values[3]);
			}
			return true;
		}

		bool ReadBool(const FElement& Element, const FAttributes& Attributes, const TCHAR* Name, bool& Out)
		{
			static const TEnumName<bool> Values[] = { { TEXT("true"), true }, { TEXT("false"), false } };
			return ReadEnum(Element, Attributes, Name, Values, Out);
		}

		template <typename EnumType, int32 Count>
		bool ReadEnum(const FElement& Element, const FAttributes& Attributes, const TCHAR* Name, const TEnumName<EnumType> (&Names)[Count], EnumType& Out)
		{
			const FString* Text = Attributes.Find(Name);
			if (!Text)
			{
				return true;
			}
			TArray<FString> Allowed;
			for (const TEnumName<EnumType>& Entry : Names)
			{
				if (Text->Equals(Entry.Name, ESearchCase::CaseSensitive))
				{
					Out = Entry.Value;
					return true;
				}
				Allowed.Add(FString::Printf(TEXT("'%s'"), Entry.Name));
			}
			Error(Element, FString::Printf(TEXT("%s must be one of %s, got '%s'"), Name, *FString::Join(Allowed, TEXT(", ")), **Text));
			return false;
		}

		/** Resolves File relative to Directory (meshdir or texturedir), or assetdir when Directory is unset, or the model's folder. */
		FString ResolveAssetPath(const FString& File, const FString& Directory) const
		{
			if (!FPaths::IsRelative(File))
			{
				return NormalizePath(File);
			}
			const FString& Base = !Directory.IsEmpty() ? Directory : AssetDir;
			if (Base.IsEmpty())
			{
				return NormalizePath(FPaths::Combine(ModelDirectory, File));
			}
			const FString BaseDirectory = FPaths::IsRelative(Base) ? FPaths::Combine(ModelDirectory, Base) : Base;
			return NormalizePath(FPaths::Combine(BaseDirectory, File));
		}

		int32 FindClass(const FString& Name) const
		{
			return Classes.IndexOfByPredicate([&Name](const FDefaultClass& Class) { return Class.Name.Equals(Name, ESearchCase::CaseSensitive); });
		}

		void Report(EMjcfSeverity Severity, const FString& File, const FString& Path, const FString& Message)
		{
			FMjcfDiagnostic& Entry = Diagnostics.AddDefaulted_GetRef();
			Entry.Severity = Severity;
			Entry.Message = Message;
			Entry.SourceFile = File;
			Entry.ElementPath = Path;
			bHasErrors |= Severity == EMjcfSeverity::Error;
		}

		void ReportOnce(const FString& Key, EMjcfSeverity Severity, const FElement& Element, const FString& Message)
		{
			bool bAlreadyReported = false;
			ReportedKeys.Add(Key, &bAlreadyReported);
			if (!bAlreadyReported)
			{
				Report(Severity, Element.SourceFile, Element.Path, Message);
			}
		}

		void Error(const FElement& Element, const FString& Message)
		{
			Report(EMjcfSeverity::Error, Element.SourceFile, Element.Path, Message);
		}

		void Warning(const FElement& Element, const FString& Message)
		{
			Report(EMjcfSeverity::Warning, Element.SourceFile, Element.Path, Message);
		}

		FString SourceFile;
		FString ModelDirectory;
		FMjcfFileReader Reader;
		TArray<FMjcfDiagnostic>& Diagnostics;

		FString MeshDir;
		FString TextureDir;
		FString AssetDir;
		bool bDegrees = true;
		bool bAutoLimits = true;
		FString EulerSequence = TEXT("xyz");
		TArray<FDefaultClass> Classes;
		TSet<FString> IncludedFiles;
		TSet<FString> ReportedKeys;
		TSet<FString> SkippedMeshes;
		TSet<FString> SkippedTextures;
		TMap<FString, FString> MaterialSourceFiles;

		/** How many bodies enclose the element being read; 0 inside <worldbody> itself. */
		int32 BodyDepth = 0;

		TSet<FString> BodyNames = { TEXT("world") };
		TSet<FString> JointNames;
		TSet<FString> ActuatorNames;
		TMap<FString, FVector2D> LimitedJointRanges;

		FMjcfModel Model;
		bool bHasErrors = false;
	};
} // namespace

FString FMjcfDiagnostic::ToString() const
{
	FString Result;
	if (!SourceFile.IsEmpty())
	{
		Result += SourceFile + TEXT(": ");
	}
	if (!ElementPath.IsEmpty())
	{
		Result += ElementPath + TEXT(": ");
	}
	return Result + Message;
}

TOptional<FMjcfModel> FMjcfParser::ParseFile(const FString& Path, TArray<FMjcfDiagnostic>& OutDiagnostics)
{
	const FString FullPath = NormalizePath(Path);
	FString Contents;
	if (!FFileHelper::LoadFileToString(Contents, *FullPath))
	{
		FMjcfDiagnostic& Entry = OutDiagnostics.AddDefaulted_GetRef();
		Entry.Severity = EMjcfSeverity::Error;
		Entry.Message = TEXT("Could not read the file");
		Entry.SourceFile = FullPath;
		return {};
	}
	return ParseString(Contents, FullPath, OutDiagnostics);
}

TOptional<FMjcfModel> FMjcfParser::ParseString(const FString& Xml, const FString& SourceFile, TArray<FMjcfDiagnostic>& OutDiagnostics, FMjcfFileReader Reader)
{
	if (!Reader)
	{
		Reader = [](const FString& Path, FString& OutContents) { return FFileHelper::LoadFileToString(OutContents, *Path); };
	}
	return FModelParser(SourceFile, MoveTemp(Reader), OutDiagnostics).Parse(Xml);
}

bool FMjcfParser::IsMjcfFile(const FString& Path)
{
	FString Contents;
	if (!FFileHelper::LoadFileToString(Contents, *Path))
	{
		return false;
	}
	const FXmlFile Xml(Contents, EConstructMethod::ConstructFromBuffer);
	return Xml.IsValid() && Xml.GetRootNode() && Xml.GetRootNode()->GetTag().Equals(TEXT("mujoco"), ESearchCase::CaseSensitive);
}
