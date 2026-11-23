// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Misc/Optional.h"
#include "MjcfModel.h"
#include "Templates/Function.h"

enum class EMjcfSeverity : uint8
{
	/** Content this importer doesn't use, such as lights. */
	Info,
	/** Something was skipped or ignored, so the result may differ from MuJoCo. */
	Warning,
	/** The file can't be imported. */
	Error,
};

struct SCHOLAMJCF_API FMjcfDiagnostic
{
	EMjcfSeverity Severity = EMjcfSeverity::Info;

	FString Message;

	/** The file the problem is in, which may be an included file. */
	FString SourceFile;

	/** Empty when the problem isn't tied to an element. */
	FString ElementPath;

	FString ToString() const;
};

/** Reads a file into OutContents, returning false if it can't be read. */
using FMjcfFileReader = TFunction<bool(const FString& Path, FString& OutContents)>;

/**
 * @brief Reads an MJCF (MuJoCo XML) file into an FMjcfModel.
 *
 * Expands <include>, applies <compiler> settings and <default> classes, resolves mesh and texture files to absolute
 * paths, and flattens <frame> elements into the placements of what they contain. It doesn't check that files exist.
 *
 * Parsing fails, returning an unset optional, if any error was reported.
 */
class SCHOLAMJCF_API FMjcfParser
{
public:
	static TOptional<FMjcfModel> ParseFile(const FString& Path, TArray<FMjcfDiagnostic>& OutDiagnostics);

	/**
	 * @param Xml The contents of the main model file.
	 * @param SourceFile Absolute path of the main model file. Mesh and texture paths are relative to its folder, and so are
	 * include paths, falling back to the including file's folder.
	 * @param Reader Reads included files. Reads from disk when unset.
	 */
	static TOptional<FMjcfModel> ParseString(const FString& Xml, const FString& SourceFile, TArray<FMjcfDiagnostic>& OutDiagnostics, FMjcfFileReader Reader = FMjcfFileReader());

	/** Returns whether Path is an XML file whose root element is <mujoco>. Includes and the rest of the model aren't checked. */
	static bool IsMjcfFile(const FString& Path);
};
