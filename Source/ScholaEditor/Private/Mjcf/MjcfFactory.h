// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Factories/Factory.h"
#include "MjcfFactory.generated.h"

/**
 * @brief Imports an MJCF (MuJoCo XML) file dropped into the Content Browser.
 *
 * Creates an Actor Blueprint named after the file, where it was dropped, that assembles the model from its bodies and the
 * geoms MuJoCo shows by default. The assets it uses go in a folder named after the file, next to it:
 * - every OBJ mesh the file declares, imported through Interchange, and every STL mesh, each named after its MJCF mesh;
 * - a material instance for each MJCF material and color that visible geoms use, and the textures they need;
 * - a mesh for each plane geom.
 * A mesh or geom that can't be imported is reported and skipped without affecting the others. Importing the same file again
 * updates these assets in place.
 *
 * Mesh vertices are converted to Unreal units and axes, with the MJCF mesh scale applied, so each mesh has the size and
 * orientation it has in MuJoCo. No collision is generated.
 */
UCLASS()
class UMjcfFactory : public UFactory
{
	GENERATED_BODY()

public:
	UMjcfFactory();

	virtual bool FactoryCanImport(const FString& Filename) override;
	virtual UObject* FactoryCreateFile(UClass* InClass, UObject* InParent, FName InName, EObjectFlags Flags, const FString& Filename, const TCHAR* Parms, FFeedbackContext* Warn, bool& bOutOperationCanceled) override;
};
