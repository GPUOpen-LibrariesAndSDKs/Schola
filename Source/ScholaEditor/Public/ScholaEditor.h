// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "Modules/ModuleInterface.h"
#include "Modules/ModuleManager.h"

#include "BlueprintEditorLibrary.h"
#include "Kismet2/BlueprintEditorUtils.h"
#include "Kismet2/KismetEditorUtilities.h"
#include "BlueprintEditor.h"
#include "UnrealEd.h"

#include "Common/UtilityFunctions.h"

/**
 * @brief Unreal module implementation for the Schola editor plugin.
 */
class FScholaEditorModule : public IModuleInterface
{
public:
    /** Registers editor extensions and assets used by Schola. */
    virtual void StartupModule() override;
    /** Unregisters editor hooks and releases editor-only resources. */
    virtual void ShutdownModule() override;
};