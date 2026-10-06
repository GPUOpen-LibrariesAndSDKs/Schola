// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "runtime/register_types.h"

#include "bindings/register_types.h"
#include "inference/register_types.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

void initialize_schola_module(godot::ModuleInitializationLevel p_level) {
	if (p_level != godot::MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	register_schola_binding_types();
	register_schola_inference_types();
}

void uninitialize_schola_module(godot::ModuleInitializationLevel p_level) {
	if (p_level != godot::MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	unregister_schola_inference_types();
	unregister_schola_binding_types();
}

extern "C" {
GDExtensionBool GDE_EXPORT schola_library_init(
		GDExtensionInterfaceGetProcAddress p_get_proc_address,
		GDExtensionClassLibraryPtr p_library,
		GDExtensionInitialization *r_initialization) {
	godot::GDExtensionBinding::InitObject init_object(p_get_proc_address, p_library, r_initialization);
	init_object.register_initializer(initialize_schola_module);
	init_object.register_terminator(uninitialize_schola_module);
	init_object.set_minimum_library_initialization_level(godot::MODULE_INITIALIZATION_LEVEL_SCENE);
	return init_object.init();
}
}
