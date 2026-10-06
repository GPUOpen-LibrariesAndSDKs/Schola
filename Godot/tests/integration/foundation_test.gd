# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

class_name FoundationIntegrationTest
extends GdUnitTestSuite


func test_runtime_extension_is_registered() -> void:
	assert_bool(ClassDB.class_exists("ScholaRuntimeProbe")).is_true()


func test_training_extension_is_registered() -> void:
	assert_bool(ClassDB.class_exists("ScholaTrainingProbe")).is_true()
