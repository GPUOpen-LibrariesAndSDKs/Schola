# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

class_name ScholaOnnxPolicyTest
extends GdUnitTestSuite

const FIXTURES := "res://tests/inference/fixtures/"


func _policy(file: String) -> ScholaOnnxPolicy:
	var policy := ScholaOnnxPolicy.new()
	policy.model_path = FIXTURES + file
	return policy


func test_inference_classes_are_registered() -> void:
	assert_bool(ClassDB.class_exists("ScholaOnnxPolicy")).is_true()
	assert_bool(ClassDB.class_exists("ScholaPolicyState")).is_true()


func test_load_reads_the_model() -> void:
	var policy := _policy("box_scale.onnx")
	assert_int(policy.load()).is_equal(OK)
	assert_bool(policy.is_loaded()).is_true()
	assert_bool(policy.is_recurrent()).is_false()
	assert_array(policy.get_input_names()).contains_exactly(["obs"])
	assert_array(policy.get_output_names()).contains_exactly(["action"])


func test_load_detects_recurrent_models() -> void:
	var policy := _policy("counter.onnx")
	assert_int(policy.load()).is_equal(OK)
	assert_bool(policy.is_recurrent()).is_true()


func test_load_reports_a_missing_file() -> void:
	var policy := _policy("does_not_exist.onnx")
	assert_int(policy.load()).is_equal(ERR_FILE_NOT_FOUND)
	assert_bool(policy.is_loaded()).is_false()
	assert_str(policy.get_last_error()).contains("does_not_exist.onnx")


func test_load_reports_an_invalid_model() -> void:
	var policy := _policy("bad_state_int.onnx")
	assert_int(policy.load()).is_equal(ERR_INVALID_DATA)
	assert_str(policy.get_last_error()).contains("must be float")


func test_create_state_needs_an_initialized_policy() -> void:
	var policy := _policy("box_scale.onnx")
	assert_int(policy.load()).is_equal(OK)
	assert_object(policy.create_state()).is_null()
