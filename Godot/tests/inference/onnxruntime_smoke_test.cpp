// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <onnxruntime_c_api.h>
#include <catch_amalgamated.hpp>

#include <string>

TEST_CASE("ONNX Runtime 1.30.0 is linked and creates an environment", "[inference]") {
	const OrtApiBase *api_base = OrtGetApiBase();
	REQUIRE(std::string(api_base->GetVersionString()) == "1.30.0");

	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	REQUIRE(api != nullptr);

	OrtEnv *environment = nullptr;
	OrtStatus *status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "schola_inference_tests", &environment);
	REQUIRE(status == nullptr);
	api->ReleaseEnv(environment);
}
