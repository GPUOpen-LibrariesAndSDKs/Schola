// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "bindings/spaces/schola_space.h"

#include <cstdint>

// Inspector resource for a US2 MultiBinary space
class ScholaMultiBinarySpace : public ScholaSpace {
	GDCLASS(ScholaMultiBinarySpace, ScholaSpace)

public:
	// Number of independent binary values
	void set_n(int32_t p_n);
	int32_t get_n() const;

	schola::Space to_space() const override;

protected:
	static void _bind_methods();

private:
	int32_t n = 0;
};
