// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "bindings/spaces/schola_space.h"

#include <cstdint>

// Inspector resource for a US2 Discrete space
class ScholaDiscreteSpace : public ScholaSpace {
	GDCLASS(ScholaDiscreteSpace, ScholaSpace)

public:
	// Number of available choices; values are in [0, n)
	void set_n(int32_t p_n);
	int32_t get_n() const;

	schola::Space to_space() const override;

protected:
	static void _bind_methods();

private:
	int32_t n = 0;
};
