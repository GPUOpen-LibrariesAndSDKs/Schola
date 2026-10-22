// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/spaces/point.h"
#include "core/spaces/space.h"

#include <catch_amalgamated.hpp>

#include <cmath>
#include <limits>
#include <string>

using namespace schola;

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();

bool message_contains(const Status &p_status, const std::string &p_text) {
	return p_status.get_message().find(p_text) != std::string::npos;
}

Space make_box(std::vector<float> p_low, std::vector<float> p_high, std::vector<int32_t> p_shape) {
	return Space(BoxSpace{ std::move(p_low), std::move(p_high), std::move(p_shape) });
}

} // namespace

TEST_CASE("a well-formed definition of every kind passes check_definition", "[spaces]") {
	CHECK(make_box({ 0, 0 }, { 1, 1 }, { 2 }).check_definition().is_ok());
	CHECK(make_box({ 0, 0, 0, 0, 0, 0 }, { 1, 1, 1, 1, 1, 1 }, { 2, 3 }).check_definition().is_ok());
	CHECK(Space(DiscreteSpace{ 4 }).check_definition().is_ok());
	CHECK(Space(MultiDiscreteSpace{ { 3, 2 } }).check_definition().is_ok());
	CHECK(Space(MultiBinarySpace{ 5 }).check_definition().is_ok());

	DictSpace dict;
	dict.entries.emplace_back("position", make_box({ 0 }, { 1 }, { 1 }));
	dict.entries.emplace_back("mode", Space(DiscreteSpace{ 2 }));
	CHECK(Space(std::move(dict)).check_definition().is_ok());
}

TEST_CASE("an omitted Box bound is unbounded, not zero", "[spaces]") {
	const Space space = make_box({ -INF, 0 }, { INF, INF }, { 2 });
	REQUIRE(space.check_definition().is_ok());

	CHECK(space.validate(Point(BoxPoint{ { -1e30f, 1e30f } })).is_ok());
	CHECK(space.validate(Point(BoxPoint{ { 5.0f, 123.0f } })).is_ok());
	CHECK_FALSE(space.validate(Point(BoxPoint{ { 0.0f, -0.5f } })).is_ok());
	CHECK(space.make_point() == Point(BoxPoint{ { 0.0f, 0.0f } }));
}

TEST_CASE("a malformed definition is rejected with the failing field", "[spaces]") {
	SECTION("Box with no dimensions") {
		const Status status = make_box({}, {}, {}).check_definition();
		CHECK(status.get_code() == StatusCode::INVALID_DATA);
		CHECK(message_contains(status, "at least one dimension"));
	}
	SECTION("Box with a non-positive dimension") {
		const Status status = make_box({ 0 }, { 1 }, { 1, 0 }).check_definition();
		CHECK(status.get_code() == StatusCode::INVALID_DATA);
		CHECK(message_contains(status, "shape[1]"));
	}
	SECTION("Box bounds that do not match the shape") {
		const Status status = make_box({ 0, 0 }, { 1, 1, 1 }, { 2 }).check_definition();
		CHECK(status.get_code() == StatusCode::INVALID_DATA);
		CHECK(message_contains(status, "high has 3 values"));
	}
	SECTION("Box with a NaN bound") {
		const Status status = make_box({ 0, std::nanf("") }, { 1, 1 }, { 2 }).check_definition();
		CHECK(status.get_code() == StatusCode::INVALID_DATA);
		CHECK(message_contains(status, "low[1] is NaN"));
	}
	SECTION("Box with low greater than high") {
		const Status status = make_box({ 0, 2 }, { 1, 1 }, { 2 }).check_definition();
		CHECK(status.get_code() == StatusCode::INVALID_DATA);
		CHECK(message_contains(status, "low[1]"));
	}
	SECTION("Discrete with zero choices") {
		const Status status = Space(DiscreteSpace{ 0 }).check_definition();
		CHECK(status.get_code() == StatusCode::INVALID_DATA);
		CHECK(message_contains(status, "Discrete n is 0"));
	}
	SECTION("MultiDiscrete that is empty or has a non-positive entry") {
		CHECK(Space(MultiDiscreteSpace{}).check_definition().get_code() == StatusCode::INVALID_DATA);
		CHECK(message_contains(Space(MultiDiscreteSpace{ { 2, -1 } }).check_definition(), "nvec[1]"));
	}
	SECTION("MultiBinary with zero values") {
		CHECK(Space(MultiBinarySpace{ 0 }).check_definition().get_code() == StatusCode::INVALID_DATA);
	}
	SECTION("Dict with a duplicate key") {
		DictSpace dict;
		dict.entries.emplace_back("a", Space(DiscreteSpace{ 2 }));
		dict.entries.emplace_back("a", Space(DiscreteSpace{ 3 }));
		const Status status = Space(std::move(dict)).check_definition();
		CHECK(status.get_code() == StatusCode::INVALID_DATA);
		CHECK(message_contains(status, "'a' appears more than once"));
	}
	SECTION("Dict with a malformed nested entry names its key path") {
		DictSpace inner;
		inner.entries.emplace_back("speed", Space(DiscreteSpace{ -2 }));
		DictSpace outer;
		outer.entries.emplace_back("motor", Space(std::move(inner)));
		const Status status = Space(std::move(outer)).check_definition();
		CHECK(message_contains(status, "motor.speed"));
	}
}

TEST_CASE("an invalid point is rejected with an error that names the failing index", "[spaces]") {
	const Space box = make_box({ 0, 0, 0 }, { 5, 5, 5 }, { 3 });

	SECTION("out-of-bounds Box value") {
		const Status status = box.validate(Point(BoxPoint{ { 1, 7.5f, 2 } }));
		CHECK(status.get_code() == StatusCode::INVALID_ARGUMENT);
		CHECK(message_contains(status, "index 1"));
		CHECK(message_contains(status, "7.5"));
	}
	SECTION("NaN Box value") {
		CHECK(message_contains(box.validate(Point(BoxPoint{ { 1, 1, std::nanf("") } })), "index 2 is NaN"));
	}
	SECTION("wrong Box size") {
		CHECK(message_contains(box.validate(Point(BoxPoint{ { 1, 1 } })), "has 2 values but the space expects 3"));
	}
	SECTION("wrong kind") {
		CHECK(message_contains(box.validate(Point(DiscretePoint{ 1 })), "expected a Box point but got a Discrete point"));
	}
	SECTION("Discrete out of range") {
		const Space discrete(DiscreteSpace{ 4 });
		CHECK(discrete.validate(Point(DiscretePoint{ 3 })).is_ok());
		CHECK(message_contains(discrete.validate(Point(DiscretePoint{ 4 })), "outside [0, 4)"));
		CHECK_FALSE(discrete.validate(Point(DiscretePoint{ -1 })).is_ok());
	}
	SECTION("MultiDiscrete value out of range") {
		const Space multi_discrete(MultiDiscreteSpace{ { 3, 2 } });
		CHECK(multi_discrete.validate(Point(MultiDiscretePoint{ { 2, 1 } })).is_ok());
		CHECK(message_contains(multi_discrete.validate(Point(MultiDiscretePoint{ { 2, 2 } })), "index 1"));
	}
	SECTION("MultiBinary value other than 0 or 1") {
		const Space multi_binary(MultiBinarySpace{ 3 });
		CHECK(multi_binary.validate(Point(MultiBinaryPoint{ { 0, 1, 1 } })).is_ok());
		CHECK(message_contains(multi_binary.validate(Point(MultiBinaryPoint{ { 0, 2, 1 } })), "index 1"));
	}
}

TEST_CASE("Dict spaces sort keys and validate points by key", "[spaces]") {
	DictSpace definition;
	definition.entries.emplace_back("velocity", Space(DiscreteSpace{ 3 }));
	definition.entries.emplace_back("alpha", Space(MultiBinarySpace{ 2 }));
	const Space space(std::move(definition));

	const DictSpace *dict = space.get_if<DictSpace>();
	REQUIRE(dict != nullptr);
	CHECK(dict->entries[0].first == "alpha");
	CHECK(dict->entries[1].first == "velocity");
	CHECK(dict->find("velocity") != nullptr);
	CHECK(dict->find("missing") == nullptr);

	const Point valid = space.make_point();
	CHECK(space.validate(valid).is_ok());

	SECTION("missing key") {
		DictPoint point;
		point.entries.emplace_back("alpha", Point(MultiBinaryPoint{ { 0, 0 } }));
		CHECK(message_contains(space.validate(Point(std::move(point))), "missing key 'velocity'"));
	}
	SECTION("extra key") {
		DictPoint point = *valid.get_if<DictPoint>();
		point.entries.emplace_back("zeta", Point(DiscretePoint{ 0 }));
		CHECK(message_contains(space.validate(Point(std::move(point))), "unexpected key 'zeta'"));
	}
	SECTION("unsorted keys") {
		DictPoint point;
		point.entries.emplace_back("velocity", Point(DiscretePoint{ 0 }));
		point.entries.emplace_back("alpha", Point(MultiBinaryPoint{ { 0, 0 } }));
		CHECK(message_contains(space.validate(Point(std::move(point))), "not sorted"));
	}
	SECTION("invalid nested value names its key") {
		DictPoint point = *valid.get_if<DictPoint>();
		*point.find("velocity") = Point(DiscretePoint{ 9 });
		CHECK(message_contains(space.validate(Point(std::move(point))), "at 'velocity'"));
	}
}

TEST_CASE("make_point returns a valid point of the space's kind and size", "[spaces]") {
	const Space box = make_box({ 1, -3 }, { 2, -1 }, { 2 });
	CHECK(box.make_point() == Point(BoxPoint{ { 1, -1 } }));
	CHECK(box.validate(box.make_point()).is_ok());
	CHECK(Space(DiscreteSpace{ 4 }).make_point() == Point(DiscretePoint{ 0 }));
	CHECK(Space(MultiDiscreteSpace{ { 2, 2, 2 } }).make_point() == Point(MultiDiscretePoint{ { 0, 0, 0 } }));
	CHECK(Space(MultiBinarySpace{ 2 }).make_point() == Point(MultiBinaryPoint{ { 0, 0 } }));
}

TEST_CASE("flattened sizes match Unreal Schola", "[spaces]") {
	CHECK(make_box({ 0, 0, 0, 0, 0, 0 }, { 1, 1, 1, 1, 1, 1 }, { 2, 3 }).get_flattened_size() == 6);
	CHECK(Space(DiscreteSpace{ 4 }).get_flattened_size() == 4);
	CHECK(Space(MultiDiscreteSpace{ { 3, 2 } }).get_flattened_size() == 5);
	CHECK(Space(MultiBinarySpace{ 7 }).get_flattened_size() == 7);

	DictSpace dict;
	dict.entries.emplace_back("a", Space(DiscreteSpace{ 4 }));
	dict.entries.emplace_back("b", make_box({ 0, 0 }, { 1, 1 }, { 2 }));
	CHECK(Space(std::move(dict)).get_flattened_size() == 6);
}

TEST_CASE("equality compares kind and definition exactly", "[spaces]") {
	CHECK(Space(DiscreteSpace{ 4 }) == Space(DiscreteSpace{ 4 }));
	CHECK(Space(DiscreteSpace{ 4 }) != Space(DiscreteSpace{ 5 }));
	CHECK(Space(DiscreteSpace{ 2 }) != Space(MultiBinarySpace{ 2 }));
	CHECK(make_box({ 0 }, { 1 }, { 1 }) != make_box({ 0 }, { 2 }, { 1 }));

	DictSpace first;
	first.entries.emplace_back("a", Space(DiscreteSpace{ 2 }));
	first.entries.emplace_back("b", Space(DiscreteSpace{ 3 }));
	DictSpace second;
	second.entries.emplace_back("b", Space(DiscreteSpace{ 3 }));
	second.entries.emplace_back("a", Space(DiscreteSpace{ 2 }));
	CHECK(Space(std::move(first)) == Space(std::move(second)));

	CHECK(Point(DiscretePoint{ 1 }) == Point(DiscretePoint{ 1 }));
	CHECK(Point(DiscretePoint{ 1 }) != Point(BoxPoint{ { 1 } }));
}

TEST_CASE("to_string describes the space and point", "[spaces]") {
	CHECK(Space(DiscreteSpace{ 4 }).to_string() == "Discrete(4)");
	CHECK(Space(MultiDiscreteSpace{ { 3, 2 } }).to_string() == "MultiDiscrete([3, 2])");
	CHECK(make_box({ 0 }, { 1 }, { 1 }).to_string() == "Box(low=[0], high=[1], shape=[1], dtype=float32)");
	CHECK(Point(MultiBinaryPoint{ { 0, 1 } }).to_string() == "MultiBinary([0, 1])");
}
