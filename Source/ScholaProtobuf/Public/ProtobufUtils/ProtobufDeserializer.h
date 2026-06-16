// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Points/Point.h"
#include "Points/MultiBinaryPoint.h"
#include "Points/BoxPoint.h"
#include "Points/DiscretePoint.h"
#include "Points/DictPoint.h"
#include "Points/TextPoint.h"
#include "TrainingDataTypes/TrainingUpdate.h"
#include "TrainingDataTypes/EnvironmentUpdate.h"
#include "TrainingDataTypes/StartRequest.h"
#include "Spaces/Space.h"
#include "Spaces/DictSpace.h"
#include "Spaces/MultiBinarySpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/BoxSpaceDimension.h"
#include "Spaces/TextSpace.h"
#include "StructUtils/InstancedStruct.h"
#include "ImitationDataTypes/ImitationState.h"
THIRD_PARTY_INCLUDES_START
#include "ScholaProtobufMacroGuardBegin.h"
#include "GymConnector.pb.h"
#include "Spaces.pb.h"
#include "ImitationState.pb.h"
#include "ScholaProtobufMacroGuardEnd.h"
THIRD_PARTY_INCLUDES_END

/**
 * @brief A class that deserializes protobuf Point messages into Unreal Point structures.
 * @details This class converts protobuf Point messages received from network or storage
 * back into their corresponding Unreal Engine representations (Box, Discrete, MultiBinary,
 * MultiDiscrete, Dict) for use in the Schola framework.
 */
class SCHOLAPROTOBUF_API ProtobufPointDeserializer
{
	TInstancedStruct<FPoint>& DeserializedPoint;

public:
	/**
	 * @brief Constructs a deserializer with a target Unreal Point buffer.
	 * @param[in,out] InitialPoint Reference to the TInstancedStruct<FPoint> to fill during deserialization.
	 */
	ProtobufPointDeserializer(TInstancedStruct<FPoint>& InitialPoint)
		: DeserializedPoint(InitialPoint){};

	/** Dispatches on oneof case and fills DeserializedPoint based on the real type of point. */
	void Deserialize(const Schola::Point& InPoint)
	{
		switch (InPoint.point_case())
		{
			case (Schola::Point::kMultiBinaryPoint):
				this->Deserialize(InPoint.multi_binary_point());
				break;

			case (Schola::Point::kBoxPoint):
				this->Deserialize(InPoint.box_point());
				break;

			case (Schola::Point::kDictPoint):
				this->Deserialize(InPoint.dict_point());
				break;

			case (Schola::Point::kDiscretePoint):
				this->Deserialize(InPoint.discrete_point());
				break;

			case (Schola::Point::kMultiDiscretePoint):
				this->Deserialize(InPoint.multi_discrete_point());
				break;

			case (Schola::Point::kTextPoint):
				this->Deserialize(InPoint.text_point());
				break;
		}
	};

	/** Deserializes a nested dict point into FDictPoint. */
	void Deserialize(const Schola::DictPoint& InDictPoint) 
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Deserialize DictPoint");
		DeserializedPoint.InitializeAs<FDictPoint>();
		FDictPoint& Point = DeserializedPoint.GetMutable<FDictPoint>();
		for (auto& Pair : InDictPoint.values())
		{
			TInstancedStruct<FPoint>& SubPoint = Point.Points.Add(FString(Pair.first.c_str()));
			ProtobufPointDeserializer(SubPoint).Deserialize(Pair.second);
		}
	};

	/** Deserializes a flat multi-binary vector into FMultiBinaryPoint. */
	void Deserialize(const Schola::MultiBinaryPoint& InBinaryPoint) 
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Deserialize MultiBinaryPoint");
		DeserializedPoint.InitializeAs<FMultiBinaryPoint>(InBinaryPoint.values().data(), InBinaryPoint.values_size());
	};

	/** Deserializes box values and shape into FBoxPoint. */
	void Deserialize(const Schola::BoxPoint& InBoxPoint) 
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Deserialize BoxPoint");
		DeserializedPoint.InitializeAs<FBoxPoint>(InBoxPoint.values().data(), InBoxPoint.values_size());
	};

	/** Deserializes multi-discrete indices into FMultiDiscretePoint. */
	void Deserialize(const Schola::MultiDiscretePoint& InMultiDiscretePoint)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Deserialize MultiDiscretePoint");
		DeserializedPoint.InitializeAs<FMultiDiscretePoint>(InMultiDiscretePoint.values().data(), InMultiDiscretePoint.values_size());
	};

	/** Deserializes a single discrete index into FDiscretePoint. */
	void Deserialize(const Schola::DiscretePoint& InDiscretePoint)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Deserialize DiscretePoint");
		DeserializedPoint.InitializeAs<FDiscretePoint>(InDiscretePoint.value());
	};

	/** Deserializes a UTF-8 string into FTextPoint. */
	void Deserialize(const Schola::TextPoint& InTextPoint)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Deserialize TextPoint");
		DeserializedPoint.InitializeAs<FTextPoint>(FString(UTF8_TO_TCHAR(InTextPoint.value().c_str())));
	};

};

/**
 * @brief A class that deserializes protobuf Space messages into Unreal Space structures.
 * @details This class converts protobuf Space messages (which define observation and action spaces)
 * back into their corresponding Unreal Engine representations (Box, Discrete, MultiBinary,
 * MultiDiscrete, Dict) for use in the Schola framework.
 */
class SCHOLAPROTOBUF_API ProtobufSpaceDeserializer
{
	TInstancedStruct<FSpace>& DeserializedSpace;

public:
	/**
	 * @brief Constructs a deserializer with a target Unreal Space buffer.
	 * @param[in,out] InitialSpace Reference to the TInstancedStruct<FSpace> to fill during deserialization.
	 */
	ProtobufSpaceDeserializer(TInstancedStruct<FSpace>& InitialSpace)
		: DeserializedSpace(InitialSpace) {}

	/** Dispatches on space oneof case and fills DeserializedSpace. */
	void Deserialize(const Schola::Space& InSpace)
	{
		switch (InSpace.space_case())
		{
			case Schola::Space::kDictSpace:
				this->Deserialize(InSpace.dict_space());
				break;
			case Schola::Space::kMultiBinarySpace:
				this->Deserialize(InSpace.multi_binary_space());
				break;
			case Schola::Space::kMultiDiscreteSpace:
				this->Deserialize(InSpace.multi_discrete_space());
				break;
			case Schola::Space::kDiscreteSpace:
				this->Deserialize(InSpace.discrete_space());
				break;
			case Schola::Space::kBoxSpace:
				this->Deserialize(InSpace.box_space());
				break;
			case Schola::Space::kTextSpace:
				this->Deserialize(InSpace.text_space());
				break;
			case Schola::Space::SPACE_NOT_SET:
			default:
				break;
		}
	};

	/** Deserializes nested named subspaces into FDictSpace. */
	void Deserialize(const Schola::DictSpace& InDictSpace)
	{
		DeserializedSpace.InitializeAs<FDictSpace>();
		FDictSpace& Space = DeserializedSpace.GetMutable<FDictSpace>();
		for (auto& Pair : InDictSpace.spaces())
		{
			TInstancedStruct<FSpace>& SubSpace = Space.Spaces.Add(FString(Pair.first.c_str()));
			ProtobufSpaceDeserializer(SubSpace).Deserialize(Pair.second);
		}
	};

	/** Deserializes shape into FMultiBinarySpace. */
	void Deserialize(const Schola::MultiBinarySpace& InMultiBinarySpace)
	{
		DeserializedSpace.InitializeAs<FMultiBinarySpace>(InMultiBinarySpace.shape());
	};

	/** Deserializes upper bound into FDiscreteSpace. */
	void Deserialize(const Schola::DiscreteSpace& InDiscreteSpace)
	{
		DeserializedSpace.InitializeAs<FDiscreteSpace>(InDiscreteSpace.high());
	};

	/** Deserializes per-axis highs into FMultiDiscreteSpace. */
	void Deserialize(const Schola::MultiDiscreteSpace& InMultiDiscreteSpace)
	{
		DeserializedSpace.InitializeAs<FMultiDiscreteSpace>(InMultiDiscreteSpace.high().data(), InMultiDiscreteSpace.high_size());
	};

	/** Deserializes per-dimension bounds and global shape into FBoxSpace. */
	void Deserialize(const Schola::BoxSpace& InBoxSpace)
	{
		DeserializedSpace.InitializeAs<FBoxSpace>();
		FBoxSpace& Space = DeserializedSpace.GetMutable<FBoxSpace>();
		// Dimensions
		Space.Dimensions.Empty();
		Space.Dimensions.Reserve(InBoxSpace.dimensions_size());
		for (const auto& Dim : InBoxSpace.dimensions())
		{
			Space.Dimensions.Add(FBoxSpaceDimension(Dim.low(), Dim.high()));
		}
		// Shape
		Space.Shape.Empty();
		Space.Shape.Append(InBoxSpace.shape_dimensions().data(), InBoxSpace.shape_dimensions_size());
	};

	/** Deserializes length bounds and charset into FTextSpace. */
	void Deserialize(const Schola::TextSpace& InTextSpace)
	{
		DeserializedSpace.InitializeAs<FTextSpace>();
		FTextSpace& Space = DeserializedSpace.GetMutable<FTextSpace>();
		Space.MaxLength = InTextSpace.max_length();
		Space.MinLength = InTextSpace.min_length();
		Space.Charset = FString(UTF8_TO_TCHAR(InTextSpace.charset().c_str()));
	};
};


/**
 * @brief A namespace containing functions to deserialize protobuf messages into Unreal Engine types
 */
namespace ProtobufDeserializer
{

	/** Scalar / identity conversion when protobuf and Unreal types match. (e.g. int)
	 * @tparam UnrealV Unreal type to convert to.
	 * @tparam ProtoV Protobuf type to convert from.
	 * @param[in] InProtobufValue Protobuf value to convert from.
	 * @param[out] OutUnrealValue Unreal value to convert to.
	 */
	template<typename UnrealV, typename ProtoV>
	inline void FromProto(const ProtoV& InProtobufValue, UnrealV& OutUnrealValue)
	{
		OutUnrealValue = InProtobufValue;
	}

	/** Copies a protobuf repeated field of primitives into a TArray. 
	 * @tparam ProtoV Element Type that is natively convertable to an Unreal Type.
	 * @param[in] InProtobufRepeatedField Protobuf repeated field to convert from.
	 * @param[out] OutUnrealArray Unreal array to convert to.
	 */
	template <typename ProtoV>
	void FromProto(const google::protobuf::RepeatedField<ProtoV>& InProtobufRepeatedField, TArray<ProtoV>& OutUnrealArray)
	{
		OutUnrealArray.Empty();
		OutUnrealArray.Reserve(InProtobufRepeatedField.size());
		for (int i = 0; i < InProtobufRepeatedField.size(); ++i)
		{
			OutUnrealArray.Add(InProtobufRepeatedField.Get(i));
		}
	}

	/** Converts each element of a repeated message field via FromProto element-wise. 
	 * @tparam UnrealV Unreal type to convert to.
	 * @tparam ProtoV Protobuf type to convert from.
	 * @param[in] InProtobufRepeatedField Protobuf repeated field to convert from.
	 * @param[out] OutUnrealArray Unreal array to convert to.
	 */
	template <typename UnrealV, typename ProtoV>
	void FromProto(const google::protobuf::RepeatedPtrField<ProtoV>& InProtobufRepeatedField, TArray<UnrealV>& OutUnrealArray)
	{
		OutUnrealArray.Empty();
		OutUnrealArray.Reserve(InProtobufRepeatedField.size());
		for (int i = 0; i < InProtobufRepeatedField.size(); ++i)
		{
			UnrealV Value;
			FromProto<UnrealV, ProtoV>(InProtobufRepeatedField.Get(i), Value);
			OutUnrealArray.Add(MoveTemp(Value));
		}
	}

	/** Converts a protobuf map with arbitrary key type into TMap. 
	 * @tparam UnrealV Unreal type to convert to.
	 * @tparam ProtoV Protobuf type to convert from.
	 * @tparam KeyType Type of the key in the protobuf map. Supported in both Unreal and Protobuf.
	 * @param[in] InProtobufMap Protobuf map to convert from.
	 * @param[out] OutUnrealMap Unreal map to convert to.
	 */
	template <typename UnrealV, typename ProtoV, typename KeyType>
	void FromProto(const google::protobuf::Map<KeyType, ProtoV>& InProtobufMap, TMap<KeyType, UnrealV>& OutUnrealMap)
	{
		OutUnrealMap.Empty();
		for (const auto& Pair : InProtobufMap)
		{
			UnrealV Value;
			FromProto<UnrealV, ProtoV>(Pair.second, Value);
			OutUnrealMap.Add(Pair.first, MoveTemp(Value));
		}
	}

	/** Converts a string-keyed protobuf map into TMap<FString, ...>. 
	 * @tparam UnrealV Unreal type to convert to.
	 * @tparam ProtoV Protobuf type to convert from.
	 * @param[in] InProtobufMap Protobuf map to convert from.
	 * @param[out] OutUnrealMap Unreal map to convert to.
	 */
	template <typename UnrealV, typename ProtoV>
	void FromProto(const google::protobuf::Map<std::string, ProtoV>& InProtobufMap, TMap<FString, UnrealV>& OutUnrealMap)
	{
		OutUnrealMap.Empty();
		for (const auto& Pair : InProtobufMap)
		{
			UnrealV Value;
			FromProto<UnrealV, ProtoV>(Pair.second, Value);
			OutUnrealMap.Add(FString(Pair.first.c_str()), MoveTemp(Value));
		}
	}

	// Update Messages
	/** Parses connector status and nested step or reset payload into FTrainingStateUpdate. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::StateUpdate& InStateUpdate, FTrainingStateUpdate& OutTrainingStateUpdate);

	/** Populates multi-env reset data from protobuf. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::Reset& InResetProto, FTrainingReset& OutTrainingStateUpdate);

	/** Populates multi-env step data from protobuf. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::Step& InStepProto, FTrainingStep& OutTrainingStateUpdate);

	/** Converts environment reset options and optional seed into FEnvReset. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::EnvironmentSettings& InEnvSettingsProto, FEnvReset& OutEnvResetSettings);

	/** Converts one environment step update (actions, etc.) into FEnvStep. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::EnvironmentStep& InEnvStepProto, FEnvStep& OutEnvStep);

	/** Parses gym connector start request (e.g. autoreset mode). */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::GymConnectorStartRequest& InStartRequestProto, FStartRequest& OutStartRequest);

	// Spaces and Points

	/** Deserializes a protobuf Point oneof into OutPoint. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::Point& InPointProto, TInstancedStruct<FPoint>& OutPoint);

	/** Deserializes a protobuf Space oneof into OutSpace. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::Space& InSpaceProto, TInstancedStruct<FSpace>& OutSpace);

	// Basic Types
	/** UTF-8 std::string to FString. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const std::string& InString, FString& OutString);

	// Imitation Learning Messages
	/** Converts one imitation agent  state into FImitationAgentState. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::ImitationAgentState& InImitationAgentStateProto, FImitationAgentState& OutImitationAgentState);

	/** Converts one imitation environment state into FImitationEnvironmentState. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::ImitationEnvironmentState& InImitationEnvStateProto, FImitationEnvironmentState& OutImitationEnvState);

	/** Parses connector status and nested step or reset payload into FImitationTrainingState. */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::ImitationTrainingState& InImitationTrainingStateProto, FImitationTrainingState& OutImitationTrainingState);

	/** Parses root imitation message (training + optional initial state). */
	template<>
	SCHOLAPROTOBUF_API void FromProto(const Schola::ImitationState& InImitationStateProto, FImitationState& OutImitationState);
};


