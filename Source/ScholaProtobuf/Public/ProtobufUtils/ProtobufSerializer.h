// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "Points/Point.h"
#include "Points/MultiBinaryPoint.h"
#include "Points/BoxPoint.h"
#include "Points/DictPoint.h"
#include "Points/DiscretePoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Points/TextPoint.h"
#include "Points/PointVisitor.h"	
#include "Spaces/SpaceVisitor.h"
#include "Spaces/DictSpace.h"
#include "Spaces/MultiBinarySpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/TextSpace.h"
#include "Containers/StringConv.h"
#include "TrainingDataTypes/EnvironmentState.h"
#include "TrainingDataTypes/TrainingState.h"
#include "TrainingDataTypes/TrainingDefinition.h"
#include "ImitationDataTypes/ImitationAgentState.h"
#include "ImitationDataTypes/ImitationTrainingState.h"
#include "ImitationDataTypes/ImitationEnvironmentState.h"
#include "StructUtils/InstancedStruct.h"
THIRD_PARTY_INCLUDES_START
#include "ScholaProtobufMacroGuardBegin.h"
#include "Spaces.pb.h"
#include "Points.pb.h"
#include "GymConnector.pb.h"
#include "ImitationState.pb.h"
#include "ScholaProtobufMacroGuardEnd.h"
THIRD_PARTY_INCLUDES_END

using Schola::DictPoint;
using Schola::MultiBinaryPoint;
using Schola::MultiDiscretePoint;
using Schola::BoxPoint;
using Schola::Point;
using Schola::BoxSpace_BoxSpaceDimension;
/**
 * @brief A visitor class that serializes Unreal Point structures into protobuf Point messages.
 * @details This class implements the visitor pattern to traverse different Point types
 * (Box, Discrete, MultiBinary, MultiDiscrete, Dict) and serialize them into their
 * corresponding protobuf representations for network transmission or storage.
 */
class SCHOLAPROTOBUF_API ProtobufPointSerializer : public ConstPointVisitor
{
	Point* SerializedPointBuffer;

public:
	/**
	 * @brief Constructs a serializer with a target protobuf Point buffer.
	 * @param[in] InitialPoint Pointer to the protobuf Point object to fill during traversal.
	 */
	ProtobufPointSerializer(Point* InitialPoint)
		: SerializedPointBuffer(InitialPoint){};

	void operator()(const FDictPoint& Point) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize DictPoint");
		for (const TPair<FString,TInstancedStruct<FPoint>>& Pair : Point.Points)
		{
			DictPoint* ConcretePoint = SerializedPointBuffer->mutable_dict_point();
			Schola::Point&		ConcretePointEntry = (*ConcretePoint->mutable_values())[TCHAR_TO_UTF8(*(Pair.Key))];
			ProtobufPointSerializer NewSerializer = ProtobufPointSerializer(&ConcretePointEntry);
			Pair.Value.Get<FPoint>().Accept(NewSerializer);
		}
	};

	void operator()(const FMultiBinaryPoint& Point) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize MultiBinaryPoint");
		Schola::MultiBinaryPoint* PointMsg = SerializedPointBuffer->mutable_multi_binary_point();
		for (auto& PointValue : Point.Values)
		{
			PointMsg->add_values(PointValue);
		}
	};

	void operator()(const FDiscretePoint& Point) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize DiscretePoint");
		Schola::DiscretePoint* PointMsg = SerializedPointBuffer->mutable_discrete_point();
		PointMsg->set_value(Point.Value);
	};

	void operator()(const FMultiDiscretePoint& Point) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize MultiDiscretePoint");
		Schola::MultiDiscretePoint* PointMsg = SerializedPointBuffer->mutable_multi_discrete_point();
		// PointMsg->mutable_values()->Add(Point.Values.begin(), Point.Values.end()); leads to a compile error here
		for (auto& PointValue : Point.Values)
		{
			PointMsg->add_values(PointValue);
		}
	};

	void operator()(const FBoxPoint& Point) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize BoxPoint");
		BoxPoint* PointMsg = SerializedPointBuffer->mutable_box_point();
		for (auto& PointValue : Point.Values)
		{
			PointMsg->add_values(PointValue);
		}
		
		for (auto& ShapeDimension : Point.Shape)
		{
			PointMsg->add_shape(ShapeDimension);
		}
	};

	void operator()(const FTextPoint& Point) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize TextPoint");
		Schola::TextPoint* PointMsg = SerializedPointBuffer->mutable_text_point();
		PointMsg->set_value(TCHAR_TO_UTF8(*Point.Value));
	};

	/** @return Underlying Schola::Point buffer (e.g. after visiting dict entries). */
	Schola::Point* GetDictPoint()
	{
		return SerializedPointBuffer;
	}

};

/**
 * @brief A visitor class that serializes Unreal Space structures into protobuf Space messages.
 * @details This class implements the visitor pattern to traverse different Space types
 * (Box, Discrete, MultiBinary, MultiDiscrete, Dict) and serialize them into their
 * corresponding protobuf representations for defining observation and action spaces.
 */
class SCHOLAPROTOBUF_API ProtobufSpaceSerializer : public ConstSpaceVisitor
{
	Schola::Space* SerializedSpaceBuffer;

public:
	/**
	 * @brief Constructs a serializer with a target protobuf Space buffer.
	 * @param[in] InitialSpace Pointer to the protobuf Space object to fill during traversal.
	 */
	ProtobufSpaceSerializer(Schola::Space* InitialSpace)
		: SerializedSpaceBuffer(InitialSpace){};

	void operator()(const FDictSpace& Space) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize DictSpace");
		Schola::DictSpace* ConcreteSpace = SerializedSpaceBuffer->mutable_dict_space();
		for (const TPair<FString, TInstancedStruct<FSpace>>& Pair : Space.Spaces)
		{
			Schola::Space&			ConcretePointEntry = (*ConcreteSpace->mutable_spaces())[TCHAR_TO_UTF8(*(Pair.Key))];
			ProtobufSpaceSerializer NewSerializer = ProtobufSpaceSerializer(&ConcretePointEntry);
			Pair.Value.Get<FSpace>().Accept(NewSerializer);
		}
	};

	void operator()(const FMultiBinarySpace& Space) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize MultiBinarySpace");
		Schola::MultiBinarySpace* ConcreteSpace = SerializedSpaceBuffer->mutable_multi_binary_space();
		ConcreteSpace->set_shape(Space.Shape);
	};

	void operator()(const FDiscreteSpace& Space) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize DiscreteSpace");
		Schola::DiscreteSpace* ConcreteSpace = SerializedSpaceBuffer->mutable_discrete_space();
		ConcreteSpace->set_high(Space.High);
	};

	void operator()(const FMultiDiscreteSpace& Space) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize MultiDiscreteSpace");
		Schola::MultiDiscreteSpace* ConcreteSpace = SerializedSpaceBuffer->mutable_multi_discrete_space();
		for (int SpaceValue : Space.High)
		{
			ConcreteSpace->add_high(SpaceValue);
		}
	};

	void operator()(const FBoxSpace& Space) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize BoxSpace");
		Schola::BoxSpace* ConcreteSpace = SerializedSpaceBuffer->mutable_box_space();
		for (const FBoxSpaceDimension& SpaceValue : Space.Dimensions)
		{
			Schola::BoxSpace_BoxSpaceDimension* Dimension = ConcreteSpace->add_dimensions();
			Dimension->set_low(SpaceValue.Low);
			Dimension->set_high(SpaceValue.High);
		}
		for (int DimensionSize : Space.Shape)
		{
			ConcreteSpace->add_shape_dimensions(DimensionSize);
		}
	};

	void operator()(const FTextSpace& Space) override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: Serialize TextSpace");
		Schola::TextSpace* ConcreteSpace = SerializedSpaceBuffer->mutable_text_space();
		ConcreteSpace->set_max_length(Space.MaxLength);
		ConcreteSpace->set_min_length(Space.MinLength);
		if (!Space.Charset.IsEmpty())
		{
			ConcreteSpace->set_charset(TCHAR_TO_UTF8(*Space.Charset));
		}
	};
};

namespace ProtobufSerializer
{	

	//Generic conversions
	
	template<typename UnrealV, typename ProtoV>
	inline void ToProto(const UnrealV& InUnrealValue, ProtoV* OutProtobufValue)
	{
		*OutProtobufValue = InUnrealValue;
	}

	template <typename UnrealV, typename ProtoV>
	inline void ToProto(const UnrealV& InUnrealValue, ProtoV& OutProtobufValue)
	{
		ProtobufSerializer::ToProto(InUnrealValue, &OutProtobufValue);
	}

	template <typename UnrealV, typename ProtoV>
	void ToProto(const TArray<UnrealV>& InUnrealArray, google::protobuf::RepeatedField<ProtoV>* OutProtobufRepeatedField)
	{
		for (const UnrealV& UnrealValue : InUnrealArray)
		{
			// Use the function pointer to convert UnrealV to ProtoV
			ProtoV* ProtoValue = OutProtobufRepeatedField->Add();
			ProtobufSerializer::ToProto<UnrealV, ProtoV>(UnrealValue, ProtoValue);
		}
	}

	template <typename UnrealV, typename ProtoV>
	void ToProto(const TArray<UnrealV>& InUnrealArray, google::protobuf::RepeatedPtrField<ProtoV>* OutProtobufRepeatedField)
	{
		for (const UnrealV& UnrealValue : InUnrealArray)
		{
			// Use the function pointer to convert UnrealV to ProtoV
			ProtoV* ProtoValue = OutProtobufRepeatedField->Add();
			ProtobufSerializer::ToProto<UnrealV, ProtoV>(UnrealValue, ProtoValue);
		}
	}

	template <typename ValueType>
	void ToProto(const ValueType& Value, void (*ProtoMsgSetter)(ValueType))
	{
		ProtoMsgSetter(Value);
	}

	template <typename UnrealV, typename ProtoV, typename KeyType>
	void ToProto(const TMap<KeyType, UnrealV>& InUnrealMap, google::protobuf::Map<KeyType, ProtoV>* OutProtobufMap)
	{
		for (const TPair<KeyType, UnrealV>& Pair : InUnrealMap)
		{
			// Use the function pointer to convert UnrealV to ProtoV
			ProtoV& ProtoValue = (*OutProtobufMap)[Pair.Key];
			ProtobufSerializer::ToProto<UnrealV, ProtoV>(Pair.Value, &ProtoValue);
		}
	}

	template <typename UnrealV, typename ProtoV, typename KeyType>
	void ToProto(const TArray<UnrealV>& InValueArray, const TArray<KeyType>& InKeyArray, google::protobuf::Map<KeyType, ProtoV>* OutProtobufMap)
	{
		//Note InValueArray is not condensed so it might contain [A,B,C] and KeyArray is [1] leading to output {1:B}
		for (int Index=0; Index < InKeyArray.Num(); Index++)
		{
			// Use the function pointer to convert UnrealV to ProtoV
			ProtoV& ProtoValue = (*OutProtobufMap)[InKeyArray[Index]];
			ProtobufSerializer::ToProto<UnrealV, ProtoV>(InValueArray[InKeyArray[Index]], &ProtoValue);
		}
	}

	template <typename UnrealV, typename ProtoV>
	void ToProto(const TArray<UnrealV>& InValueArray, google::protobuf::Map<int, ProtoV>* OutProtobufMap)
	{
		// Note InValueArray is not condensed so it might contain [A,B,C] and KeyArray is [1] leading to output {1:B}
		for (int Index = 0; Index < InValueArray.Num(); Index++)
		{
			// Use the function pointer to convert UnrealV to ProtoV
			ProtoV& ProtoValue = (*OutProtobufMap)[Index];
			ProtobufSerializer::ToProto<UnrealV, ProtoV>(InValueArray[Index], &ProtoValue);
		}
	}

	template <typename UnrealV, typename ProtoV>
	void ToProto(const TMap<FString, UnrealV>& InUnrealMap, google::protobuf::Map<std::string, ProtoV>* OutProtobufMap)
	{
		for (const TPair<FString, UnrealV>& Pair : InUnrealMap)
		{
			// Use the function pointer to convert UnrealV to ProtoV
			ProtoV& ProtoValue = (*OutProtobufMap)[TCHAR_TO_UTF8(*Pair.Key)];
			ProtobufSerializer::ToProto<UnrealV, ProtoV>(Pair.Value, &ProtoValue);
		}
	}

	//Specific Conversions

	template <>
	SCHOLAPROTOBUF_API void ToProto(const FString& InUnrealString, std::string* OutProtobufString);

	//Mid Episode Messages

	template<>
	SCHOLAPROTOBUF_API void ToProto(const FTrainingState& InTrainingState, Schola::TrainingState* OutTrainingStateProto);

	template <>
	SCHOLAPROTOBUF_API void ToProto(const FEnvironmentState& InEnvironmentState, Schola::EnvironmentState* OutEnvironmentStateProto);
	
	template <>
	SCHOLAPROTOBUF_API void ToProto(const FAgentState& InAgentState, Schola::AgentState* OutAgentStateProto);

	// Start of Episode Messages

	template <>
	SCHOLAPROTOBUF_API void ToProto(const FInitialState& InInitialState, Schola::InitialState* OutInitialStateProto);

	template <>
	SCHOLAPROTOBUF_API void ToProto(const FInitialEnvironmentState& InInitialEnvironmentState, Schola::InitialEnvironmentState* OutInitialEnvironmentStateProto);

	template <>
	SCHOLAPROTOBUF_API void ToProto(const FInitialAgentState& InInitialAgentState, Schola::InitialAgentState* OutInitialAgentStateProto);


	// Definition Messages
	template <>
	SCHOLAPROTOBUF_API void ToProto(const FTrainingDefinition& InTrainingDefinition, Schola::TrainingDefinition* OutTrainingDefinitionProto);

	template <>
	SCHOLAPROTOBUF_API void ToProto(const FInteractionDefinition& InInteractionDefinition, Schola::AgentDefinition* OutInteractionDefinitionProto);

	template <>
	SCHOLAPROTOBUF_API void ToProto(const FEnvironmentDefinition& InEnvironmentDefinition, Schola::EnvironmentDefinition* OutEnvironmentDefinition);

	//Imitation Messages

	template<>
	SCHOLAPROTOBUF_API void ToProto(const FImitationTrainingState& InState, Schola::ImitationTrainingState* OutStateProto);

	template<>
	SCHOLAPROTOBUF_API void ToProto(const FImitationEnvironmentState& InEnvState, Schola::ImitationEnvironmentState* OutEnvStateProto);

	template<>
	SCHOLAPROTOBUF_API void ToProto(const FImitationAgentState& InAgentState, Schola::ImitationAgentState* OutAgentStateProto);

	//Handling for Spaces and Points

	template<>
	SCHOLAPROTOBUF_API void ToProto(const TInstancedStruct<FSpace>& InSpace, Schola::Space* OutSpaceProto);

	template <>
	SCHOLAPROTOBUF_API void ToProto(const TInstancedStruct<FPoint>& InPoint, Schola::Point* OutPointProto);


};

