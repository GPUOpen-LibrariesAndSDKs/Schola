// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: StateUpdates.proto

#pragma warning (disable : 4800)
#pragma warning (disable : 4125)
#include "StateUpdates.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG

namespace _pb = ::PROTOBUF_NAMESPACE_ID;
namespace _pbi = _pb::internal;

namespace Schola {
PROTOBUF_CONSTEXPR AgentStateUpdate::AgentStateUpdate(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.actions_)*/nullptr
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct AgentStateUpdateDefaultTypeInternal {
  PROTOBUF_CONSTEXPR AgentStateUpdateDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~AgentStateUpdateDefaultTypeInternal() {}
  union {
    AgentStateUpdate _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 AgentStateUpdateDefaultTypeInternal _AgentStateUpdate_default_instance_;
PROTOBUF_CONSTEXPR EnvironmentStep_UpdatesEntry_DoNotUse::EnvironmentStep_UpdatesEntry_DoNotUse(
    ::_pbi::ConstantInitialized) {}
struct EnvironmentStep_UpdatesEntry_DoNotUseDefaultTypeInternal {
  PROTOBUF_CONSTEXPR EnvironmentStep_UpdatesEntry_DoNotUseDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~EnvironmentStep_UpdatesEntry_DoNotUseDefaultTypeInternal() {}
  union {
    EnvironmentStep_UpdatesEntry_DoNotUse _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 EnvironmentStep_UpdatesEntry_DoNotUseDefaultTypeInternal _EnvironmentStep_UpdatesEntry_DoNotUse_default_instance_;
PROTOBUF_CONSTEXPR EnvironmentStep::EnvironmentStep(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.updates_)*/{::_pbi::ConstantInitialized()}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct EnvironmentStepDefaultTypeInternal {
  PROTOBUF_CONSTEXPR EnvironmentStepDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~EnvironmentStepDefaultTypeInternal() {}
  union {
    EnvironmentStep _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 EnvironmentStepDefaultTypeInternal _EnvironmentStep_default_instance_;
}  // namespace Schola
static ::_pb::Metadata file_level_metadata_StateUpdates_2eproto[3];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_StateUpdates_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_StateUpdates_2eproto = nullptr;

const uint32_t TableStruct_StateUpdates_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::Schola::AgentStateUpdate, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::Schola::AgentStateUpdate, _impl_.actions_),
  PROTOBUF_FIELD_OFFSET(::Schola::EnvironmentStep_UpdatesEntry_DoNotUse, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::Schola::EnvironmentStep_UpdatesEntry_DoNotUse, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::Schola::EnvironmentStep_UpdatesEntry_DoNotUse, key_),
  PROTOBUF_FIELD_OFFSET(::Schola::EnvironmentStep_UpdatesEntry_DoNotUse, value_),
  0,
  1,
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::Schola::EnvironmentStep, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::Schola::EnvironmentStep, _impl_.updates_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::Schola::AgentStateUpdate)},
  { 7, 15, -1, sizeof(::Schola::EnvironmentStep_UpdatesEntry_DoNotUse)},
  { 17, -1, -1, sizeof(::Schola::EnvironmentStep)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::Schola::_AgentStateUpdate_default_instance_._instance,
  &::Schola::_EnvironmentStep_UpdatesEntry_DoNotUse_default_instance_._instance,
  &::Schola::_EnvironmentStep_default_instance_._instance,
};

const char descriptor_table_protodef_StateUpdates_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\022StateUpdates.proto\022\006Schola\032\014Points.pro"
  "to\"6\n\020AgentStateUpdate\022\"\n\007actions\030\001 \001(\0132"
  "\021.Schola.DictPoint\"\222\001\n\017EnvironmentStep\0225"
  "\n\007updates\030\001 \003(\0132$.Schola.EnvironmentStep"
  ".UpdatesEntry\032H\n\014UpdatesEntry\022\013\n\003key\030\001 \001"
  "(\005\022\'\n\005value\030\002 \001(\0132\030.Schola.AgentStateUpd"
  "ate:\0028\001b\006proto3"
  ;
static const ::_pbi::DescriptorTable* const descriptor_table_StateUpdates_2eproto_deps[1] = {
  &::descriptor_table_Points_2eproto,
};
static ::_pbi::once_flag descriptor_table_StateUpdates_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_StateUpdates_2eproto = {
    false, false, 255, descriptor_table_protodef_StateUpdates_2eproto,
    "StateUpdates.proto",
    &descriptor_table_StateUpdates_2eproto_once, descriptor_table_StateUpdates_2eproto_deps, 1, 3,
    schemas, file_default_instances, TableStruct_StateUpdates_2eproto::offsets,
    file_level_metadata_StateUpdates_2eproto, file_level_enum_descriptors_StateUpdates_2eproto,
    file_level_service_descriptors_StateUpdates_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_StateUpdates_2eproto_getter() {
  return &descriptor_table_StateUpdates_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_StateUpdates_2eproto(&descriptor_table_StateUpdates_2eproto);
namespace Schola {

// ===================================================================

class AgentStateUpdate::_Internal {
 public:
  static const ::Schola::DictPoint& actions(const AgentStateUpdate* msg);
};

const ::Schola::DictPoint&
AgentStateUpdate::_Internal::actions(const AgentStateUpdate* msg) {
  return *msg->_impl_.actions_;
}
void AgentStateUpdate::clear_actions() {
  if (GetArenaForAllocation() == nullptr && _impl_.actions_ != nullptr) {
    delete _impl_.actions_;
  }
  _impl_.actions_ = nullptr;
}
AgentStateUpdate::AgentStateUpdate(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:Schola.AgentStateUpdate)
}
AgentStateUpdate::AgentStateUpdate(const AgentStateUpdate& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  AgentStateUpdate* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.actions_){nullptr}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  if (from._internal_has_actions()) {
    _this->_impl_.actions_ = new ::Schola::DictPoint(*from._impl_.actions_);
  }
  // @@protoc_insertion_point(copy_constructor:Schola.AgentStateUpdate)
}

inline void AgentStateUpdate::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.actions_){nullptr}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

AgentStateUpdate::~AgentStateUpdate() {
  // @@protoc_insertion_point(destructor:Schola.AgentStateUpdate)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void AgentStateUpdate::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  if (this != internal_default_instance()) delete _impl_.actions_;
}

void AgentStateUpdate::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void AgentStateUpdate::Clear() {
// @@protoc_insertion_point(message_clear_start:Schola.AgentStateUpdate)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  if (GetArenaForAllocation() == nullptr && _impl_.actions_ != nullptr) {
    delete _impl_.actions_;
  }
  _impl_.actions_ = nullptr;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* AgentStateUpdate::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // .Schola.DictPoint actions = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr = ctx->ParseMessage(_internal_mutable_actions(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* AgentStateUpdate::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:Schola.AgentStateUpdate)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // .Schola.DictPoint actions = 1;
  if (this->_internal_has_actions()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(1, _Internal::actions(this),
        _Internal::actions(this).GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:Schola.AgentStateUpdate)
  return target;
}

size_t AgentStateUpdate::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:Schola.AgentStateUpdate)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // .Schola.DictPoint actions = 1;
  if (this->_internal_has_actions()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *_impl_.actions_);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData AgentStateUpdate::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    AgentStateUpdate::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*AgentStateUpdate::GetClassData() const { return &_class_data_; }


void AgentStateUpdate::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<AgentStateUpdate*>(&to_msg);
  auto& from = static_cast<const AgentStateUpdate&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:Schola.AgentStateUpdate)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (from._internal_has_actions()) {
    _this->_internal_mutable_actions()->::Schola::DictPoint::MergeFrom(
        from._internal_actions());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void AgentStateUpdate::CopyFrom(const AgentStateUpdate& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:Schola.AgentStateUpdate)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool AgentStateUpdate::IsInitialized() const {
  return true;
}

void AgentStateUpdate::InternalSwap(AgentStateUpdate* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_impl_.actions_, other->_impl_.actions_);
}

::PROTOBUF_NAMESPACE_ID::Metadata AgentStateUpdate::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_StateUpdates_2eproto_getter, &descriptor_table_StateUpdates_2eproto_once,
      file_level_metadata_StateUpdates_2eproto[0]);
}

// ===================================================================

EnvironmentStep_UpdatesEntry_DoNotUse::EnvironmentStep_UpdatesEntry_DoNotUse() {}
EnvironmentStep_UpdatesEntry_DoNotUse::EnvironmentStep_UpdatesEntry_DoNotUse(::PROTOBUF_NAMESPACE_ID::Arena* arena)
    : SuperType(arena) {}
void EnvironmentStep_UpdatesEntry_DoNotUse::MergeFrom(const EnvironmentStep_UpdatesEntry_DoNotUse& other) {
  MergeFromInternal(other);
}
::PROTOBUF_NAMESPACE_ID::Metadata EnvironmentStep_UpdatesEntry_DoNotUse::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_StateUpdates_2eproto_getter, &descriptor_table_StateUpdates_2eproto_once,
      file_level_metadata_StateUpdates_2eproto[1]);
}

// ===================================================================

class EnvironmentStep::_Internal {
 public:
};

EnvironmentStep::EnvironmentStep(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  if (arena != nullptr && !is_message_owned) {
    arena->OwnCustomDestructor(this, &EnvironmentStep::ArenaDtor);
  }
  // @@protoc_insertion_point(arena_constructor:Schola.EnvironmentStep)
}
EnvironmentStep::EnvironmentStep(const EnvironmentStep& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  EnvironmentStep* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      /*decltype(_impl_.updates_)*/{}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  _this->_impl_.updates_.MergeFrom(from._impl_.updates_);
  // @@protoc_insertion_point(copy_constructor:Schola.EnvironmentStep)
}

inline void EnvironmentStep::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      /*decltype(_impl_.updates_)*/{::_pbi::ArenaInitialized(), arena}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

EnvironmentStep::~EnvironmentStep() {
  // @@protoc_insertion_point(destructor:Schola.EnvironmentStep)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    ArenaDtor(this);
    return;
  }
  SharedDtor();
}

inline void EnvironmentStep::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.updates_.Destruct();
  _impl_.updates_.~MapField();
}

void EnvironmentStep::ArenaDtor(void* object) {
  EnvironmentStep* _this = reinterpret_cast< EnvironmentStep* >(object);
  _this->_impl_.updates_.Destruct();
}
void EnvironmentStep::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void EnvironmentStep::Clear() {
// @@protoc_insertion_point(message_clear_start:Schola.EnvironmentStep)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.updates_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* EnvironmentStep::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // map<int32, .Schola.AgentStateUpdate> updates = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(&_impl_.updates_, ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* EnvironmentStep::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:Schola.EnvironmentStep)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // map<int32, .Schola.AgentStateUpdate> updates = 1;
  if (!this->_internal_updates().empty()) {
    using MapType = ::_pb::Map<int32_t, ::Schola::AgentStateUpdate>;
    using WireHelper = EnvironmentStep_UpdatesEntry_DoNotUse::Funcs;
    const auto& map_field = this->_internal_updates();

    if (stream->IsSerializationDeterministic() && map_field.size() > 1) {
      for (const auto& entry : ::_pbi::MapSorterFlat<MapType>(map_field)) {
        target = WireHelper::InternalSerialize(1, entry.first, entry.second, target, stream);
      }
    } else {
      for (const auto& entry : map_field) {
        target = WireHelper::InternalSerialize(1, entry.first, entry.second, target, stream);
      }
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:Schola.EnvironmentStep)
  return target;
}

size_t EnvironmentStep::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:Schola.EnvironmentStep)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // map<int32, .Schola.AgentStateUpdate> updates = 1;
  total_size += 1 *
      ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_updates_size());
  for (::PROTOBUF_NAMESPACE_ID::Map< int32_t, ::Schola::AgentStateUpdate >::const_iterator
      it = this->_internal_updates().begin();
      it != this->_internal_updates().end(); ++it) {
    total_size += EnvironmentStep_UpdatesEntry_DoNotUse::Funcs::ByteSizeLong(it->first, it->second);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData EnvironmentStep::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    EnvironmentStep::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*EnvironmentStep::GetClassData() const { return &_class_data_; }


void EnvironmentStep::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<EnvironmentStep*>(&to_msg);
  auto& from = static_cast<const EnvironmentStep&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:Schola.EnvironmentStep)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.updates_.MergeFrom(from._impl_.updates_);
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void EnvironmentStep::CopyFrom(const EnvironmentStep& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:Schola.EnvironmentStep)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool EnvironmentStep::IsInitialized() const {
  return true;
}

void EnvironmentStep::InternalSwap(EnvironmentStep* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.updates_.InternalSwap(&other->_impl_.updates_);
}

::PROTOBUF_NAMESPACE_ID::Metadata EnvironmentStep::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_StateUpdates_2eproto_getter, &descriptor_table_StateUpdates_2eproto_once,
      file_level_metadata_StateUpdates_2eproto[2]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace Schola
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::Schola::AgentStateUpdate*
Arena::CreateMaybeMessage< ::Schola::AgentStateUpdate >(Arena* arena) {
  return Arena::CreateMessageInternal< ::Schola::AgentStateUpdate >(arena);
}
template<> PROTOBUF_NOINLINE ::Schola::EnvironmentStep_UpdatesEntry_DoNotUse*
Arena::CreateMaybeMessage< ::Schola::EnvironmentStep_UpdatesEntry_DoNotUse >(Arena* arena) {
  return Arena::CreateMessageInternal< ::Schola::EnvironmentStep_UpdatesEntry_DoNotUse >(arena);
}
template<> PROTOBUF_NOINLINE ::Schola::EnvironmentStep*
Arena::CreateMaybeMessage< ::Schola::EnvironmentStep >(Arena* arena) {
  return Arena::CreateMessageInternal< ::Schola::EnvironmentStep >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
