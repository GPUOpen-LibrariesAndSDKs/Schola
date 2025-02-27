// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: StateUpdates.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_StateUpdates_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_StateUpdates_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3021000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3021012 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/map.h>  // IWYU pragma: export
#include <google/protobuf/map_entry.h>
#include <google/protobuf/map_field_inl.h>
#include <google/protobuf/unknown_field_set.h>
#include "Points.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_StateUpdates_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_StateUpdates_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_StateUpdates_2eproto;
namespace Schola {
class AgentStateUpdate;
struct AgentStateUpdateDefaultTypeInternal;
extern AgentStateUpdateDefaultTypeInternal _AgentStateUpdate_default_instance_;
class EnvironmentStep;
struct EnvironmentStepDefaultTypeInternal;
extern EnvironmentStepDefaultTypeInternal _EnvironmentStep_default_instance_;
class EnvironmentStep_UpdatesEntry_DoNotUse;
struct EnvironmentStep_UpdatesEntry_DoNotUseDefaultTypeInternal;
extern EnvironmentStep_UpdatesEntry_DoNotUseDefaultTypeInternal _EnvironmentStep_UpdatesEntry_DoNotUse_default_instance_;
}  // namespace Schola
PROTOBUF_NAMESPACE_OPEN
template<> ::Schola::AgentStateUpdate* Arena::CreateMaybeMessage<::Schola::AgentStateUpdate>(Arena*);
template<> ::Schola::EnvironmentStep* Arena::CreateMaybeMessage<::Schola::EnvironmentStep>(Arena*);
template<> ::Schola::EnvironmentStep_UpdatesEntry_DoNotUse* Arena::CreateMaybeMessage<::Schola::EnvironmentStep_UpdatesEntry_DoNotUse>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace Schola {

// ===================================================================

class AgentStateUpdate final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:Schola.AgentStateUpdate) */ {
 public:
  inline AgentStateUpdate() : AgentStateUpdate(nullptr) {}
  ~AgentStateUpdate() override;
  explicit PROTOBUF_CONSTEXPR AgentStateUpdate(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  AgentStateUpdate(const AgentStateUpdate& from);
  AgentStateUpdate(AgentStateUpdate&& from) noexcept
    : AgentStateUpdate() {
    *this = ::std::move(from);
  }

  inline AgentStateUpdate& operator=(const AgentStateUpdate& from) {
    CopyFrom(from);
    return *this;
  }
  inline AgentStateUpdate& operator=(AgentStateUpdate&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const AgentStateUpdate& default_instance() {
    return *internal_default_instance();
  }
  static inline const AgentStateUpdate* internal_default_instance() {
    return reinterpret_cast<const AgentStateUpdate*>(
               &_AgentStateUpdate_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(AgentStateUpdate& a, AgentStateUpdate& b) {
    a.Swap(&b);
  }
  inline void Swap(AgentStateUpdate* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(AgentStateUpdate* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  AgentStateUpdate* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<AgentStateUpdate>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const AgentStateUpdate& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const AgentStateUpdate& from) {
    AgentStateUpdate::MergeImpl(*this, from);
  }
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::PROTOBUF_NAMESPACE_ID::Arena* arena, bool is_message_owned);
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(AgentStateUpdate* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "Schola.AgentStateUpdate";
  }
  protected:
  explicit AgentStateUpdate(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kActionsFieldNumber = 1,
  };
  // .Schola.DictPoint actions = 1;
  bool has_actions() const;
  private:
  bool _internal_has_actions() const;
  public:
  void clear_actions();
  const ::Schola::DictPoint& actions() const;
  PROTOBUF_NODISCARD ::Schola::DictPoint* release_actions();
  ::Schola::DictPoint* mutable_actions();
  void set_allocated_actions(::Schola::DictPoint* actions);
  private:
  const ::Schola::DictPoint& _internal_actions() const;
  ::Schola::DictPoint* _internal_mutable_actions();
  public:
  void unsafe_arena_set_allocated_actions(
      ::Schola::DictPoint* actions);
  ::Schola::DictPoint* unsafe_arena_release_actions();

  // @@protoc_insertion_point(class_scope:Schola.AgentStateUpdate)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::Schola::DictPoint* actions_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_StateUpdates_2eproto;
};
// -------------------------------------------------------------------

class EnvironmentStep_UpdatesEntry_DoNotUse : public ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<EnvironmentStep_UpdatesEntry_DoNotUse, 
    int32_t, ::Schola::AgentStateUpdate,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_MESSAGE> {
public:
  typedef ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<EnvironmentStep_UpdatesEntry_DoNotUse, 
    int32_t, ::Schola::AgentStateUpdate,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_MESSAGE> SuperType;
  EnvironmentStep_UpdatesEntry_DoNotUse();
  explicit PROTOBUF_CONSTEXPR EnvironmentStep_UpdatesEntry_DoNotUse(
      ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);
  explicit EnvironmentStep_UpdatesEntry_DoNotUse(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  void MergeFrom(const EnvironmentStep_UpdatesEntry_DoNotUse& other);
  static const EnvironmentStep_UpdatesEntry_DoNotUse* internal_default_instance() { return reinterpret_cast<const EnvironmentStep_UpdatesEntry_DoNotUse*>(&_EnvironmentStep_UpdatesEntry_DoNotUse_default_instance_); }
  static bool ValidateKey(void*) { return true; }
  static bool ValidateValue(void*) { return true; }
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  friend struct ::TableStruct_StateUpdates_2eproto;
};

// -------------------------------------------------------------------

class EnvironmentStep final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:Schola.EnvironmentStep) */ {
 public:
  inline EnvironmentStep() : EnvironmentStep(nullptr) {}
  ~EnvironmentStep() override;
  explicit PROTOBUF_CONSTEXPR EnvironmentStep(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  EnvironmentStep(const EnvironmentStep& from);
  EnvironmentStep(EnvironmentStep&& from) noexcept
    : EnvironmentStep() {
    *this = ::std::move(from);
  }

  inline EnvironmentStep& operator=(const EnvironmentStep& from) {
    CopyFrom(from);
    return *this;
  }
  inline EnvironmentStep& operator=(EnvironmentStep&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const EnvironmentStep& default_instance() {
    return *internal_default_instance();
  }
  static inline const EnvironmentStep* internal_default_instance() {
    return reinterpret_cast<const EnvironmentStep*>(
               &_EnvironmentStep_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  friend void swap(EnvironmentStep& a, EnvironmentStep& b) {
    a.Swap(&b);
  }
  inline void Swap(EnvironmentStep* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(EnvironmentStep* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  EnvironmentStep* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<EnvironmentStep>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const EnvironmentStep& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const EnvironmentStep& from) {
    EnvironmentStep::MergeImpl(*this, from);
  }
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::PROTOBUF_NAMESPACE_ID::Arena* arena, bool is_message_owned);
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(EnvironmentStep* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "Schola.EnvironmentStep";
  }
  protected:
  explicit EnvironmentStep(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------


  // accessors -------------------------------------------------------

  enum : int {
    kUpdatesFieldNumber = 1,
  };
  // map<int32, .Schola.AgentStateUpdate> updates = 1;
  int updates_size() const;
  private:
  int _internal_updates_size() const;
  public:
  void clear_updates();
  private:
  const ::PROTOBUF_NAMESPACE_ID::Map< int32_t, ::Schola::AgentStateUpdate >&
      _internal_updates() const;
  ::PROTOBUF_NAMESPACE_ID::Map< int32_t, ::Schola::AgentStateUpdate >*
      _internal_mutable_updates();
  public:
  const ::PROTOBUF_NAMESPACE_ID::Map< int32_t, ::Schola::AgentStateUpdate >&
      updates() const;
  ::PROTOBUF_NAMESPACE_ID::Map< int32_t, ::Schola::AgentStateUpdate >*
      mutable_updates();

  // @@protoc_insertion_point(class_scope:Schola.EnvironmentStep)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::internal::MapField<
        EnvironmentStep_UpdatesEntry_DoNotUse,
        int32_t, ::Schola::AgentStateUpdate,
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32,
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_MESSAGE> updates_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_StateUpdates_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// AgentStateUpdate

// .Schola.DictPoint actions = 1;
inline bool AgentStateUpdate::_internal_has_actions() const {
  return this != internal_default_instance() && _impl_.actions_ != nullptr;
}
inline bool AgentStateUpdate::has_actions() const {
  return _internal_has_actions();
}
inline const ::Schola::DictPoint& AgentStateUpdate::_internal_actions() const {
  const ::Schola::DictPoint* p = _impl_.actions_;
  return p != nullptr ? *p : reinterpret_cast<const ::Schola::DictPoint&>(
      ::Schola::_DictPoint_default_instance_);
}
inline const ::Schola::DictPoint& AgentStateUpdate::actions() const {
  // @@protoc_insertion_point(field_get:Schola.AgentStateUpdate.actions)
  return _internal_actions();
}
inline void AgentStateUpdate::unsafe_arena_set_allocated_actions(
    ::Schola::DictPoint* actions) {
  if (GetArenaForAllocation() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(_impl_.actions_);
  }
  _impl_.actions_ = actions;
  if (actions) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:Schola.AgentStateUpdate.actions)
}
inline ::Schola::DictPoint* AgentStateUpdate::release_actions() {
  
  ::Schola::DictPoint* temp = _impl_.actions_;
  _impl_.actions_ = nullptr;
#ifdef PROTOBUF_FORCE_COPY_IN_RELEASE
  auto* old =  reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(temp);
  temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  if (GetArenaForAllocation() == nullptr) { delete old; }
#else  // PROTOBUF_FORCE_COPY_IN_RELEASE
  if (GetArenaForAllocation() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
#endif  // !PROTOBUF_FORCE_COPY_IN_RELEASE
  return temp;
}
inline ::Schola::DictPoint* AgentStateUpdate::unsafe_arena_release_actions() {
  // @@protoc_insertion_point(field_release:Schola.AgentStateUpdate.actions)
  
  ::Schola::DictPoint* temp = _impl_.actions_;
  _impl_.actions_ = nullptr;
  return temp;
}
inline ::Schola::DictPoint* AgentStateUpdate::_internal_mutable_actions() {
  
  if (_impl_.actions_ == nullptr) {
    auto* p = CreateMaybeMessage<::Schola::DictPoint>(GetArenaForAllocation());
    _impl_.actions_ = p;
  }
  return _impl_.actions_;
}
inline ::Schola::DictPoint* AgentStateUpdate::mutable_actions() {
  ::Schola::DictPoint* _msg = _internal_mutable_actions();
  // @@protoc_insertion_point(field_mutable:Schola.AgentStateUpdate.actions)
  return _msg;
}
inline void AgentStateUpdate::set_allocated_actions(::Schola::DictPoint* actions) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(_impl_.actions_);
  }
  if (actions) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalGetOwningArena(
                reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(actions));
    if (message_arena != submessage_arena) {
      actions = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, actions, submessage_arena);
    }
    
  } else {
    
  }
  _impl_.actions_ = actions;
  // @@protoc_insertion_point(field_set_allocated:Schola.AgentStateUpdate.actions)
}

// -------------------------------------------------------------------

// -------------------------------------------------------------------

// EnvironmentStep

// map<int32, .Schola.AgentStateUpdate> updates = 1;
inline int EnvironmentStep::_internal_updates_size() const {
  return _impl_.updates_.size();
}
inline int EnvironmentStep::updates_size() const {
  return _internal_updates_size();
}
inline void EnvironmentStep::clear_updates() {
  _impl_.updates_.Clear();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< int32_t, ::Schola::AgentStateUpdate >&
EnvironmentStep::_internal_updates() const {
  return _impl_.updates_.GetMap();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< int32_t, ::Schola::AgentStateUpdate >&
EnvironmentStep::updates() const {
  // @@protoc_insertion_point(field_map:Schola.EnvironmentStep.updates)
  return _internal_updates();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< int32_t, ::Schola::AgentStateUpdate >*
EnvironmentStep::_internal_mutable_updates() {
  return _impl_.updates_.MutableMap();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< int32_t, ::Schola::AgentStateUpdate >*
EnvironmentStep::mutable_updates() {
  // @@protoc_insertion_point(field_mutable_map:Schola.EnvironmentStep.updates)
  return _internal_mutable_updates();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace Schola

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_StateUpdates_2eproto
