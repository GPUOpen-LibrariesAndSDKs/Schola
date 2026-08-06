# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Unreal Editor Python: dump world actors to JSON for Schola Fabrica.

Run inside the Editor with -ExecutePythonScript=...; set env FABRICA_SNAPSHOT_OUT to a file path.
Optional: FABRICA_MAX_ACTORS (int), FABRICA_CLASS_FILTER (comma-separated substrings).
"""

import json
import os

try:
    import unreal  # type: ignore
except ImportError:
    unreal = None  # noqa: N816 — module name matches UE API


def _actor_class_chain(actor) -> list[str]:
    names: list[str] = []
    cls = actor.get_class()
    depth = 0
    while cls and depth < 32:
        names.append(cls.get_name())
        parent = None
        for getter in ("get_super_class", "get_super_struct"):
            if hasattr(cls, getter):
                try:
                    parent = getattr(cls, getter)()
                except Exception:
                    parent = None
                if parent:
                    break
        cls = parent
        depth += 1
    return names


def _components(actor) -> list[dict]:
    out: list[dict] = []
    try:
        for comp in actor.get_components_by_class(unreal.ActorComponent):
            out.append(
                {
                    "class": comp.get_class().get_name(),
                    "name": comp.get_name(),
                }
            )
    except Exception:
        pass
    return out


def _serialize_actor(actor) -> dict:
    loc = [0.0, 0.0, 0.0]
    rot = [0.0, 0.0, 0.0]
    scl = [1.0, 1.0, 1.0]
    try:
        loc_v = actor.get_actor_location()
        loc = [loc_v.x, loc_v.y, loc_v.z]
    except Exception:
        pass
    try:
        rot_v = actor.get_actor_rotation()
        rot = [rot_v.pitch, rot_v.yaw, rot_v.roll]
    except Exception:
        pass
    try:
        scl_v = actor.get_actor_scale3d()
        scl = [scl_v.x, scl_v.y, scl_v.z]
    except Exception:
        pass
    try:
        tags = [str(t) for t in actor.tags]
    except Exception:
        tags = []
    return {
        "label": actor.get_actor_label(),
        "path": actor.get_path_name(),
        "parent_classes": _actor_class_chain(actor),
        "location": loc,
        "rotation": rot,
        "scale": scl,
        "tags": tags,
        "components": _components(actor),
    }


def _level_path(world) -> str:
    if not world:
        return ""
    try:
        level = world.get_persistent_level()
        if level:
            return level.get_path_name()
    except AttributeError:
        pass
    try:
        level = world.get_editor_property("persistent_level")
        if level:
            return level.get_path_name()
    except Exception:
        pass
    return ""


def _main():
    if unreal is None:
        raise RuntimeError("This script must run inside Unreal Editor Python.")
    out_path = os.environ.get("FABRICA_SNAPSHOT_OUT")
    if not out_path:
        raise RuntimeError("Set FABRICA_SNAPSHOT_OUT to an output JSON path.")
    max_actors = int(os.environ.get("FABRICA_MAX_ACTORS", "500"))
    filters = [
        s.strip().lower()
        for s in os.environ.get("FABRICA_CLASS_FILTER", "").split(",")
        if s.strip()
    ]

    # exclude actors that are not relevant to the environment
    exclude = [
        x.lower()
        for x in [
            "WorldPartitionHLOD",
            "LandscapeStreamingProxy",
            "WorldPartitionMiniMap",
            "WorldDataLayers",
        ]
    ]

    editor_subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    if not editor_subsystem:
        raise RuntimeError("UnrealEditorSubsystem unavailable.")
    world = editor_subsystem.get_editor_world()
    if not world:
        raise RuntimeError("No editor world available.")

    level_path = _level_path(world)
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not actor_subsystem:
        raise RuntimeError("EditorActorSubsystem unavailable.")
    actors = list(actor_subsystem.get_all_level_actors())
    if not level_path and actors:
        marker = ":PersistentLevel."
        for actor in actors:
            actor_path = actor.get_path_name()
            if marker in actor_path:
                level_path = actor_path.split(marker)[0] + ":PersistentLevel"
                break
    payload: list[dict] = []

    for actor in actors:
        if len(payload) >= max_actors:
            break
        cls_name = actor.get_class().get_name().lower()

        if cls_name in exclude or (filters and not any(f in cls_name for f in filters)):
            continue
        try:
            payload.append(_serialize_actor(actor))
        except Exception as exc:  # noqa: BLE001
            payload.append({"error": str(exc), "path": actor.get_path_name()})

    data = {
        "world": world.get_path_name() if world else "",
        "level": level_path,
        "actor_count": len(payload),
        "actors": payload,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    _main()
