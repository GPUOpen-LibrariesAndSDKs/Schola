"""Map, batched native transitions, and goal-distance fields for CampusCrowd navigation."""
import ctypes as ct
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELL_SIZE = .5
DT = .1
PACE = 1.4
HEADER = struct.Struct("<4sIIffff")


def held_out(origin, goal, count):
    """Reserve directed stop pairs, independent of the PPO seed."""
    return (origin * count + goal) % 5 == 0


def leg_steps(distance):
    # Allow twice the unobstructed travel time, plus time to start and arrive.
    return np.ceil((2 * distance / PACE + 8) / DT).astype(np.int32)


@dataclass
class NavigationMap:
    walls: np.ndarray
    origin: tuple[float, float]
    cell: float = CELL_SIZE
    bits: np.ndarray = field(init=False)

    def __post_init__(self):
        self.height, self.width = self.walls.shape
        self.scale = self.cell * np.hypot(self.width, self.height)
        self.bits = np.packbits(self.walls.ravel(), bitorder="little")

    @classmethod
    def load(cls, path):
        data = path.read_bytes()
        magic, width, height, cell, x, y, scale = HEADER.unpack_from(data)
        if (magic != b"CNV1" or width <= 0 or height <= 0 or
                not np.isfinite([cell, x, y, scale]).all() or cell <= 0 or scale <= 0):
            raise ValueError(f"Invalid navigation grid {path}")
        if len(data) != HEADER.size + (width * height + 7) // 8:
            raise ValueError(f"Invalid navigation grid size in {path}")
        cells = np.unpackbits(np.frombuffer(data, np.uint8, offset=HEADER.size), bitorder="little", count=width*height)
        result = cls(cells.reshape(height, width).astype(bool), (x, y), cell)
        result.scale = scale
        return result

    def save(self, path):
        path.write_bytes(HEADER.pack(b"CNV1", self.width, self.height, self.cell, *self.origin, self.scale)
                         + self.bits.tobytes())

    def cell_indices(self, positions):
        relative = np.asarray(positions, np.float32) - np.asarray(self.origin, np.float32)
        xy = np.floor(relative * np.float32(1 / self.cell)).astype(np.int32)
        return xy[..., 0].clip(0, self.width-1), xy[..., 1].clip(0, self.height-1)


class NativeNavigation:
    def __init__(self, nav_map, output):
        output.mkdir(parents=True, exist_ok=True)
        source = ROOT / "Tools/navigation_native.cpp"
        shared = ROOT / "Source/CampusCrowd/CampusNavigation.h"
        headers = [shared, shared.with_name("CampusShortcutRoute.h")]
        digest = hashlib.sha256(b"".join(path.read_bytes() for path in [source, *headers])).hexdigest()[:12]
        library = output / f"navigation-{digest}.so"
        if not library.exists():
            scratch = library.with_suffix(f".{os.getpid()}.so")
            subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-shared", "-fPIC",
                            str(source), "-o", str(scratch)], check=True)
            scratch.replace(library)
        self.lib = ct.CDLL(str(library.resolve()))
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        grid = [u8, ct.c_int, ct.c_int, ct.c_float, ct.c_float, ct.c_float]
        tail = [ct.c_float, ct.c_int, f32, f32, f32]
        self.lib.nav_step.argtypes = grid + tail + [u8]
        self.lib.nav_clear_segments.argtypes = grid + [ct.c_int, f32, f32, u8]
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.lib.nav_clear_segments.restype = None
        self.lib.nav_shortcut_times.argtypes = grid + [ct.c_int, f32, i32, f32]
        self.lib.nav_shortcut_step.argtypes = grid + [ct.c_float, ct.c_int, f32, f32, f32, i32, i32, i32, f32, u8]
        self.lib.nav_shortcut_times.restype = self.lib.nav_shortcut_step.restype = None
        self.lib.nav_distance.argtypes = grid + [ct.c_int, ct.c_int, f32]
        self.lib.nav_step.restype = self.lib.nav_distance.restype = None
        self.map = nav_map
        self.args = (nav_map.bits, nav_map.width, nav_map.height, nav_map.cell, *nav_map.origin)

    def step(self, states, goals, actions):
        result = np.empty(len(states), np.uint8)
        self.lib.nav_step(*self.args, DT, len(states), states, goals, np.ascontiguousarray(actions, np.float32), result)
        return result

    def clear_segments(self, starts, ends):
        starts = np.ascontiguousarray(starts, np.float32).reshape(-1, 2)
        ends = np.ascontiguousarray(ends, np.float32).reshape(-1, 2)
        if len(starts) != len(ends):
            raise ValueError("Segment starts and ends must have the same length")
        result = np.empty(len(starts), np.uint8)
        self.lib.nav_clear_segments(*self.args, len(starts), starts, ends, result)
        return result.astype(bool)

    def shortcut_times(self, points, offsets):
        output = np.empty(len(points), np.float32)
        self.lib.nav_shortcut_times(*self.args, len(offsets) - 1, points, offsets, output)
        return output

    def shortcut_step(self, states, points, times, offsets, route_ids, cursors, replan_in, dt=DT):
        result = np.empty(len(states), np.uint8)
        self.lib.nav_shortcut_step(*self.args, dt, len(states), states, points, times, offsets,
                                   route_ids, cursors, replan_in, result)
        return result

    def field(self, goal):
        x, y = self.map.cell_indices(goal)
        output = np.empty(self.map.walls.shape, np.float32)
        self.lib.nav_distance(*self.args, int(x), int(y), output)
        return output


def campus_map():
    content = ROOT / "Content/Campus"
    nav = NavigationMap.load(content / "navigation_grid.bin")
    metadata = json.loads((content / "navigation_stops.json").read_text())
    goals = np.asarray([stop["position"] for stop in metadata["stops"]], np.float32)
    return nav, goals, metadata["gate_count"]


def toy_map(rectangle=True):
    walls = np.zeros((28, 40), dtype=bool)
    if rectangle:
        walls[9:19, 15:25] = True
    stops = np.array([[2.75, 7.25], [17.25, 7.25], [2.75, 5.25], [17.25, 9.25]], np.float32)
    return NavigationMap(walls, (0, 0)), stops, 2


class GoalDistances:
    def __init__(self, native, goals, output):
        self.native = native
        checksum = hashlib.sha256(native.map.bits.tobytes() + goals.tobytes()
                                  + repr((native.map.origin, native.map.walls.shape, native.map.cell)).encode()
                                  + (ROOT / "Source/CampusCrowd/CampusNavigation.h").read_bytes()
                                  + (ROOT / "Tools/navigation_native.cpp").read_bytes().split(b"void nav_distance(", 1)[1]).hexdigest()[:12]
        self.fields = []
        for i, goal in enumerate(goals):
            path = output / f"distance-{checksum}-{i}.npy"
            if not path.exists():
                tmp = path.with_suffix(f".{os.getpid()}.npy")
                np.save(tmp, native.field(goal))
                tmp.replace(path)
            self.fields.append(np.load(path, mmap_mode="r"))

    def at(self, positions, goal_ids):
        nav = self.native.map
        ix, iy = nav.cell_indices(positions)
        output = np.empty(len(positions), np.float32)
        fractional = (positions - np.asarray(nav.origin, np.float32)) / nav.cell - .5
        left, top = np.floor(fractional).astype(np.int32).T
        wx, wy = (fractional - np.floor(fractional)).T
        for i in np.unique(goal_ids):
            mask = goal_ids == i
            field = self.fields[i]
            centre = field[iy[mask], ix[mask]]
            result = np.zeros_like(centre)
            for dx, dy, weight in ((0, 0, (1-wx)*(1-wy)), (1, 0, wx*(1-wy)), (0, 1, (1-wx)*wy), (1, 1, wx*wy)):
                x = (left[mask]+dx).clip(0, nav.width-1)
                y = (top[mask]+dy).clip(0, nav.height-1)
                sample = field[y, x]
                result += np.where(np.isfinite(sample), sample, centre) * weight[mask]
            output[mask] = result
        return output
