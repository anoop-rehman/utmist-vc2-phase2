"""Parser for UTMIST .creature files (Unity C# BinaryFormatter / MS-NRBF).

The files contain a Karl-Sims-style `CreatureGenotype` object graph:
CreatureGenotype{name, stage, segments: List[SegmentGenotype], ...}
SegmentGenotype{connections: List[SegmentConnectionGenotype],
                neurons: List[NeuronGenotype], recursiveLimit,
                dimensionX/Y/Z, jointType, ...}

This module implements the subset of MS-NRBF ([MS-NRBF] spec) that
BinaryFormatter emits for such graphs, and returns plain Python dicts/lists.

Usage:
    from rower_soccer.tools.creature_format import load_creature
    genotype = load_creature("creature_configs/3_SEG_WORM.creature")
"""

import struct
from io import BytesIO

# --- record type constants -------------------------------------------------
RT_SERIALIZED_STREAM_HEADER = 0x00
RT_CLASS_WITH_ID = 0x01
RT_SYSTEM_CLASS_WITH_MEMBERS_AND_TYPES = 0x04
RT_CLASS_WITH_MEMBERS_AND_TYPES = 0x05
RT_BINARY_OBJECT_STRING = 0x06
RT_BINARY_ARRAY = 0x07
RT_MEMBER_PRIMITIVE_TYPED = 0x08
RT_MEMBER_REFERENCE = 0x09
RT_OBJECT_NULL = 0x0A
RT_MESSAGE_END = 0x0B
RT_BINARY_LIBRARY = 0x0C
RT_OBJECT_NULL_MULTIPLE_256 = 0x0D
RT_OBJECT_NULL_MULTIPLE = 0x0E
RT_ARRAY_SINGLE_PRIMITIVE = 0x0F
RT_ARRAY_SINGLE_OBJECT = 0x10
RT_ARRAY_SINGLE_STRING = 0x11

# BinaryTypeEnum
BT_PRIMITIVE = 0
BT_STRING = 1
BT_OBJECT = 2
BT_SYSTEM_CLASS = 3
BT_CLASS = 4
BT_OBJECT_ARRAY = 5
BT_STRING_ARRAY = 6
BT_PRIMITIVE_ARRAY = 7

# PrimitiveTypeEnum -> (struct fmt, size)
PRIMITIVES = {
    1: ("?", 1),   # Boolean
    2: ("B", 1),   # Byte
    3: ("c", 1),   # Char (approx)
    5: (None, 16), # Decimal (unsupported)
    6: ("<d", 8),  # Double
    7: ("<h", 2),  # Int16
    8: ("<i", 4),  # Int32
    9: ("<q", 8),  # Int64
    10: ("b", 1),  # SByte
    11: ("<f", 4), # Single
    12: ("<q", 8), # TimeSpan
    13: ("<q", 8), # DateTime
    14: ("<H", 2), # UInt16
    15: ("<I", 4), # UInt32
    16: ("<Q", 8), # UInt64
    17: (None, 0), # Null
    18: (None, 0), # String
}


class Ref:
    """Placeholder for a MemberReference, resolved in a second pass."""
    __slots__ = ("id",)

    def __init__(self, id_):
        self.id = id_

    def __repr__(self):
        return f"Ref({self.id})"


class NRBFReader:
    def __init__(self, data: bytes):
        self.f = BytesIO(data)
        self.objects = {}       # objectId -> value
        self.class_meta = {}    # objectId of defining record -> (name, members, types, infos)
        self.root_id = None

    # --- low-level readers ---
    def _u8(self):
        return self.f.read(1)[0]

    def _i32(self):
        return struct.unpack("<i", self.f.read(4))[0]

    def _string(self):
        # LengthPrefixedString: 7-bit encoded length
        length = 0
        shift = 0
        while True:
            b = self._u8()
            length |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        return self.f.read(length).decode("utf-8")

    def _primitive(self, ptype):
        fmt, size = PRIMITIVES[ptype]
        if fmt is None:
            raise NotImplementedError(f"primitive type {ptype}")
        raw = self.f.read(size)
        if fmt == "?":
            return raw[0] != 0
        if fmt == "c":
            return raw.decode("latin1")
        return struct.unpack(fmt, raw)[0]

    # --- class metadata ---
    def _read_class_info(self):
        object_id = self._i32()
        name = self._string()
        member_count = self._i32()
        members = [self._string() for _ in range(member_count)]
        return object_id, name, members

    def _read_member_type_info(self, count):
        types = [self._u8() for _ in range(count)]
        infos = []
        for t in types:
            if t == BT_PRIMITIVE or t == BT_PRIMITIVE_ARRAY:
                infos.append(self._u8())
            elif t == BT_SYSTEM_CLASS:
                infos.append(self._string())
            elif t == BT_CLASS:
                infos.append((self._string(), self._i32()))
            else:
                infos.append(None)
        return types, infos

    def _read_class_values(self, object_id, name, members, types, infos):
        obj = {"__class__": name}
        self.objects[object_id] = obj  # register before reading (cycles)
        for member, t, info in zip(members, types, infos):
            if t == BT_PRIMITIVE:
                obj[member] = self._primitive(info)
            else:
                obj[member] = self._read_record()
        return obj

    # --- records ---
    def _read_record(self, record_type=None):
        rt = self._u8() if record_type is None else record_type
        if rt == RT_SERIALIZED_STREAM_HEADER:
            self.root_id = self._i32()
            self.f.read(12)  # headerId, major, minor
            return None
        if rt == RT_BINARY_LIBRARY:
            self._i32()
            self._string()
            return self._read_record()
        if rt == RT_CLASS_WITH_MEMBERS_AND_TYPES:
            object_id, name, members = self._read_class_info()
            types, infos = self._read_member_type_info(len(members))
            self._i32()  # library id
            self.class_meta[object_id] = (name, members, types, infos)
            return self._read_class_values(object_id, name, members, types, infos)
        if rt == RT_SYSTEM_CLASS_WITH_MEMBERS_AND_TYPES:
            object_id, name, members = self._read_class_info()
            types, infos = self._read_member_type_info(len(members))
            self.class_meta[object_id] = (name, members, types, infos)
            return self._read_class_values(object_id, name, members, types, infos)
        if rt == RT_CLASS_WITH_ID:
            object_id = self._i32()
            metadata_id = self._i32()
            name, members, types, infos = self.class_meta[metadata_id]
            return self._read_class_values(object_id, name, members, types, infos)
        if rt == RT_BINARY_OBJECT_STRING:
            object_id = self._i32()
            s = self._string()
            self.objects[object_id] = s
            return s
        if rt == RT_MEMBER_REFERENCE:
            return Ref(self._i32())
        if rt == RT_OBJECT_NULL:
            return None
        if rt == RT_OBJECT_NULL_MULTIPLE_256:
            return [None] * self._u8()
        if rt == RT_OBJECT_NULL_MULTIPLE:
            return [None] * self._i32()
        if rt == RT_ARRAY_SINGLE_PRIMITIVE:
            object_id = self._i32()
            length = self._i32()
            ptype = self._u8()
            arr = [self._primitive(ptype) for _ in range(length)]
            self.objects[object_id] = arr
            return arr
        if rt == RT_ARRAY_SINGLE_OBJECT or rt == RT_ARRAY_SINGLE_STRING:
            object_id = self._i32()
            length = self._i32()
            arr = []
            self.objects[object_id] = arr
            while len(arr) < length:
                v = self._read_record()
                if isinstance(v, list) and v and all(x is None for x in v):
                    arr.extend(v)  # null-multiple burst
                else:
                    arr.append(v)
            return arr
        if rt == RT_BINARY_ARRAY:
            object_id = self._i32()
            array_type = self._u8()
            rank = self._i32()
            lengths = [self._i32() for _ in range(rank)]
            if array_type in (3, 4, 5):  # offset variants
                [self._i32() for _ in range(rank)]
            bt = self._u8()
            if bt == BT_PRIMITIVE or bt == BT_PRIMITIVE_ARRAY:
                self._u8()
            elif bt == BT_SYSTEM_CLASS:
                self._string()
            elif bt == BT_CLASS:
                self._string(); self._i32()
            total = 1
            for L in lengths:
                total *= L
            arr = []
            self.objects[object_id] = arr
            while len(arr) < total:
                v = self._read_record()
                if isinstance(v, list) and v and all(x is None for x in v):
                    arr.extend(v)
                else:
                    arr.append(v)
            return arr
        if rt == RT_MEMBER_PRIMITIVE_TYPED:
            ptype = self._u8()
            return self._primitive(ptype)
        if rt == RT_MESSAGE_END:
            raise EOFError
        raise NotImplementedError(f"record type 0x{rt:02X} at offset {self.f.tell()-1}")

    def parse(self):
        root = None
        try:
            while True:
                v = self._read_record()
                if root is None and isinstance(v, dict):
                    root = v
        except EOFError:
            pass
        return self._resolve(root, seen=set())

    def _resolve(self, v, seen):
        if isinstance(v, Ref):
            return self._resolve(self.objects.get(v.id), seen)
        if isinstance(v, dict):
            key = id(v)
            if key in seen:
                return v
            seen.add(key)
            for k in list(v.keys()):
                v[k] = self._resolve(v[k], seen)
            return v
        if isinstance(v, list):
            key = id(v)
            if key in seen:
                return v
            seen.add(key)
            for i in range(len(v)):
                v[i] = self._resolve(v[i], seen)
            return v
        return v


def simplify(obj):
    """Collapse .NET List wrappers and enums into plain Python values."""
    if isinstance(obj, dict):
        name = obj.get("__class__", "")
        if name.startswith("System.Collections.Generic.List"):
            items = obj.get("_items") or []
            size = obj.get("_size", len(items))
            return [simplify(x) for x in items[:size]]
        if set(obj.keys()) == {"__class__", "value__"}:  # enum
            return obj["value__"]
        return {k: simplify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [simplify(x) for x in obj]
    return obj


def load_creature(path):
    with open(path, "rb") as f:
        data = f.read()
    return simplify(NRBFReader(data).parse())


if __name__ == "__main__":
    import json
    import sys

    genotype = load_creature(sys.argv[1])
    print(json.dumps(genotype, indent=1, default=str)[:6000])
