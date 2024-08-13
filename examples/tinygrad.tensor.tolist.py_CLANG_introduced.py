# CLANG=1 python pyIntroducer.py examples/tinygrad.tensor.tolist.py
from tinygrad.tensor import Tensor                                                                                                                                                                       # ...s/tinygrad.tensor.tolist.py:1
from tinygrad.tensor import Tensor                            # noqa: F401                                                                                                                               # __init__.py:1
from __future__ import annotations                                                                                                                                                                       # tensor.py:2
import dataclasses                                                                                                                                                                                       # tensor.py:3
import time, math, itertools, functools, struct, sys, inspect                                                                                                                                            # tensor.py:4
from contextlib import ContextDecorator                                                                                                                                                                  # tensor.py:5
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Dict, DefaultDict, cast, get_args, Set                                                                              # tensor.py:6
from collections import defaultdict                                                                                                                                                                      # tensor.py:7
import numpy as np                                                                                                                                                                                       # tensor.py:8
from tinygrad.dtype import DType, DTypeLike, dtypes, ImageDType, ConstType, least_upper_float, least_upper_dtype, sum_acc_dtype, to_dtype                                                                # tensor.py:10
from typing import Final, Optional, ClassVar, Set, Tuple, Dict, Union                                                                                                                                    # dtype.py:1
from dataclasses import dataclass                                                                                                                                                                        # dtype.py:2
import functools                                                                                                                                                                                         # dtype.py:3
from tinygrad.helpers import getenv                                                                                                                                                                      # dtype.py:4
from __future__ import annotations                                                                                                                                                                       # helpers.py:1
import os, functools, platform, time, re, contextlib, operator, hashlib, pickle, sqlite3, cProfile, pstats, tempfile, pathlib, string, ctypes, sys                                                       # helpers.py:2
import itertools, urllib.request, subprocess, shutil, math, json, contextvars                                                                                                                            # helpers.py:3
from dataclasses import dataclass                                                                                                                                                                        # helpers.py:4
from typing import Dict, Tuple, Union, List, ClassVar, Optional, Iterable, Any, TypeVar, TYPE_CHECKING, Callable, Sequence                                                                               # helpers.py:5
if TYPE_CHECKING:  # TODO: remove this and import TypeGuard from typing once minimum python supported version is 3.10                                                                                    # helpers.py:6
T = TypeVar("T")                                                                                                                                                                                         # helpers.py:10
U = TypeVar("U")                                                                                                                                                                                         # helpers.py:11
OSX = platform.system() == "Darwin"                                                                                                                                                                      # helpers.py:16
CI = os.getenv("CI", "") != ""                                                                                                                                                                           # helpers.py:17

class Context(contextlib.ContextDecorator):                                                                                                                                                              # helpers.py:82
  stack: ClassVar[List[dict[str, int]]] = [{}]                                                                                                                                                           # helpers.py:83

class ContextVar:                                                                                                                                                                                        # helpers.py:92
  _cache: ClassVar[Dict[str, ContextVar]] = {}                                                                                                                                                           # helpers.py:93
  value: int                                                                                                                                                                                             # helpers.py:94
  key: str                                                                                                                                                                                               # helpers.py:95

DEBUG, IMAGE, BEAM, NOOPT, JIT = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0), ContextVar("BEAM", 0), ContextVar("NOOPT", 0), ContextVar("JIT", 1)                                                     # helpers.py:106

  class ContextVar:                                                                                                                                                                                      # helpers.py:92
    def __new__(cls, key, default_value):                                                                                                                                                                # helpers.py:96
      if key in ContextVar._cache: return ContextVar._cache[key]                                                                                                                                         # helpers.py:97
      instance = ContextVar._cache[key] = super().__new__(cls)                                                                                                                                           # helpers.py:98
      instance.value, instance.key = getenv(key, default_value), key                                                                                                                                     # helpers.py:99

        @functools.lru_cache(maxsize=None)                                                                                                                                                               # helpers.py:79
        def getenv(key:str, default=0): return type(default)(os.getenv(key, default))

      return instance                                                                                                                                                                                    # helpers.py:100

WINO, THREEFRY, CAPTURING, TRACEMETA = ContextVar("WINO", 0), ContextVar("THREEFRY", 0), ContextVar("CAPTURING", 1), ContextVar("TRACEMETA", 1)                                                          # helpers.py:107

GRAPH, GRAPHPATH, SAVE_SCHEDULE, RING = ContextVar("GRAPH", 0), getenv("GRAPHPATH", "/tmp/net"), ContextVar("SAVE_SCHEDULE", 0), ContextVar("RING", 1)                                                   # helpers.py:108

MULTIOUTPUT, PROFILE, PROFILEPATH = ContextVar("MULTIOUTPUT", 1), ContextVar("PROFILE", 0), ContextVar("PROFILEPATH", temp("tinygrad_profile.json"))                                                     # helpers.py:109

  def temp(x:str) -> str: return (pathlib.Path(tempfile.gettempdir()) / x).as_posix()                                                                                                                    # helpers.py:80

USE_TC, TC_OPT, TRANSCENDENTAL = ContextVar("TC", 1), ContextVar("TC_OPT", 0), ContextVar("TRANSCENDENTAL", 1)                                                                                           # helpers.py:110

FUSE_ARANGE, FUSE_CONV_BW = ContextVar("FUSE_ARANGE", 0), ContextVar("FUSE_CONV_BW", 0)                                                                                                                  # helpers.py:111

SPLIT_REDUCEOP, ARANGE_DIFF = ContextVar("SPLIT_REDUCEOP", 1), ContextVar("ARANGE_DIFF", 0)                                                                                                              # helpers.py:112

@dataclass(frozen=True)                                                                                                                                                                                  # helpers.py:115
class Metadata:
  name: str                                                                                                                                                                                              # helpers.py:116
  caller: str                                                                                                                                                                                            # helpers.py:117
  backward: bool = False                                                                                                                                                                                 # helpers.py:118

_METADATA: contextvars.ContextVar[Optional[Metadata]] = contextvars.ContextVar("_METADATA", default=None)                                                                                                # helpers.py:122

class GlobalCounters:                                                                                                                                                                                    # helpers.py:126
  global_ops: ClassVar[int] = 0                                                                                                                                                                          # helpers.py:127
  global_mem: ClassVar[int] = 0                                                                                                                                                                          # helpers.py:128
  time_sum_s: ClassVar[float] = 0.0                                                                                                                                                                      # helpers.py:129
  kernel_count: ClassVar[int] = 0                                                                                                                                                                        # helpers.py:130
  mem_used: ClassVar[int] = 0   # NOTE: this is not reset                                                                                                                                                # helpers.py:131

class ProfileLogger:                                                                                                                                                                                     # helpers.py:163
  writers: int = 0                                                                                                                                                                                       # helpers.py:164
  mjson: List[Dict] = []                                                                                                                                                                                 # helpers.py:165
  actors: Dict[Union[str, Tuple[str, str]], int] = {}                                                                                                                                                    # helpers.py:166

_cache_dir: str = getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache"))                                                                                                # helpers.py:203

CACHEDB: str = getenv("CACHEDB", os.path.abspath(os.path.join(_cache_dir, "tinygrad", "cache.db")))                                                                                                      # helpers.py:204

CACHELEVEL = getenv("CACHELEVEL", 2)                                                                                                                                                                     # helpers.py:205

VERSION = 16                                                                                                                                                                                             # helpers.py:207
_db_connection = None                                                                                                                                                                                    # helpers.py:208
_db_tables = set()                                                                                                                                                                                       # helpers.py:237

ConstType = Union[float, int, bool]                                                                                                                                                                      # dtype.py:6

@dataclass(frozen=True, order=True)                                                                                                                                                                      # dtype.py:9
class DType:
  priority: int  # this determines when things get upcasted                                                                                                                                              # dtype.py:10
  itemsize: int                                                                                                                                                                                          # dtype.py:11
  name: str                                                                                                                                                                                              # dtype.py:12
  fmt: Optional[str]                                                                                                                                                                                     # dtype.py:13
  count: int                                                                                                                                                                                             # dtype.py:14

# dependent typing?                                                                                                                                                                                      # dtype.py:23
@dataclass(frozen=True, repr=False)
class ImageDType(DType):
  shape: Tuple[int, ...]   # arbitrary arg for the dtype, used in image for the shape                                                                                                                    # dtype.py:24
  base: DType                                                                                                                                                                                            # dtype.py:25

class dtypes:                                                                                                                                                                                            # dtype.py:38
  bigint: Final[DType] = DType(-1, 0, "bigint", None, 1)   # arbitrary precision integer                                                                                                                 # dtype.py:65
  bool: Final[DType] = DType(0, 1, "bool", '?', 1)                                                                                                                                                       # dtype.py:66
  int8: Final[DType] = DType(1, 1, "char", 'b', 1)                                                                                                                                                       # dtype.py:67
  uint8: Final[DType] = DType(2, 1, "unsigned char", 'B', 1)                                                                                                                                             # dtype.py:68
  int16: Final[DType] = DType(3, 2, "short", 'h', 1)                                                                                                                                                     # dtype.py:69
  uint16: Final[DType] = DType(4, 2, "unsigned short", 'H', 1)                                                                                                                                           # dtype.py:70
  int32: Final[DType] = DType(5, 4, "int", 'i', 1)                                                                                                                                                       # dtype.py:71
  uint32: Final[DType] = DType(6, 4, "unsigned int", 'I', 1)                                                                                                                                             # dtype.py:72
  int64: Final[DType] = DType(7, 8, "long", 'l', 1)                                                                                                                                                      # dtype.py:73
  uint64: Final[DType] = DType(8, 8, "unsigned long", 'L', 1)                                                                                                                                            # dtype.py:74
  float16: Final[DType] = DType(9, 2, "half", 'e', 1)                                                                                                                                                    # dtype.py:75
  bfloat16: Final[DType] = DType(10, 2, "__bf16", None, 1)                                                                                                                                               # dtype.py:77
  float32: Final[DType] = DType(11, 4, "float", 'f', 1)                                                                                                                                                  # dtype.py:78
  float64: Final[DType] = DType(12, 8, "double", 'd', 1)                                                                                                                                                 # dtype.py:79
  half = float16; float = float32; double = float64 # noqa: E702                                                                                                                                         # dtype.py:82
  uchar = uint8; ushort = uint16; uint = uint32; ulong = uint64 # noqa: E702                                                                                                                             # dtype.py:83
  char = int8; short = int16; int = int32; long = int64 # noqa: E702                                                                                                                                     # dtype.py:84
  default_float: ClassVar[DType] = float32                                                                                                                                                               # dtype.py:92
  default_int: ClassVar[DType] = int32                                                                                                                                                                   # dtype.py:93

if (env_default_float := getenv("DEFAULT_FLOAT", "")):                                                                                                                                                   # dtype.py:95

DTypeLike = Union[str, DType]                                                                                                                                                                            # dtype.py:99
promo_lattice = { dtypes.bool: [dtypes.int8, dtypes.uint8], dtypes.int8: [dtypes.int16], dtypes.int16: [dtypes.int32], dtypes.int32: [dtypes.int64],                                                     # dtype.py:104
  dtypes.int64: [dtypes.float16, dtypes.bfloat16], dtypes.uint8: [dtypes.int16, dtypes.uint16], dtypes.uint16: [dtypes.int32, dtypes.uint32],
  dtypes.uint32: [dtypes.int64, dtypes.uint64], dtypes.uint64: [dtypes.float16, dtypes.bfloat16],
  dtypes.float16: [dtypes.float32], dtypes.bfloat16: [dtypes.float32], dtypes.float32: [dtypes.float64], }
DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not (k.startswith(('__', 'default', 'bigint')) or v.__class__ is staticmethod)}                                                               # dtype.py:118
INVERSE_DTYPES_DICT = {v.name:k for k,v in DTYPES_DICT.items()}                                                                                                                                          # dtype.py:119
INVERSE_DTYPES_DICT['bigint'] = 'bigint'                                                                                                                                                                 # dtype.py:120
from tinygrad.helpers import argfix, make_pair, flatten, prod, all_int, round_up, merge_dicts, argsort, getenv, get_shape, fully_flatten, dedup                                                          # tensor.py:11
from tinygrad.helpers import IMAGE, DEBUG, WINO, THREEFRY, _METADATA, Metadata, TRACEMETA                                                                                                                # tensor.py:12
from tinygrad.lazy import LazyBuffer                                                                                                                                                                     # tensor.py:13
from __future__ import annotations                                                                                                                                                                       # lazy.py:1
from typing import Union, Optional, Any, Tuple, List, get_args                                                                                                                                           # lazy.py:2
from tinygrad.dtype import dtypes, DType, DTypeLike, ConstType, to_dtype                                                                                                                                 # lazy.py:3
from tinygrad.helpers import prod, getenv, all_int, all_same, DEBUG, _METADATA, Metadata, SPLIT_REDUCEOP                                                                                                 # lazy.py:4
from tinygrad.ops import MetaOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, Op, exec_alu, python_alu, reduce_st                                                                                        # lazy.py:5
from __future__ import annotations                                                                                                                                                                       # ops.py:1
from typing import Union, Tuple, Any, List, Dict, Callable                                                                                                                                               # ops.py:2
import functools, hashlib, math, operator, ctypes, struct                                                                                                                                                # ops.py:3
from enum import Enum, auto                                                                                                                                                                              # ops.py:4
from dataclasses import dataclass                                                                                                                                                                        # ops.py:5
from tinygrad.helpers import prod, dedup, pretty_print                                                                                                                                                   # ops.py:6
from tinygrad.dtype import dtypes, DType, ConstType                                                                                                                                                      # ops.py:7
from tinygrad.shape.symbolic import Variable, sint                                                                                                                                                       # ops.py:8
from __future__ import annotations                                                                                                                                                                       # shape/symbolic.py:1
import functools                                                                                                                                                                                         # shape/symbolic.py:2
from math import gcd                                                                                                                                                                                     # shape/symbolic.py:3
from tinygrad.helpers import partition                                                                                                                                                                   # shape/symbolic.py:4
from typing import List, Dict, Callable, Tuple, Type, Union, Optional, Any, Set, Mapping                                                                                                                 # shape/symbolic.py:5

class Node:                                                                                                                                                                                              # shape/symbolic.py:10
  b: Union[Node, int]                                                                                                                                                                                    # shape/symbolic.py:11
  min: int                                                                                                                                                                                               # shape/symbolic.py:12
  max: sint                                                                                                                                                                                              # shape/symbolic.py:13

sint = Union[int, Variable, MulNode, SumNode]                                                                                                                                                            # shape/symbolic.py:304
render_python: Dict[Type, Callable[..., str]] = {                                                                                                                                                        # shape/symbolic.py:312
  Variable: lambda self,ops,ctx: f"{self.expr}[{self.min}-{self.max}{'='+str(self.val) if self._val is not None else ''}]" if ctx == "DEBUG" \
    else (f"Variable('{self.expr}', {self.min}, {self.max})"+(f".bind({self.val})" if self._val is not None else '') if ctx == "REPR" \
    else f"{self.expr}"),
  NumNode: lambda self,ops,ctx: f"NumNode({self.b})" if ctx == "REPR" else f"{self.b}",
  MulNode: render_mulnode,
  DivNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}//{self.b})",
  ModNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}%{self.b})",
  LtNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}<{sym_render(self.b,ops,ctx)})",
  SumNode: lambda self,ops,ctx: f"({'+'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})",
  AndNode: lambda self,ops,ctx: f"({' and '.join(sorted([x.render(ops,ctx) for x in self.nodes]))})",
}
from tinygrad.shape.shapetracker import ShapeTracker                                                                                                                                                     # ops.py:9
from __future__ import annotations                                                                                                                                                                       # shape/shapetracker.py:2
from dataclasses import dataclass                                                                                                                                                                        # shape/shapetracker.py:3
from typing import Tuple, List, Optional, Dict, Set, Iterable, cast                                                                                                                                      # shape/shapetracker.py:4
from tinygrad.helpers import merge_dicts, getenv                                                                                                                                                         # shape/shapetracker.py:5
from tinygrad.shape.symbolic import Variable, MulNode, Node, SumNode, NumNode, sint                                                                                                                      # shape/shapetracker.py:6
from tinygrad.shape.view import View, strides_for_shape                                                                                                                                                  # shape/shapetracker.py:7
from __future__ import annotations                                                                                                                                                                       # shape/view.py:1
import functools, operator, itertools, math                                                                                                                                                              # shape/view.py:2
from dataclasses import dataclass                                                                                                                                                                        # shape/view.py:3
from typing import Tuple, List, Optional, Dict, Set, cast                                                                                                                                                # shape/view.py:4
from tinygrad.helpers import prod, all_int, argsort                                                                                                                                                      # shape/view.py:5
from tinygrad.shape.symbolic import Node, NumNode, Variable, sint, sym_infer, create_lt_node, create_ge_node                                                                                             # shape/view.py:6

@dataclass(frozen=True)                                                                                                                                                                                  # shape/view.py:85
class View:
  shape:Tuple[sint, ...]                                                                                                                                                                                 # shape/view.py:86
  strides:Tuple[sint, ...]                                                                                                                                                                               # shape/view.py:87
  offset:sint                                                                                                                                                                                            # shape/view.py:88
  mask:Optional[Tuple[Tuple[sint, sint], ...]]                                                                                                                                                           # shape/view.py:89
  contiguous:bool                                                                                                                                                                                        # shape/view.py:90

@dataclass(frozen=True)                                                                                                                                                                                  # shape/shapetracker.py:10
class ShapeTracker:
  views: Tuple[View, ...]                                                                                                                                                                                # shape/shapetracker.py:11

# these are the llops your accelerator must implement, along with toCpu                                                                                                                                  # ops.py:15
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: many GPUs don't have DIV, but UnaryOps.RECIP doesn't work for integer division
class UnaryOps(Enum):
  """A -> A (elementwise)"""                                                                                                                                                                             # ops.py:16
  EXP2 = auto(); LOG2 = auto(); CAST = auto(); BITCAST = auto(); SIN = auto(); SQRT = auto(); NEG = auto(); RECIP = auto() # noqa: E702                                                                  # ops.py:17

class BinaryOps(Enum):                                                                                                                                                                                   # ops.py:18
  """A + A -> A (elementwise)"""                                                                                                                                                                         # ops.py:19
  ADD = auto(); MUL = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPNE = auto(); XOR = auto() # noqa: E702                                                                       # ops.py:20
  SHL = auto(); SHR = auto(); OR = auto(); AND = auto(); THREEFRY = auto() # noqa: E702                                                                                                                  # ops.py:21

class TernaryOps(Enum):                                                                                                                                                                                  # ops.py:22
  """A + A + A -> A (elementwise)"""                                                                                                                                                                     # ops.py:23
  WHERE = auto(); MULACC = auto() # noqa: E702                                                                                                                                                           # ops.py:24

class ReduceOps(Enum):                                                                                                                                                                                   # ops.py:25
  """A -> B (reduce)"""                                                                                                                                                                                  # ops.py:26
  SUM = auto(); MAX = auto(); WMMA = auto() # noqa: E702                                                                                                                                                 # ops.py:27

class BufferOps(Enum): LOAD = auto(); CONST = auto(); STORE = auto() # noqa: E702                                                                                                                        # ops.py:28

class MetaOps(Enum):                                                                                                                                                                                     # ops.py:29
  EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); CUSTOM = auto(); ASSIGN = auto(); VIEW = auto(); KERNEL = auto() # noqa: E702                                                      # ops.py:30

Op = Union[UnaryOps, BinaryOps, ReduceOps, MetaOps, TernaryOps, BufferOps]                                                                                                                               # ops.py:31
UNSAFE_PAD_OPS = {UnaryOps.RECIP, UnaryOps.LOG2, UnaryOps.EXP2, BinaryOps.IDIV}                                                                                                                          # ops.py:34

@dataclass(frozen=True)                                                                                                                                                                                  # ops.py:37
class MemBuffer:
  idx: int                                                                                                                                                                                               # ops.py:38
  dtype: DType                                                                                                                                                                                           # ops.py:39
  st: ShapeTracker                                                                                                                                                                                       # ops.py:40

@dataclass(frozen=True)                                                                                                                                                                                  # ops.py:43
class ConstBuffer:
  val: ConstType | Variable                                                                                                                                                                              # ops.py:44
  dtype: DType                                                                                                                                                                                           # ops.py:45
  st: ShapeTracker                                                                                                                                                                                       # ops.py:46

@dataclass(frozen=True)                                                                                                                                                                                  # ops.py:49
class KernelInfo:
  local_dims: int = 0           # number of local dimensions  (this is remapping RANGE to SPECIAL)                                                                                                       # ops.py:50
  upcasted: int = 0             # count that are upcasted     (this is remapping RANGE to EXPAND)                                                                                                        # ops.py:51
  dont_use_locals: bool = False # don't use local indexing                                                                                                                                               # ops.py:52

@dataclass(frozen=True, eq=False)                                                                                                                                                                        # ops.py:55
class LazyOp:
  op: Op                                                                                                                                                                                                 # ops.py:56
  src: Tuple[LazyOp, ...] = ()                                                                                                                                                                           # ops.py:57
  arg: Any = None                                                                                                                                                                                        # ops.py:58

python_alu: Dict[Op, Callable]  = {                                                                                                                                                                      # ops.py:109
  UnaryOps.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan, UnaryOps.EXP2: hook_overflow(math.inf, lambda x: 2**x),
  UnaryOps.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, UnaryOps.RECIP: lambda x: 1/x if x != 0 else math.copysign(math.inf, x),
  UnaryOps.SIN: lambda x: math.sin(x) if not math.isinf(x) else math.nan, UnaryOps.NEG: lambda x: (not x) if isinstance(x, bool) else -x,
  BinaryOps.SHR: operator.rshift, BinaryOps.SHL: operator.lshift, BinaryOps.MUL: operator.mul, BinaryOps.ADD: operator.add,
  BinaryOps.XOR: operator.xor, BinaryOps.MAX: max, BinaryOps.CMPNE: operator.ne, BinaryOps.CMPLT: operator.lt,
  BinaryOps.OR: operator.or_, BinaryOps.AND: operator.and_,
  BinaryOps.MOD: lambda x,y: abs(int(x))%abs(int(y))*(1,-1)[x<0], BinaryOps.IDIV: lambda x,y: abs(x)//abs(y)*(1,-1)[x*y<0] if y != 0 else x*math.inf,
  TernaryOps.MULACC: lambda x,y,z: (x*y)+z, TernaryOps.WHERE: lambda x,y,z: y if x else z}

  def hook_overflow(dv, fxn):                                                                                                                                                                            # ops.py:103
    return wfxn                                                                                                                                                                                          # ops.py:107

truncate: Dict[DType, Callable] = {dtypes.bool: bool,                                                                                                                                                    # ops.py:126
  # TODO: bfloat16
  dtypes.float16: truncate_fp16, dtypes.float32: lambda x: ctypes.c_float(x).value, dtypes.float64: lambda x: ctypes.c_double(x).value,
  dtypes.uint8: lambda x: ctypes.c_uint8(x).value, dtypes.uint16: lambda x: ctypes.c_uint16(x).value,
  dtypes.uint32: lambda x: ctypes.c_uint32(x).value, dtypes.uint64: lambda x: ctypes.c_uint64(x).value,
  dtypes.int8: lambda x: ctypes.c_int8(x).value, dtypes.int16: lambda x: ctypes.c_int16(x).value, dtypes.int32: lambda x: ctypes.c_int32(x).value \
      if isinstance(x,int) else x, dtypes.int64: lambda x: ctypes.c_int64(x).value}
from tinygrad.shape.symbolic import sint, Variable                                                                                                                                                       # lazy.py:6
from tinygrad.shape.shapetracker import ShapeTracker                                                                                                                                                     # lazy.py:7
from tinygrad.device import Buffer                                                                                                                                                                       # lazy.py:8
from __future__ import annotations                                                                                                                                                                       # device.py:1
import multiprocessing, decimal, statistics, random                                                                                                                                                      # device.py:2
from dataclasses import dataclass                                                                                                                                                                        # device.py:3
from collections import defaultdict                                                                                                                                                                      # device.py:4
from typing import List, Optional, Dict, Tuple, Any, cast, Protocol, Type                                                                                                                                # device.py:5
import importlib, inspect, functools, pathlib, os, ctypes, atexit, time, contextlib, array                                                                                                               # device.py:6
from tinygrad.helpers import SAVE_SCHEDULE, getenv, diskcache_get, diskcache_put, DEBUG, GlobalCounters, flat_mv, from_mv, ProfileLogger, PROFILE                                                        # device.py:7
from tinygrad.dtype import DType, ImageDType                                                                                                                                                             # device.py:8
from tinygrad.renderer import Renderer                                                                                                                                                                   # device.py:9
from typing import Optional, List, Tuple, Dict, Any                                                                                                                                                      # renderer/__init__.py:1
import functools                                                                                                                                                                                         # renderer/__init__.py:2
from dataclasses import dataclass, field                                                                                                                                                                 # renderer/__init__.py:3
from tinygrad.helpers import to_function_name, dedup                                                                                                                                                     # renderer/__init__.py:4
from tinygrad.codegen.uops import UOps, UOp, flops_mem                                                                                                                                                   # renderer/__init__.py:5
from __future__ import annotations                                                                                                                                                                       # codegen/uops.py:1
from typing import Optional, Tuple, Any, Set, cast, List, Union, DefaultDict, Callable, Dict                                                                                                             # codegen/uops.py:2
import functools, itertools, math                                                                                                                                                                        # codegen/uops.py:3
from collections import defaultdict                                                                                                                                                                      # codegen/uops.py:4
from enum import Enum, auto                                                                                                                                                                              # codegen/uops.py:5
from dataclasses import dataclass                                                                                                                                                                        # codegen/uops.py:6
from tinygrad.dtype import ConstType, dtypes, DType                                                                                                                                                      # codegen/uops.py:7
from tinygrad.shape.symbolic import sint, Variable                                                                                                                                                       # codegen/uops.py:8
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, exec_alu                                                                                                                                       # codegen/uops.py:9
from tinygrad.helpers import prod, pretty_print                                                                                                                                                          # codegen/uops.py:10

# the order of these UOps controls the order of the toposort                                                                                                                                             # codegen/uops.py:13
class UOps(Enum):
  SINK = auto(); EXPAND = auto(); CONTRACT = auto() # noqa: E702                                                                                                                                         # codegen/uops.py:15
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # noqa: E702                                                                                                   # codegen/uops.py:16
  CONST = auto(); SPECIAL = auto() # noqa: E702                                                                                                                                                          # codegen/uops.py:17
  NOOP = auto(); GEP = auto() # noqa: E702                                                                                                                                                               # codegen/uops.py:18
  CAST = auto(); BITCAST = auto(); VECTORIZE = auto() # noqa: E702                                                                                                                                       # codegen/uops.py:20
  ALU = auto(); REDUCE = auto(); WMMA = auto() # noqa: E702                                                                                                                                              # codegen/uops.py:21
  LOAD = auto(); STORE = auto(); PHI = auto() # noqa: E702                                                                                                                                               # codegen/uops.py:23
  BARRIER = auto(); IF = auto(); RANGE = auto() # noqa: E702                                                                                                                                             # codegen/uops.py:25
  ENDRANGE = auto(); ENDIF = auto() # noqa: E702                                                                                                                                                         # codegen/uops.py:27

END_FOR_UOP = {UOps.IF:(UOps.STORE, UOps.ENDIF), UOps.RANGE:(UOps.PHI, UOps.ENDRANGE)}                                                                                                                   # codegen/uops.py:29

@dataclass(frozen=True, eq=False)                                                                                                                                                                        # codegen/uops.py:32
class UOp:
  op: UOps                                                                                                                                                                                               # codegen/uops.py:33
  dtype: Optional[DType] = None                                                                                                                                                                          # codegen/uops.py:34
  src: Tuple[UOp, ...] = tuple()                                                                                                                                                                         # codegen/uops.py:35
  arg: Any = None                                                                                                                                                                                        # codegen/uops.py:36

@dataclass(frozen=True, repr=False)  # reuse repr from UOp                                                                                                                                               # codegen/uops.py:143
class NOp(UOp):
  name:Optional[str] = None                                                                                                                                                                              # codegen/uops.py:144
  src:Tuple[NOp, ...] = tuple()                                                                                                                                                                          # codegen/uops.py:145
  allow_any_len:bool = False                                                                                                                                                                             # codegen/uops.py:146

from tinygrad.shape.symbolic import sym_infer, sint, Variable                                                                                                                                            # renderer/__init__.py:6
from tinygrad.dtype import DType                                                                                                                                                                         # renderer/__init__.py:7

@dataclass(frozen=True)                                                                                                                                                                                  # renderer/__init__.py:10
class TensorCore: # D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  dims: Tuple[int,int,int] # N, M, K                                                                                                                                                                     # renderer/__init__.py:11
  dtype_in: DType # dtype for A and B                                                                                                                                                                    # renderer/__init__.py:12
  dtype_out: DType # dtype for C and D                                                                                                                                                                   # renderer/__init__.py:13
  threads: List[Tuple[int,int]] # list of (TC dim,amt) that construct the warp thread structure                                                                                                          # renderer/__init__.py:14

@dataclass                                                                                                                                                                                               # renderer/__init__.py:18
class Program:
  name:str                                                                                                                                                                                               # renderer/__init__.py:19
  src:str                                                                                                                                                                                                # renderer/__init__.py:20
  dname:str                                                                                                                                                                                              # renderer/__init__.py:21
  uops:Optional[List[UOp]]=None                                                                                                                                                                          # renderer/__init__.py:22
  mem_estimate:sint=0  # TODO: get this from the load/store uops once min/max are good                                                                                                                   # renderer/__init__.py:23
  global_size:Optional[List[int]]=None                                                                                                                                                                   # renderer/__init__.py:26
  local_size:Optional[List[int]]=None                                                                                                                                                                    # renderer/__init__.py:27
  vars:List[Variable]=field(default_factory=list)                                                                                                                                                        # renderer/__init__.py:28
  globals:List[int]=field(default_factory=list)                                                                                                                                                          # renderer/__init__.py:29
  outs:List[int]=field(default_factory=list)                                                                                                                                                             # renderer/__init__.py:30
  _ran_post_init:bool=False  # NOTE: this is needed if you call replace on the Program                                                                                                                   # renderer/__init__.py:31

class Renderer:                                                                                                                                                                                          # renderer/__init__.py:71
  device: str = ""                                                                                                                                                                                       # renderer/__init__.py:72
  suffix: str = ""                                                                                                                                                                                       # renderer/__init__.py:73
  supports_float4: bool = True                                                                                                                                                                           # renderer/__init__.py:75
  has_local: bool = True                                                                                                                                                                                 # renderer/__init__.py:76
  has_shared: bool = True                                                                                                                                                                                # renderer/__init__.py:77
  global_max: Optional[Tuple[int, ...]] = (0x8FFFFFFF,) * (3) # TODO: UOps.SPECIAL int32 indexes right now                                                                                               # renderer/__init__.py:79
  local_max: Optional[Tuple[int, ...]] = (0x8FFFFFFF,) * (3) # TODO: UOps.SPECIAL int32 indexes right now                                                                                                # renderer/__init__.py:80
  shared_max: int = 32768                                                                                                                                                                                # renderer/__init__.py:81
  tensor_cores: List[TensorCore] = []                                                                                                                                                                    # renderer/__init__.py:82
  extra_matcher: Any = None                                                                                                                                                                              # renderer/__init__.py:83

Device = _Device()                                                                                                                                                                                       # device.py:40

  class _Device:                                                                                                                                                                                         # device.py:13
    def __init__(self) -> None: self._devices: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]  # noqa: E501        # device.py:14

@dataclass(frozen=True, eq=True)                                                                                                                                                                         # device.py:45
class BufferOptions:
  image: Optional[ImageDType] = None                                                                                                                                                                     # device.py:46
  uncached: bool = False                                                                                                                                                                                 # device.py:47
  cpu_access: bool = False                                                                                                                                                                               # device.py:48
  host: bool = False                                                                                                                                                                                     # device.py:49
  nolru: bool = False                                                                                                                                                                                    # device.py:50

MallocAllocator = _MallocAllocator()                                                                                                                                                                     # device.py:170

  class LRUAllocator(Allocator):  # pylint: disable=abstract-method                                                                                                                                      # device.py:143
    def __init__(self): self.cache: Dict[Tuple[int, Optional[BufferOptions]], Any] = defaultdict(list)                                                                                                   # device.py:148

  def hcq_command(func):                                                                                                                                                                                 # device.py:200
    return __wrapper                                                                                                                                                                                     # device.py:216

class HCQCompiled(Compiled):                                                                                                                                                                             # device.py:481
  """                                                                                                                                                                                                    # device.py:482
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """
  devices: List[HCQCompiled] = []                                                                                                                                                                        # device.py:485
  gpu2cpu_copy_time_diff: decimal.Decimal = decimal.Decimal('nan')                                                                                                                                       # device.py:486
  gpu2cpu_compute_time_diff: decimal.Decimal = decimal.Decimal('nan')                                                                                                                                    # device.py:487

# Protocol for hcq compatible allocators for allocated buffers to contain VA address and it's size.                                                                                                      # device.py:599
class HCQBuffer(Protocol): va_addr:int; size:int # noqa: E702

from weakref import ref, ReferenceType, WeakValueDictionary                                                                                                                                              # lazy.py:9
lazycache: WeakValueDictionary[Any, LazyBuffer] = WeakValueDictionary()                                                                                                                                  # lazy.py:11

view_supported_devices = {"LLVM", "CLANG", "CUDA", "NV", "AMD", "METAL", "DISK"}                                                                                                                         # lazy.py:25

from tinygrad.multi import MultiLazyBuffer                                                                                                                                                               # tensor.py:14
from __future__ import annotations                                                                                                                                                                       # multi.py:1
from typing import Optional, Union, Any, Tuple, List, Dict                                                                                                                                               # multi.py:2
import functools, itertools, operator                                                                                                                                                                    # multi.py:3
from tinygrad.helpers import all_same, all_int, dedup, prod, DEBUG, RING, getenv                                                                                                                         # multi.py:4
from tinygrad.dtype import DType, ConstType                                                                                                                                                              # multi.py:5
from tinygrad.ops import BinaryOps, MetaOps, UnaryOps, TernaryOps, ReduceOps                                                                                                                             # multi.py:6
from tinygrad.lazy import LazyBuffer                                                                                                                                                                     # multi.py:7
from tinygrad.shape.shapetracker import sint                                                                                                                                                             # multi.py:8

from tinygrad.ops import MetaOps, truncate                                                                                                                                                               # tensor.py:15
from tinygrad.device import Device, Buffer, BufferOptions                                                                                                                                                # tensor.py:16
from tinygrad.shape.symbolic import sint, Variable, MulNode, SumNode, NumNode, Node                                                                                                                      # tensor.py:17
from tinygrad.engine.realize import run_schedule, memory_planner                                                                                                                                         # tensor.py:18
from typing import List, Dict, Optional, cast, Generator, Tuple, Union                                                                                                                                   # engine/realize.py:1
import time, pprint                                                                                                                                                                                      # engine/realize.py:2
from collections import defaultdict                                                                                                                                                                      # engine/realize.py:3
from dataclasses import dataclass, replace                                                                                                                                                               # engine/realize.py:4
from tinygrad.helpers import colored, getenv, DEBUG, GlobalCounters, ansilen, BEAM, NOOPT, all_int, CAPTURING, Metadata, Context, TRACEMETA, dedup                                                       # engine/realize.py:5
from tinygrad.ops import MetaOps, LazyOp                                                                                                                                                                 # engine/realize.py:6
from tinygrad.dtype import dtypes                                                                                                                                                                        # engine/realize.py:7
from tinygrad.device import Device, Buffer                                                                                                                                                               # engine/realize.py:8
from tinygrad.shape.symbolic import Variable, sym_infer, sint                                                                                                                                            # engine/realize.py:9
from tinygrad.renderer import Renderer, Program                                                                                                                                                          # engine/realize.py:10
from tinygrad.codegen.kernel import Kernel                                                                                                                                                               # engine/realize.py:11
from __future__ import annotations                                                                                                                                                                       # codegen/kernel.py:1
import itertools, functools                                                                                                                                                                              # codegen/kernel.py:2
from dataclasses import dataclass, replace                                                                                                                                                               # codegen/kernel.py:3
from collections import defaultdict                                                                                                                                                                      # codegen/kernel.py:4
from typing import Optional, List, Tuple, cast, Dict, Union, Final, DefaultDict, Any                                                                                                                     # codegen/kernel.py:5
from tinygrad.ops import LazyOp, UnaryOps, BinaryOps, ReduceOps, MemBuffer, ConstBuffer, BufferOps, MetaOps, UNSAFE_PAD_OPS, verify_lazyop, KernelInfo                                                   # codegen/kernel.py:7
from tinygrad.device import Device                                                                                                                                                                       # codegen/kernel.py:8
from tinygrad.renderer import Renderer, TensorCore, Program                                                                                                                                              # codegen/kernel.py:9
from tinygrad.dtype import ImageDType                                                                                                                                                                    # codegen/kernel.py:10
from tinygrad.helpers import all_same, colored, ansilen, dedup, getenv, prod, DEBUG, TC_OPT, USE_TC, round_up, all_int, \                                                                                # codegen/kernel.py:11
                             get_contraction, to_function_name, diskcache_put, ContextVar
from tinygrad.shape.shapetracker import ShapeTracker                                                                                                                                                     # codegen/kernel.py:13
from tinygrad.shape.symbolic import sint                                                                                                                                                                 # codegen/kernel.py:14
from tinygrad.shape.view import strides_for_shape                                                                                                                                                        # codegen/kernel.py:15
from tinygrad.codegen.uopgraph import UOpGraph                                                                                                                                                           # codegen/kernel.py:16
from __future__ import annotations                                                                                                                                                                       # codegen/uopgraph.py:1
from typing import Iterator, Optional, Tuple, Dict, List, Set, Union, cast, TYPE_CHECKING, Any, DefaultDict, Callable                                                                                    # codegen/uopgraph.py:2
import functools, itertools, heapq, math, operator                                                                                                                                                       # codegen/uopgraph.py:3
from collections import defaultdict                                                                                                                                                                      # codegen/uopgraph.py:4
from tinygrad.dtype import dtypes, PtrDType, ImageDType, DType                                                                                                                                           # codegen/uopgraph.py:5
from tinygrad.ops import UnaryOps, BinaryOps, exec_alu                                                                                                                                                   # codegen/uopgraph.py:6
from tinygrad.helpers import DEBUG, getenv, flatten, dedup, TRANSCENDENTAL, prod, CI, all_same, partition                                                                                                # codegen/uopgraph.py:7
from tinygrad.codegen.uops import UOp, NOp, UOps, UPat, PatternMatcher, END_FOR_UOP, type_verify, print_uops                                                                                             # codegen/uopgraph.py:8
from tinygrad.codegen.transcendental import xexp2, xlog2, xsin, TRANSCENDENTAL_SUPPORTED_DTYPES                                                                                                          # codegen/uopgraph.py:9
import math, functools                                                                                                                                                                                   # codegen/transcendental.py:1
from typing import Tuple, List                                                                                                                                                                           # codegen/transcendental.py:2
from tinygrad.dtype import dtypes, DType                                                                                                                                                                 # codegen/transcendental.py:3
from tinygrad.codegen.uops import UOp                                                                                                                                                                    # codegen/transcendental.py:4
TRANSCENDENTAL_SUPPORTED_DTYPES = {dtypes.float16, dtypes.float32, dtypes.float64}                                                                                                                       # codegen/transcendental.py:6
if TYPE_CHECKING: from tinygrad.renderer import Renderer                                                                                                                                                 # codegen/uopgraph.py:10
float4_folding = PatternMatcher([                                                                                                                                                                        # codegen/uopgraph.py:72
  (UPat(UOps.EXPAND, src=UPat(UOps.LOAD, src=(UPat(name="buf"), UPat()), allow_any_len=True), name="ex"), fold_expanded),
  (UPat({UOps.BARRIER, UOps.SINK}, src=UPat(UOps.STORE, src=(UPat(name="buf"), UPat(), UPat()), allow_any_len=True), name="ex"), fold_expanded),
  (UPat(UOps.VECTORIZE, src=UPat(UOps.REDUCE), name="vec"), vectorize_reduce),
  (UPat(UOps.VECTORIZE, src=UPat({UOps.ALU, UOps.CAST, UOps.BITCAST}), name="vec"), vectorize_alu),
])

  class UPat:                                                                                                                                                                                            # codegen/uops.py:157
    def __init__(self, op:Optional[Union[UOps, Set[UOps]]]=None, arg:Any=None, src:Optional[Union[Tuple[UPat, ...], List[UPat], UPat]]=None,                                                             # codegen/uops.py:158
                 name:Optional[str]=None, dtype:Optional[Union[DType, Set[DType]]]=None, allow_any_len:bool=False):
      self.op: Optional[Tuple[UOps, ...]] = None if op is None else (tuple(op) if isinstance(op, set) else (op,))                                                                                        # codegen/uops.py:160
      self.dtype: Optional[Tuple[DType, ...]] = None if dtype is None else (tuple(dtype) if isinstance(dtype, set) else (dtype,))                                                                        # codegen/uops.py:161
      self.arg, self.name = arg, name                                                                                                                                                                    # codegen/uops.py:162
      self.src: Any = None                                                                                                                                                                               # codegen/uops.py:163
      if isinstance(src, list): self.src = list(itertools.permutations(src))                                                                                                                             # codegen/uops.py:165
      elif isinstance(src, tuple): self.src = [src]                                                                                                                                                      # codegen/uops.py:167
      elif isinstance(src, UPat): self.src = [itertools.repeat(src)]                                                                                                                                     # codegen/uops.py:169
      self.allowed_len: int = 0 if allow_any_len or isinstance(src, UPat) or src is None else len(src)                                                                                                   # codegen/uops.py:171

  class PatternMatcher:                                                                                                                                                                                  # codegen/uops.py:194
    def __init__(self, patterns:List[Tuple[Union[UPat, NOp], Callable]]):                                                                                                                                # codegen/uops.py:195
      self.patterns = patterns                                                                                                                                                                           # codegen/uops.py:196
      self.pdict: DefaultDict[Tuple[UOps, Any], List[Tuple[UPat, Callable]]] = defaultdict(list)                                                                                                         # codegen/uops.py:197
      for p,fxn in self.patterns:                                                                                                                                                                        # codegen/uops.py:199
        if isinstance(p, NOp): p = p.compile()                                                                                                                                                           # codegen/uops.py:200
        assert p.op is not None                                                                                                                                                                          # codegen/uops.py:201
        for uop in p.op: self.pdict[(uop, p.arg)].append((p, fxn))                                                                                                                                       # codegen/uops.py:202

transcendental_folding = PatternMatcher([(UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat(name="d"),), arg=k), cast(Callable, v))                                                        # codegen/uopgraph.py:136
                                         for k,v in ((UnaryOps.EXP2, xexp2), (UnaryOps.LOG2, xlog2), (UnaryOps.SIN, xsin))])

constant_folder = PatternMatcher([                                                                                                                                                                       # codegen/uopgraph.py:184
  # bigint is rewritten to int32
  (UPat({UOps.CONST, UOps.ALU, UOps.SPECIAL, UOps.RANGE, UOps.EXPAND}, dtype=dtypes.bigint, name="x"),
   lambda x: UOp(x.op, dtypes.int32, x.src, x.arg)),
  # VECTORIZE/GEP
  (NOp(UOps.GEP, src=(NOp(UOps.VECTORIZE, name="cast"),), name="gep"), lambda gep, cast: cast.src[gep.arg]),
  *[(NOp(UOps.VECTORIZE, dtypes.float.vec(i), tuple(NOp(UOps.GEP, dtypes.float,
                         src=(NOp.var('x', dtype=dtypes.float.vec(i)),), arg=j) for j in range(i))), lambda x: x) for i in [2, 4, 8, 16]],
  *[(NOp(UOps.VECTORIZE, dtypes.half.vec(i), tuple(NOp(UOps.GEP, dtypes.half,
                         src=(NOp.var('x', dtype=dtypes.half.vec(i)),), arg=j) for j in range(i))), lambda x: x) for i in [2, 4, 8, 16]],
  # tensor core with a 0 input is acc
  *[(NOp(UOps.WMMA, src=(NOp(UOps.VECTORIZE, src=tuple(NOp.const(None, 0.0) for _ in range(i))), NOp.var(), NOp.var('acc'))),
     lambda acc: acc) for i in [2, 4, 8]],
  *[(NOp(UOps.WMMA, src=(NOp.var(), NOp(UOps.VECTORIZE, src=tuple(NOp.const(None, 0.0) for _ in range(i))), NOp.var('acc'))),
     lambda acc: acc) for i in [2, 4, 8]],
  # tensor core cleanups
  *[(NOp(UOps.REDUCE, src=(NOp(UOps.EXPAND, src=tuple(NOp(UOps.GEP, dtypes.float, src=(NOp.var('x'),), arg=i) for i in range(j)), name="expand"),)
    ,name="reduce", allow_any_len=True), reduce_before_expand) for j in [2,4,8]],
  (NOp.var("add") + NOp(UOps.WMMA, name="wmma"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
  # threefry
  (NOp(UOps.ALU, dtype=dtypes.uint64, src=(NOp.var("x"), NOp.var("seed")), arg=BinaryOps.THREEFRY), threefry2x32),
  # extra arange loop folding because we don't fold adds. TODO: fold adds
  (NOp(UOps.REDUCE, src=((NOp.var("idx") + NOp.cvar("mval") * NOp(UOps.RANGE, src=(NOp.var("loop_start"), NOp.var("loop_end")), name="rng") +
                          NOp.var("idx2") + NOp.var("idx3"))
   .lt(NOp.cvar("compval")).where(NOp.cvar("multconst"), NOp.const(None, 0)),), arg=BinaryOps.ADD, name="reduce", allow_any_len=True), loop_collapse),
  (NOp(UOps.REDUCE, src=((NOp.var("idx") + NOp.cvar("mval") * NOp(UOps.RANGE, src=(NOp.var("loop_start"), NOp.var("loop_end")), name="rng") +
                          NOp.var("idx2"))
   .lt(NOp.cvar("compval")).where(NOp.cvar("multconst"), NOp.const(None, 0)),), arg=BinaryOps.ADD, name="reduce", allow_any_len=True), loop_collapse),
  # arange loop folding (reduce)
  (NOp(UOps.REDUCE, src=((NOp.var("idx") + NOp.cvar("mval") * NOp(UOps.RANGE, src=(NOp.var("loop_start"), NOp.var("loop_end")), name="rng"))
   .lt(NOp.cvar("compval")).where(NOp.cvar("multconst"), NOp.const(None, 0)),), arg=BinaryOps.ADD, name="reduce", allow_any_len=True), loop_collapse),
  (NOp(UOps.REDUCE, src=((NOp.var("idx") - NOp(UOps.RANGE, src=(NOp.var("loop_start"), NOp.var("loop_end")), name="rng"))
   .lt(NOp.cvar("compval")).where(NOp.cvar("multconst"), NOp.const(None, 0)),), arg=BinaryOps.ADD, name="reduce", allow_any_len=True),
   lambda **kwargs: loop_collapse(mval=UOp.const(dtypes.int, -1), **kwargs)),
  # arange loop folding (unrolled)
  (NOp(UOps.REDUCE, src=((NOp.var("idx") + NOp.cvar("mval") * NOp(UOps.RANGE, src=(NOp.var("loop_start"), NOp.var("loop_end")), name="rng"))
   .lt(NOp.cvar("compval")).where(NOp.cvar("multconst"), NOp.const(None, 0)) + NOp.var("extra"),),
   arg=BinaryOps.ADD, name="reduce", allow_any_len=True), loop_collapse),
  # indexing (with a multiply offset)!
  (NOp(UOps.REDUCE, src=(NOp.var('idx').eq(NOp(UOps.RANGE, name="rng")).cast()*
    NOp(UOps.LOAD, src=(NOp.var("buf"), NOp.var('add')+NOp.var('mul')*NOp(UOps.RANGE, name="rng")), name="ld"),),
    arg=BinaryOps.ADD, name="reduce", allow_any_len=True), index_collapse),
  (NOp(UOps.REDUCE, src=(NOp.var('idx').ne(NOp(UOps.RANGE, name="rng")).__neg__().cast()*
    NOp(UOps.LOAD, src=(NOp.var("buf"), NOp(UOps.RANGE, name="rng")), name="ld"),),
    arg=BinaryOps.ADD, name="reduce", allow_any_len=True),
    lambda **kwargs: index_collapse(add=UOp.const(dtypes.int, 0), mul=UOp.const(dtypes.int, 1), **kwargs)),
  (NOp(UOps.REDUCE, src=(NOp.var('idx').eq(NOp(UOps.RANGE, name="rng")).where(
    NOp(UOps.LOAD, src=(NOp.var("buf"), NOp.var('add')+NOp.var('mul')*NOp(UOps.RANGE, name="rng")), name="ld"), NOp.const(None, 0.0)),),
    arg=BinaryOps.ADD, name="reduce", allow_any_len=True), index_collapse),
  # other arange folders
  (NOp.cvar("c1") - (NOp.var("x") + NOp.cvar("c2")), lambda c1, c2, x: (c1-c2)-x),  # c1 - (x + c2) -> (c1-c2) - x
  (-(NOp.var("x") * NOp.cvar("c1")), lambda x, c1: x*-c1),
  # max folding
  (NOp.max(NOp.var('x'), NOp.var('y')), lambda x,y: x if x.vmin.arg >= y.vmax.arg else y if x.vmax.arg <= y.vmin.arg else None),
  # const rules
  (NOp(UOps.GEP, src=(NOp.cvar("c"),), name="root"), lambda root, c: root.const(c.arg)),
  (UPat(UOps.CAST, name="root", src=UPat(UOps.CONST, name="c")), lambda root, c: root.const(c.arg)),
  # a REDUCE without ranges is a NOOP
  (NOp(UOps.REDUCE, src=(NOp.var('x'),)), lambda x: x),
  # GEP on a const is the const
  (NOp(UOps.GEP, src=(NOp.cvar("x"),), name="root"), lambda root,x: root.const(x.arg)),
  # a conditional with the same results either way is a noop, also fold const conditionals
  (NOp.var().where(NOp.var("val"), NOp.var("val")), lambda val: val),
  (NOp.cvar('gate').where(NOp.var('c0'), NOp.var('c1')), lambda gate, c0, c1: c0 if gate.arg else c1),
  # ** constant folding **
  (UPat(UOps.ALU, name="root", src=UPat(UOps.CONST)), lambda root: root.const(exec_alu(root.arg, root.dtype, [x.arg for x in root.src]))),
  # ** self folding **
  (-(-NOp.var('x')), lambda x: x),    # -(-x) -> x
  (NOp.var('x') + 0, lambda x: x),    # x+0 -> x
  (NOp.var('x') * 1, lambda x: x),    # x*1 -> x
  (NOp.var('x') * -1, lambda x: -x),  # x*-1 -> -x
  (NOp.var('x') // NOp.var('x'), lambda x: x.const(1)), # x//x -> 1
  (NOp.var('x') // 1, lambda x: x),   # x//1 -> x
  (NOp.var('x') // -1, lambda x: -x), # x//-1 -> -x
  (NOp.var('x') / NOp.var('x'), lambda x: x.const(1)), # x/x -> 1
  (NOp.var('x') / NOp.cvar('c'), lambda x,c: x*exec_alu(UnaryOps.RECIP, c.dtype, [c.arg])),    # x/c -> x*(1/c)
  # ** zero folding **
  # x*0 -> 0 or 0*x -> 0
  # if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (NOp.var('x') * 0, lambda x: x.const(float('nan') if isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0)),
  # x-x -> 0
  (NOp.var('x') - NOp.var('x'), lambda x: x.const(0)),
  (UPat(UOps.ALU, name='x'), lambda x: x.const(x.vmin.arg) if x.vmin.arg == x.vmax.arg else None),
  # ** load/store folding **
  (NOp.store(NOp.var("buf"), NOp.var("idx"), NOp.load(NOp.var("buf"), NOp.var("idx"))), lambda buf,idx:UOp(UOps.NOOP)),
  # ** two stage add/mul folding **
  ((NOp.var('x') + NOp.cvar('c1')) + NOp.cvar('c2'), lambda x,c1,c2: x+x.const(exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, c2.arg]))),
  ((NOp.var("x") * NOp.cvar("c1")) * NOp.cvar("c2"), lambda x,c1,c2: x*x.const(exec_alu(BinaryOps.MUL, x.dtype, [c1.arg, c2.arg]))),
  # *** rules from symbolic ***
  # ** lt **
  # c0*x<c1 for positive int c0,c1
  ((NOp.cvar('c0')*NOp.var('x')).lt(NOp.cvar('c1')),
   lambda x,c0,c1: x.lt(math.ceil(c1.arg/c0.arg)) if dtypes.is_int(x.dtype) and c0.arg > 0 and c1.arg > 0 else None),
  # mul add lt
  (((NOp.cvar('c0')*NOp.var('x'))+NOp.var('x2')).lt(NOp.cvar('c1')),
   lambda x,x2,c0,c1: x.lt(c1.arg//c0.arg) if c1.arg % c0.arg == 0 and c0.arg > x2.vmax.arg and x2.vmin.arg >= 0 else None),
  # ** div **
  # # div folding
  (NOp.var('x') // NOp.cvar('c'), lambda x,c:
   newx if 0 < c.arg and not dtypes.is_unsigned(x.dtype) and (newx:=div_folding(x,c.arg)) is not None else None),
  # mul add div
  (((NOp.cvar('c0')*NOp.var('x'))+NOp.var('x2')) // NOp.cvar('c1'), lambda x,x2,c0,c1:\
   x*(c0.arg//g)//(c1.arg//g) if c0.arg > 0 and c1.arg > 0 and (g:=math.gcd(c0.arg,c1.arg)) > 1 and g > x2.vmax.arg and x2.vmin.arg >= 0 else None),
  # ** mod **
  # apply mod to mod input
  (NOp.var('x') % NOp.cvar('c'), lambda x,c: newx%c if 0 < c.arg and (newx:=mod_folding(x,c.arg)) is not None else None),
  # remove mod
  (NOp.var('x') % NOp.cvar('c'), lambda x,c:\
   x-(x.vmin.arg//c.arg)*c.arg if 0 < c.arg and 0 <= x.vmin.arg and x.vmin.arg//c.arg == x.vmax.arg//c.arg else None),
  # mul mod
  ((NOp.cvar('c0')*NOp.var('x')) % NOp.cvar('c1'), lambda x,c0,c1: (x%(c1.arg//c0.arg))*c0 if c1.arg%c0.arg == 0 else None),
  # mod mod
  ((NOp.var('x') % NOp.cvar('c0')) % NOp.cvar('c1'), lambda x,c0,c1: x % c1 if c0.arg % c1.arg == 0 else None),
  # ** combine terms **
  # -(x+y) -> -x + -y
  (-(NOp.var("x") + NOp.var("y")), lambda x,y: (-x)+(-y)),
  # (x+c0)*c1 -> x*c1+c0*c1. only for signed int, float have inf*0=nan issue
  ((NOp.var("x") + NOp.cvar("c0")) * NOp.cvar("c1"), lambda x,c0,c1:
   x*c1+c0.arg*c1.arg if dtypes.is_int(x.dtype) and not dtypes.is_unsigned(x.dtype) else None),
  # (x*c0)+(x*c1) -> x*(c0+c1)
  (NOp.var("x") * NOp.cvar("c0") + NOp.var("x") * NOp.cvar("c1"), lambda x,c0,c1: x*exec_alu(BinaryOps.ADD, x.dtype, [c0.arg, c1.arg])),
  # (x*c0)+(y*c0) -> (x+y)*c0
  #((NOp.var("x") * NOp.cvar("c0")) + (NOp.var("y") * NOp.cvar("c0")), lambda x,y,c0: c0*(x+y)),
  # (x*x2)/x2 -> x
  ((NOp.var("x") * NOp.var("x2")) / NOp.var("x2"), lambda x,x2: x),
  # (x//c0)//c1 -> x//(c0*c1)
  ((NOp.var("x") // NOp.cvar("c0")) // NOp.cvar("c1"), lambda x,c0,c1: x//x.const(exec_alu(BinaryOps.MUL, x.dtype, [c0.arg, c1.arg]))),
  # (x/x1)/x2 -> x/(x1*x2)
  ((NOp.var("x") / NOp.var("x2")) / NOp.var("x3"), lambda x,x2,x3: x/(x2*x3)),
  # c0 + x < c1 -> x < c1 - c0
  ((NOp.cvar("c0") + NOp.var("x")).lt(NOp.cvar("c1")), lambda x,c0,c1: UOp.lt(x, x.const(exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, -c0.arg])))),
  # (x+x*c0)-> x*(c0+1)
  (NOp.var("x") + NOp.var("x") * NOp.cvar("c0"), lambda x,c0: x*(c0.arg+1)),
  # x!=0 -> (bool)x
  (NOp.var("x").ne(0), lambda x: x.cast(dtypes.bool)),
  # bool != 1 -> not bool
  (NOp.var("x", dtype=dtypes.bool).ne(1), lambda x: -x),
  # TODO: can do the invert of this (flip alt/load) when we fix double ops
  (NOp.store(NOp.var("buf"), NOp.var("idx"), NOp.var("gate").where(NOp.var("alt"), NOp.load(NOp.var("buf"), NOp.var("idx")))),
   lambda buf, idx, gate, alt: UOp.store(buf, idx, alt, gate)),
  # VECTORIZE-PHI-GEP -> PHI-VECTORIZE
  (NOp(UOps.VECTORIZE, src=tuple(NOp(UOps.PHI, src=(NOp(UOps.GEP, src=(NOp.var("val"),), arg=i), NOp.var(f"v{i}"))) for i in range(4)), name="root"),
   lambda root, val, v0, v1, v2, v3: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.VECTORIZE, val.dtype, (v0, v1, v2, v3))))),
  (NOp(UOps.VECTORIZE, src=tuple(NOp(UOps.PHI, src=(NOp(UOps.GEP, src=(NOp.var("val"),), arg=i), NOp.var(f"v{i}"))) for i in range(2)), name="root"),
   lambda root, val, v0, v1: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.VECTORIZE, val.dtype, (v0, v1))))),
  # cast NOOP (NOTE: it's str to deal with PtrDType)
  (NOp(UOps.CAST, name="root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  (NOp(UOps.VECTORIZE, name="root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  # fold gated LOAD/STORE
  (NOp.load(NOp.var("buf"), NOp.var("idx"), NOp.var("var"), NOp.const(dtypes.bool, True)), lambda buf,idx,var: UOp.load(buf, idx, dtype=var.dtype)),
  (NOp.load(NOp.var("buf"), NOp.var("idx"), NOp.var("var"), NOp.const(dtypes.bool, True), NOp.var("barrier")),
   lambda buf,idx,var,barrier: UOp.load(buf, idx, barrier, dtype=var.dtype)),
  (NOp.load(NOp.var(), NOp.var(), NOp.var("var"), NOp.const(dtypes.bool, False)), lambda var: var),
  (NOp.load(NOp.var(), NOp.var(), NOp.var("var"), NOp.const(dtypes.bool, False), NOp.var()), lambda var: var),
  (NOp.store(NOp.var("buf"), NOp.var("idx"), NOp.var("val"), NOp.const(dtypes.bool, True)), UOp.store),
  (NOp.store(NOp.var(), NOp.var(), NOp.var(), NOp.const(dtypes.bool, False)), lambda: UOp(UOps.NOOP)),
  # remove NOOPs from SINK
  (NOp(UOps.SINK, name="root"),
    lambda root: UOp(UOps.SINK, root.dtype, a, root.arg) if len(a:=tuple(x for x in root.src if x.op is not UOps.NOOP)) != len(root.src) else None),
  # ** move add consts to end (NOTE: this is still happening before constant folding) **
  (UPat(UOps.ALU, BinaryOps.ADD, src=(UPat(UOps.CONST, name='c1'), UPat(name='x'))), lambda c1,x: x+c1 if x.op is not UOps.CONST else None),
  (UPat(UOps.ALU, BinaryOps.ADD, src=[UPat(UOps.ALU, BinaryOps.ADD, src=(UPat(name='x'), UPat(UOps.CONST, name='c1'))), UPat(name='y')]),
    lambda x,c1,y: (x+y)+c1),
])

  @dataclass(frozen=True, order=True)                                                                                                                                                                    # dtype.py:9
  class DType:
    def vec(self, sz:int):                                                                                                                                                                               # dtype.py:16
      assert sz > 1 and self.count == 1, f"can't vectorize {self} with size {sz}"                                                                                                                        # dtype.py:17
      return DType(self.priority, self.itemsize*sz, f"{INVERSE_DTYPES_DICT[self.name]}{sz}", None, sz)                                                                                                   # dtype.py:18

  @dataclass(frozen=True, repr=False)  # reuse repr from UOp                                                                                                                                             # codegen/uops.py:143
  class NOp(UOp):
    @staticmethod                                                                                                                                                                                        # codegen/uops.py:148
    def var(name:Optional[str]=None, dtype:Optional[DType]=None): return NOp(UOps.NOOP, dtype=dtype, name=name)

  @dataclass(frozen=True, repr=False)  # reuse repr from UOp                                                                                                                                             # codegen/uops.py:143
  class NOp(UOp):
    def const(self:Union[UOp, DType, None], b:ConstType|Variable): return NOp((x:=UOp.const(self, b)).op, x.dtype, x.src, x.arg)                                                                         # codegen/uops.py:151

      @dataclass(frozen=True, eq=False)                                                                                                                                                                  # codegen/uops.py:32
      class UOp:
        def const(self:Union[UOp, DType, None], b:ConstType|Variable): return UOp._const(self.dtype if isinstance(self, UOp) else self, b)                                                               # codegen/uops.py:72

          @dataclass(frozen=True, eq=False)                                                                                                                                                              # codegen/uops.py:32
          class UOp:
            @staticmethod                                                                                                                                                                                # codegen/uops.py:77
            @functools.lru_cache(maxsize=None)
            def _const(dtype:Optional[DType], b:ConstType|Variable):
              if isinstance(b, Variable): return UOp(UOps.DEFINE_VAR, dtype, (UOp.const(dtypes.int, b.min), UOp.const(dtypes.int, cast(int,b.max))), b)                                                  # codegen/uops.py:79
              if dtype is not None and dtype != (sdtype := dtype.scalar()):                                                                                                                              # codegen/uops.py:80
              return UOp(UOps.CONST, dtype, arg=dtypes.as_const(b, dtype) if dtype is not None else b)                                                                                                   # codegen/uops.py:82

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def __add__(self, x): return self.alu(BinaryOps.ADD, self.ufix(x))                                                                                                                                   # codegen/uops.py:53

      @dataclass(frozen=True, eq=False)                                                                                                                                                                  # codegen/uops.py:32
      class UOp:
        # *** uop syntactic sugar                                                                                                                                                                        # codegen/uops.py:48
        def ufix(self, x): return self.const(x) if not isinstance(x, UOp) else x

      @dataclass(frozen=True, eq=False)                                                                                                                                                                  # codegen/uops.py:32
      class UOp:
        def alu(self, arg, *src:UOp):                                                                                                                                                                    # codegen/uops.py:83
          return type(self)(UOps.ALU, dtypes.bool if arg in {BinaryOps.CMPLT, BinaryOps.CMPNE} else (self, *src)[-1].dtype, (self,)+src, arg)                                                            # codegen/uops.py:84

  @dataclass(frozen=True, repr=False)  # reuse repr from UOp                                                                                                                                             # codegen/uops.py:143
  class NOp(UOp):
    @staticmethod                                                                                                                                                                                        # codegen/uops.py:150
    def cvar(name:Optional[str]=None, dtype:Optional[DType]=None): return NOp(UOps.CONST, dtype=dtype, name=name)

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def __mul__(self, x): return self.alu(BinaryOps.MUL, self.ufix(x))                                                                                                                                   # codegen/uops.py:56

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def lt(self, x): return self.alu(BinaryOps.CMPLT, self.ufix(x))                                                                                                                                      # codegen/uops.py:66

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def where(self, x, y): return self.alu(TernaryOps.WHERE, x, y)                                                                                                                                       # codegen/uops.py:70

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def __sub__(self, x): return self.alu(BinaryOps.ADD, self.ufix(-x))                                                                                                                                  # codegen/uops.py:55

      @dataclass(frozen=True, eq=False)                                                                                                                                                                  # codegen/uops.py:32
      class UOp:
        def __neg__(self): return self.alu(UnaryOps.NEG)                                                                                                                                                 # codegen/uops.py:52

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def eq(self, x): return -self.ne(x)                                                                                                                                                                  # codegen/uops.py:65

      @dataclass(frozen=True, eq=False)                                                                                                                                                                  # codegen/uops.py:32
      class UOp:
        def ne(self, x): return self.alu(BinaryOps.CMPNE, self.ufix(x))                                                                                                                                  # codegen/uops.py:64

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def cast(self, dtype=None): return type(self)(UOps.CAST, dtype, (self,))                                                                                                                             # codegen/uops.py:49

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def max(self, x): return self.alu(BinaryOps.MAX, x)                                                                                                                                                  # codegen/uops.py:68

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def __floordiv__(self, x): return self.alu(BinaryOps.IDIV, self.ufix(x))                                                                                                                             # codegen/uops.py:58

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def __truediv__(self, x): return self.alu(BinaryOps.MUL, self.ufix(x).alu(UnaryOps.RECIP))                                                                                                           # codegen/uops.py:59

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    @staticmethod                                                                                                                                                                                        # codegen/uops.py:86
    def load(*src:UOp, dtype:Optional[DType]=None, **kwargs): return type(src[0])(UOps.LOAD, dtype, tuple(src)+tuple(kwargs.values()))

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    @staticmethod                                                                                                                                                                                        # codegen/uops.py:88
    def store(*src:UOp, **kwargs): return type((src:=(*src, *kwargs.values()))[0])(UOps.STORE, None, src)

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def __mod__(self, x): return self.alu(BinaryOps.MOD, self.ufix(x))                                                                                                                                   # codegen/uops.py:60

  @dataclass(frozen=True, order=True)                                                                                                                                                                    # dtype.py:9
  class DType:
    def scalar(self): return DTYPES_DICT[self.name[:-len(str(self.count))]] if self.count > 1 else self                                                                                                  # dtype.py:19

  class dtypes:                                                                                                                                                                                          # dtype.py:38
    @staticmethod                                                                                                                                                                                        # dtype.py:54
    def as_const(val: ConstType, dtype:DType): return int(val) if dtypes.is_int(dtype) else float(val) if dtypes.is_float(dtype) else bool(val)

      class dtypes:                                                                                                                                                                                      # dtype.py:38
        @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool                                                                                                         # dtype.py:42
        def is_int(x: DType) -> bool: return x.scalar() in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.bigint) or dtypes.is_unsigned(x)

          class dtypes:                                                                                                                                                                                  # dtype.py:38
            @staticmethod                                                                                                                                                                                # dtype.py:44
            def is_unsigned(x: DType) -> bool: return x.scalar() in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)

      class dtypes:                                                                                                                                                                                      # dtype.py:38
        @staticmethod                                                                                                                                                                                    # dtype.py:40
        def is_float(x: DType) -> bool: return x.scalar() in (dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64)

  @dataclass(frozen=True, repr=False)  # reuse repr from UOp                                                                                                                                             # codegen/uops.py:143
  class NOp(UOp):
    def compile(self: NOp, name:Optional[str]=None) -> UPat:                                                                                                                                             # codegen/uops.py:153
      return UPat(name=self.name, dtype=self.dtype) if self.op is UOps.NOOP else UPat(self.op, self.arg, (list if self.commutative()                                                                     # codegen/uops.py:154
        else tuple)(src.compile() for src in self.src) or None, self.name or name, self.dtype, self.allow_any_len)

        @dataclass(frozen=True, eq=False)                                                                                                                                                                # codegen/uops.py:32
        class UOp:
          def commutative(self) -> bool:                                                                                                                                                                 # codegen/uops.py:37
            return (self.op is UOps.ALU and \                                                                                                                                                            # codegen/uops.py:38
              self.arg in {BinaryOps.ADD, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPNE, BinaryOps.XOR, BinaryOps.AND, BinaryOps.OR})

acc_number = 0                                                                                                                                                                                           # codegen/uopgraph.py:398
expander = PatternMatcher([                                                                                                                                                                              # codegen/uopgraph.py:440
  # create gate MUST BE BEFORE expander
  (NOp(UOps.STORE, name="root"), create_gate),
  # do expansion
  (UPat({UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.GEP, UOps.WMMA, UOps.LOAD, UOps.STORE,
         UOps.VECTORIZE, UOps.REDUCE, UOps.EXPAND, UOps.IF}, name="root"), do_expand),
  (NOp(UOps.CONTRACT, name="con"), do_contract),
  # remove EXPANDs from SINK
  (NOp(UOps.SINK, name="root"),
   lambda root: UOp(UOps.SINK, root.dtype, a, root.arg)
    if len(a:=tuple(flatten(x.src if x.op is UOps.EXPAND else (x,) for x in root.src))) != len(root.src) else None),
  # BARRIERs aren't actually expanded
  (NOp(UOps.BARRIER, src=(NOp(UOps.EXPAND, name="ex"),)), lambda ex: UOp(UOps.EXPAND, None, (UOp(UOps.BARRIER, None, ex.src),)*len(ex.src), ex.arg)),
  # empty EXPAND is NOOP
  (NOp(UOps.EXPAND, src=(NOp.var('x'),), arg=()), lambda x: x),
  # EXPAND GEP (needed for WMMA, generalize this) -> vectorized ALU
  (NOp(UOps.EXPAND, name="ex", src=tuple(NOp.var('x').gep(i)+NOp.var('y').gep(i) for i in range(8))),
    lambda ex,x,y: UOp(UOps.EXPAND, ex.dtype, tuple((x+y).gep(i) for i in range(8)), ex.arg)),
])

  @dataclass(frozen=True, eq=False)                                                                                                                                                                      # codegen/uops.py:32
  class UOp:
    def gep(self, i:int): return type(self)(UOps.GEP, self.dtype.scalar() if self.dtype is not None else None, (self,), i)                                                                               # codegen/uops.py:51

reducer = PatternMatcher([                                                                                                                                                                               # codegen/uopgraph.py:468
  (NOp(UOps.REDUCE, name="root"), do_reduce),
  # no ALU on vectorized dtypes
  (UPat({UOps.ALU, UOps.CAST, UOps.BITCAST}, name="alu"), no_vectorized_alu),
  # delete_redundant_gates (after expand, is this still needed?)
  (NOp(UOps.STORE, name="root"), delete_redundant_gates),
])

from tinygrad.codegen.lowerer import lazyop_to_uop                                                                                                                                                       # codegen/kernel.py:17
from __future__ import annotations                                                                                                                                                                       # codegen/lowerer.py:1
from typing import List, Tuple, cast, Optional, Any, Dict                                                                                                                                                # codegen/lowerer.py:2
import functools                                                                                                                                                                                         # codegen/lowerer.py:3
from tinygrad.shape.shapetracker import ShapeTracker, View                                                                                                                                               # codegen/lowerer.py:4
from tinygrad.shape.symbolic import sint                                                                                                                                                                 # codegen/lowerer.py:5
from tinygrad.dtype import dtypes, PtrDType, ImageDType, DType                                                                                                                                           # codegen/lowerer.py:6
from tinygrad.ops import BufferOps, LazyOp, ReduceOps, UnaryOps, MetaOps, KernelInfo, MemBuffer, BinaryOps                                                                                               # codegen/lowerer.py:7
from tinygrad.codegen.uops import UOp, UOps                                                                                                                                                              # codegen/lowerer.py:8
from tinygrad.renderer import Renderer                                                                                                                                                                   # codegen/lowerer.py:9
from tinygrad.helpers import getenv, all_int, get_contraction, prod, partition, flatten                                                                                                                  # codegen/lowerer.py:10
from tinygrad.shape.symbolic import Variable, NumNode, SumNode, MulNode, DivNode, ModNode, LtNode, AndNode                                                                                               # codegen/lowerer.py:13
render_ops: Any = { NumNode: lambda self, ops, ctx: UOp.const(dtypes.bigint, self.b),                                                                                                                    # codegen/lowerer.py:15
                    MulNode: lambda self, ops, ctx: self.a.render(ops, ctx)*variable_to_uop(self.b, ctx),
                    DivNode: lambda self, ops, ctx: self.a.render(ops, ctx)//variable_to_uop(self.b, ctx),
                    ModNode: lambda self, ops, ctx: self.a.render(ops, ctx)%variable_to_uop(self.b, ctx),
                    LtNode: lambda self, ops, ctx: self.a.render(ops, ctx).lt(variable_to_uop(self.b, ctx)),
  Variable: lambda self,ops,ctx: ctx[self] if ctx is not None and self in ctx else \
    UOp(UOps.DEFINE_VAR, dtypes.int, (UOp.const(dtypes.int, self.min), UOp.const(dtypes.int, self.max)), self),
  SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a+b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)),
  AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a*b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

from enum import Enum, auto                                                                                                                                                                              # codegen/kernel.py:18

class OptOps(Enum):                                                                                                                                                                                      # codegen/kernel.py:20
  TC = auto(); UPCAST = auto(); UPCASTMID = auto(); UNROLL = auto(); LOCAL = auto() # noqa: E702                                                                                                         # codegen/kernel.py:21
  GROUP = auto(); GROUPTOP = auto(); NOLOCALS = auto(); PADTO = auto(); SWAP = auto() # noqa: E702                                                                                                       # codegen/kernel.py:22

@dataclass(frozen=True, order=True)                                                                                                                                                                      # codegen/kernel.py:31
class Opt:
  op: OptOps                                                                                                                                                                                             # codegen/kernel.py:32
  axis: Optional[int] = None                                                                                                                                                                             # codegen/kernel.py:33
  amt: Optional[int] = None                                                                                                                                                                              # codegen/kernel.py:34

@dataclass                                                                                                                                                                                               # codegen/kernel.py:43
class TensorCoreOptions:
  axes: Tuple[int, ...] # the location of the original N and M axes if still in the shape                                                                                                                # codegen/kernel.py:44
  axes_exist: Tuple[bool, ...] # true if the original N and M axes are still in the shape                                                                                                                # codegen/kernel.py:45
  axis_pads: Tuple[Tuple[int, int], ...]                                                                                                                                                                 # codegen/kernel.py:46

class Kernel:                                                                                                                                                                                            # codegen/kernel.py:54
  kernel_cnt: Final[DefaultDict[str, int]] = defaultdict(int)                                                                                                                                            # codegen/kernel.py:620

from tinygrad.engine.schedule import ScheduleItem                                                                                                                                                        # engine/realize.py:12
import sys, pickle, atexit, importlib, contextlib                                                                                                                                                        # engine/schedule.py:1
from collections import defaultdict, deque                                                                                                                                                               # engine/schedule.py:2
from dataclasses import dataclass, field                                                                                                                                                                 # engine/schedule.py:3
from typing import Tuple, List, Dict, Optional, Set, DefaultDict, cast, get_args                                                                                                                         # engine/schedule.py:4
from tinygrad.ops import MetaOps, BufferOps, LazyOp, Op, ReduceOps, ConstBuffer, MemBuffer, UNSAFE_PAD_OPS, UnaryOps, reduce_st                                                                          # engine/schedule.py:5
from tinygrad.engine.graph import log_lazybuffer, realized_lazybuffer                                                                                                                                    # engine/schedule.py:6
import os, atexit, functools, contextlib                                                                                                                                                                 # engine/graph.py:1
from collections import defaultdict                                                                                                                                                                      # engine/graph.py:2
from typing import List, Any, DefaultDict                                                                                                                                                                # engine/graph.py:3
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MetaOps, BufferOps, TernaryOps                                                                                                                  # engine/graph.py:4
from tinygrad.device import Device                                                                                                                                                                       # engine/graph.py:5
from tinygrad.helpers import GRAPHPATH, DEBUG, GlobalCounters                                                                                                                                            # engine/graph.py:6
from tinygrad.codegen.uops import UOps, UOp                                                                                                                                                              # engine/graph.py:7
from tinygrad.shape.symbolic import NumNode                                                                                                                                                              # engine/graph.py:8
from tinygrad.lazy import LazyBuffer                                                                                                                                                                     # engine/graph.py:9
with contextlib.suppress(ImportError): import networkx as nx                                                                                                                                             # engine/graph.py:11
if DEBUG >= 2: atexit.register(print_globalcounters)                                                                                                                                                     # engine/graph.py:19

  class ContextVar:                                                                                                                                                                                      # helpers.py:92
    def __ge__(self, x): return self.value >= x                                                                                                                                                          # helpers.py:102

G:Any = None                                                                                                                                                                                             # engine/graph.py:26
counts: DefaultDict[type, int] = defaultdict(int)                                                                                                                                                        # engine/graph.py:33
top_colors = {MetaOps: '#FFFFa0', UnaryOps: "#c0c0c0", ReduceOps: "#FFA0A0", BinaryOps: "#c0c0c0",                                                                                                       # engine/graph.py:46
              TernaryOps: "#c0c0c0", BufferOps: '#a0a0ff'}
graph_uops_cnt = 0                                                                                                                                                                                       # engine/graph.py:76
from tinygrad.helpers import GRAPH, DEBUG, MULTIOUTPUT, SAVE_SCHEDULE, FUSE_CONV_BW, FUSE_ARANGE, \                                                                                                      # engine/schedule.py:7
                             GlobalCounters, colored, prod, dedup, all_int, merge_dicts, getenv, Metadata
from tinygrad.shape.symbolic import Variable, sint                                                                                                                                                       # engine/schedule.py:9
from tinygrad.dtype import ConstType, ImageDType, dtypes                                                                                                                                                 # engine/schedule.py:10
from tinygrad.lazy import LazyBuffer                                                                                                                                                                     # engine/schedule.py:11
from tinygrad.shape.shapetracker import ShapeTracker                                                                                                                                                     # engine/schedule.py:12
from tinygrad.device import Buffer                                                                                                                                                                       # engine/schedule.py:13
from tinygrad.shape.view import View, strides_for_shape                                                                                                                                                  # engine/schedule.py:14
sys.setrecursionlimit(10000)                                                                                                                                                                             # engine/schedule.py:17
logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None                                                                                                                               # engine/schedule.py:20

@dataclass(frozen=True)                                                                                                                                                                                  # engine/schedule.py:25
class ScheduleItem:
  ast: LazyOp                                                                                                                                                                                            # engine/schedule.py:26
  bufs: Tuple[Buffer, ...]                                                                                                                                                                               # engine/schedule.py:27
  metadata: Optional[List[Metadata]] = None                                                                                                                                                              # engine/schedule.py:28

@dataclass(frozen=True)                                                                                                                                                                                  # engine/schedule.py:39
class LBScheduleItem:
  ast: LazyOp                                                                                                                                                                                            # engine/schedule.py:40
  outputs: List[LazyBuffer]                                                                                                                                                                              # engine/schedule.py:41
  inputs: List[LazyBuffer]                                                                                                                                                                               # engine/schedule.py:42
  var_vals: Dict[Variable, int] = field(default_factory=dict)                                                                                                                                            # engine/schedule.py:43
  metadata: List[Metadata] = field(default_factory=list)                                                                                                                                                 # engine/schedule.py:44

SCHEDULES: List[Tuple[DefaultDict[LBScheduleItem, List[LBScheduleItem]], DefaultDict[LBScheduleItem, int]]] = []                                                                                         # engine/schedule.py:259
logkerns, logkerns_level = open(getenv("LOGKERNS", ""), "a") if getenv("LOGKERNS", "") else None, getenv("LOGKERNS_LEVEL", 1)                                                                            # engine/realize.py:16

method_cache: Dict[Tuple[str, LazyOp, int, int, bool], CompiledRunner] = {}                                                                                                                              # engine/realize.py:149

@dataclass(frozen=True)                                                                                                                                                                                  # engine/realize.py:167
class ExecItem:
  prg: Runner                                                                                                                                                                                            # engine/realize.py:168
  bufs: List[Optional[Buffer]]                                                                                                                                                                           # engine/realize.py:169
  metadata: Optional[List[Metadata]] = None                                                                                                                                                              # engine/realize.py:170

capturing: List = []  # put classes with an add method in here                                                                                                                                           # engine/realize.py:218
from tinygrad.engine.schedule import ScheduleItem, create_schedule_with_vars                                                                                                                             # tensor.py:19

import tinygrad.function as F                                                                                                                                                                            # tensor.py:42
"""This is where the forwards and backwards passes live."""                                                                                                                                              # function.py:1
import math                                                                                                                                                                                              # function.py:2
from typing import Tuple, Optional                                                                                                                                                                       # function.py:3
from tinygrad.helpers import argsort                                                                                                                                                                     # function.py:4
from tinygrad.dtype import dtypes, DType, sum_acc_dtype                                                                                                                                                  # function.py:5
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps                                                                                                                                      # function.py:6
from tinygrad.tensor import Function                                                                                                                                                                     # function.py:7
from tinygrad.lazy import LazyBuffer                                                                                                                                                                     # function.py:8
from tinygrad.shape.symbolic import sint                                                                                                                                                                 # function.py:9

class Tensor:                                                                                                                                                                                            # tensor.py:92
  """                                                                                                                                                                                                    # tensor.py:93
  A `Tensor` is a multi-dimensional matrix containing elements of a single data type.

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn
  import numpy as np
  import math
  np.set_printoptions(precision=4)
  ```
  """
  __slots__ = "lazydata", "requires_grad", "grad", "_ctx"                                                                                                                                                # tensor.py:103
  __deletable__ = ('_ctx',)                                                                                                                                                                              # tensor.py:104
  training: ClassVar[bool] = False                                                                                                                                                                       # tensor.py:105
  no_grad: ClassVar[bool] = False                                                                                                                                                                        # tensor.py:106

  _seed: int = int(time.time())                                                                                                                                                                          # tensor.py:386
  _rng_counter: Optional[Tensor] = None                                                                                                                                                                  # tensor.py:387

for device in Device._devices: setattr(Tensor, f"{device.lower()}", functools.partialmethod(Tensor.to, device))                                                                                          # tensor.py:3209
if IMAGE:                                                                                                                                                                                                # tensor.py:3211

  class ContextVar:                                                                                                                                                                                      # helpers.py:92
    def __bool__(self): return bool(self.value)                                                                                                                                                          # helpers.py:101

if TRACEMETA >= 1:                                                                                                                                                                                       # tensor.py:3256

  for name, fn in inspect.getmembers(Tensor, inspect.isfunction):                                                                                                                                        # tensor.py:3257
    if name in ["__class__", "__init__", "__new__", "__repr__", "backward", "sequential"]: continue                                                                                                      # tensor.py:3258
    setattr(Tensor, name, functools.wraps(fn)(_metadata_wrapper(fn)))                                                                                                                                    # tensor.py:3259

      def _metadata_wrapper(fn):                                                                                                                                                                         # tensor.py:3224
        return _wrapper                                                                                                                                                                                  # tensor.py:3254

from tinygrad.engine.jit import TinyJit                       # noqa: F401                                                                                                                               # __init__.py:2
from __future__ import annotations                                                                                                                                                                       # engine/jit.py:1
from typing import TypeVar, Generic, Callable, List, Tuple, Union, Dict, cast, Optional, Any                                                                                                             # engine/jit.py:2
import functools, itertools, collections                                                                                                                                                                 # engine/jit.py:3
from tinygrad.tensor import Tensor                                                                                                                                                                       # engine/jit.py:4
from tinygrad.lazy import LazyBuffer                                                                                                                                                                     # engine/jit.py:5
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, GRAPH, BEAM, getenv, all_int, colored, JIT, dedup                                                                                     # engine/jit.py:6
from tinygrad.device import Buffer, Compiled, Device                                                                                                                                                     # engine/jit.py:7
from tinygrad.dtype import DType                                                                                                                                                                         # engine/jit.py:8
from tinygrad.shape.shapetracker import ShapeTracker                                                                                                                                                     # engine/jit.py:9
from tinygrad.shape.symbolic import Variable, sint, sym_infer                                                                                                                                            # engine/jit.py:10
from tinygrad.engine.realize import ExecItem, capturing, EmptyOp, ViewOp, BufferXfer, CompiledRunner, Runner, _internal_memory_planner                                                                   # engine/jit.py:11
from tinygrad.nn.state import get_parameters                                                                                                                                                             # engine/jit.py:12
import math                                                                                                                                                                                              # nn/__init__.py:1
from typing import Optional, Union, Tuple                                                                                                                                                                # nn/__init__.py:2
from tinygrad.tensor import Tensor                                                                                                                                                                       # nn/__init__.py:3
from tinygrad.helpers import prod                                                                                                                                                                        # nn/__init__.py:4
from tinygrad.nn import optim, state, datasets  # noqa: F401                                                                                                                                             # nn/__init__.py:5
from typing import List                                                                                                                                                                                  # nn/optim.py:2
from tinygrad.helpers import dedup, flatten, getenv                                                                                                                                                      # nn/optim.py:3
from tinygrad.tensor import Tensor                                                                                                                                                                       # nn/optim.py:4
from tinygrad.dtype import dtypes, least_upper_dtype                                                                                                                                                     # nn/optim.py:5

import os, json, pathlib, zipfile, pickle, tarfile, struct                                                                                                                                               # nn/state.py:1
from typing import Dict, Union, List, Optional, Any, Tuple                                                                                                                                               # nn/state.py:2
from tinygrad.tensor import Tensor                                                                                                                                                                       # nn/state.py:3
from tinygrad.dtype import dtypes                                                                                                                                                                        # nn/state.py:4
from tinygrad.helpers import prod, argsort, DEBUG, Timing, CI, unwrap, GlobalCounters, tqdm                                                                                                              # nn/state.py:5
from tinygrad.shape.view import strides_for_shape                                                                                                                                                        # nn/state.py:6
from tinygrad.multi import MultiLazyBuffer                                                                                                                                                               # nn/state.py:7
safe_dtypes = {"BOOL":dtypes.bool, "I8":dtypes.int8, "U8":dtypes.uint8, "I16":dtypes.int16, "U16":dtypes.uint16, "I32":dtypes.int, "U32":dtypes.uint,                                                    # nn/state.py:9
               "I64":dtypes.int64, "U64":dtypes.uint64, "F16":dtypes.float16, "BF16":dtypes.bfloat16, "F32":dtypes.float32, "F64":dtypes.float64}
inverse_safe_dtypes = {v:k for k,v in safe_dtypes.items()}                                                                                                                                               # nn/state.py:11
from collections import OrderedDict                                                                                                                                                                      # nn/state.py:62
import gzip                                                                                                                                                                                              # nn/datasets.py:1
from tinygrad.tensor import Tensor                                                                                                                                                                       # nn/datasets.py:2
from tinygrad.helpers import fetch                                                                                                                                                                       # nn/datasets.py:3

BatchNorm2d = BatchNorm3d = BatchNorm                                                                                                                                                                    # nn/__init__.py:62

from dataclasses import dataclass                                                                                                                                                                        # engine/jit.py:13
from weakref import WeakKeyDictionary                                                                                                                                                                    # engine/jit.py:14

ReturnType = TypeVar('ReturnType')                                                                                                                                                                       # engine/jit.py:130

@dataclass                                                                                                                                                                                               # engine/jit.py:132
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object                                                                                                                                          # engine/jit.py:133
  jit_cache: List[ExecItem]                                                                                                                                                                              # engine/jit.py:134
  input_replace: Dict[Tuple[int, int], int]                                                                                                                                                              # engine/jit.py:135
  extra_view_inputs: List[Tuple[int, int, str, int, DType]]                                                                                                                                              # engine/jit.py:136
  expected_names: List[Union[int, str]]                                                                                                                                                                  # engine/jit.py:137
  expected_st_vars_dtype_device: List[Tuple[ShapeTracker, Tuple[Variable, ...], DType, str]]                                                                                                             # engine/jit.py:138

from tinygrad.shape.symbolic import Variable                  # noqa: F401                                                                                                                               # __init__.py:3
from tinygrad.dtype import dtypes                             # noqa: F401                                                                                                                               # __init__.py:4
from tinygrad.helpers import GlobalCounters, fetch, Context   # noqa: F401                                                                                                                               # __init__.py:5
from tinygrad.device import Device                            # noqa: F401                                                                                                                               # __init__.py:6
a = (Tensor([1,2,3], device="CLANG") + 2)                                                                                                                                                                # ...s/tinygrad.tensor.tolist.py:2

  class Tensor:                                                                                                                                                                                          # tensor.py:92
    def __init__(self, data:Union[None, ConstType, List, Tuple, LazyBuffer, np.ndarray, bytes, MultiLazyBuffer, Variable],                                                                               # tensor.py:108
                 device:Optional[Union[str, tuple, list]]=None, dtype:Optional[DTypeLike]=None, requires_grad:Optional[bool]=None):
      if dtype is not None: dtype = to_dtype(dtype)                                                                                                                                                      # tensor.py:110
      assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"                                                                                                                         # tensor.py:111
      device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)                                                                       # tensor.py:112

        class _Device:                                                                                                                                                                                   # device.py:13
          # NOTE: you can't cache canonicalize in case Device.DEFAULT changes                                                                                                                            # device.py:18
          def canonicalize(self, device:Optional[str]) -> str: return self._canonicalize(device) if device is not None else Device.DEFAULT

            class _Device:                                                                                                                                                                               # device.py:13
              @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none                                                                                # device.py:16
              def _canonicalize(self, device:str) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "")   # noqa: E501

      self.grad: Optional[Tensor] = None                                                                                                                                                                 # tensor.py:115
      self.requires_grad: Optional[bool] = requires_grad                                                                                                                                                 # tensor.py:119
      self._ctx: Optional[Function] = None                                                                                                                                                               # tensor.py:122
      if isinstance(data, LazyBuffer): assert dtype is None or dtype == data.dtype, "dtype doesn't match, and casting isn't supported"                                                                   # tensor.py:125
      elif isinstance(data, get_args(ConstType)): data = _metaop(MetaOps.CONST, tuple(), dtype or dtypes.from_py(data), device, data)                                                                    # tensor.py:126
      elif isinstance(data, Variable): data = _metaop(MetaOps.CONST, tuple(), dtype or dtypes.from_py(data.unbind()[1]), device, data)                                                                   # tensor.py:127
      elif isinstance(data, bytes): data = _frompy(data, dtypes.uint8 if dtype is None else dtype)                                                                                                       # tensor.py:128
      elif isinstance(data, (list, tuple)):                                                                                                                                                              # tensor.py:129
        if dtype is None:                                                                                                                                                                                # tensor.py:130
          if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): dtype = dtypes.bool                                                                                                     # tensor.py:131

            def fully_flatten(l): return [item for sublist in l for item in (fully_flatten(sublist) if isinstance(sublist, (tuple, list)) else [sublist])]                                               # helpers.py:35

          else: dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float                                                                                                                 # tensor.py:132

            def all_int(t: Sequence[Any]) -> TypeGuard[Tuple[int, ...]]: return all(isinstance(s, int) for s in t)                                                                                       # helpers.py:27

        if dtype == dtypes.bfloat16: data = Tensor(_fromnp(np.array(data, np.float32)), device=device).cast(dtypes.bfloat16).lazydata                                                                    # tensor.py:133
        else: data = _fromnp(np.array(data).astype(_to_np_dtype(dtype)))                                                                                                                                 # tensor.py:134

          def _to_np_dtype(dtype:DType) -> Optional[type]: return np.dtype(dtype.fmt).type if dtype.fmt is not None else None                                                                            # tensor.py:49

          def _fromnp(x: np.ndarray) -> LazyBuffer:                                                                                                                                                      # tensor.py:51
            ret = LazyBuffer.metaop(MetaOps.EMPTY, x.shape, _from_np_dtype(x.dtype), "NPY")                                                                                                              # tensor.py:52

              def _from_np_dtype(npdtype:np.dtype) -> DType: return dtypes.fields()[np.dtype(npdtype).name]                                                                                              # tensor.py:48

                class dtypes:                                                                                                                                                                            # dtype.py:38
                  @staticmethod                                                                                                                                                                          # dtype.py:64
                  def fields() -> Dict[str, DType]: return DTYPES_DICT

              class LazyBuffer:                                                                                                                                                                          # lazy.py:26
                @staticmethod                                                                                                                                                                            # lazy.py:70
                def metaop(op, shape:Tuple[sint,...], dtype:DTypeLike, device:str, arg=None, src:Tuple[LazyBuffer, ...]=(), enable_cache=False) -> LazyBuffer:
                  assert isinstance(src, tuple)                                                                                                                                                          # lazy.py:71
                  return create_lazybuffer(device, ShapeTracker.from_shape(shape), dtype, op, arg, src, enable_cache=enable_cache)                                                                       # lazy.py:72

                    @dataclass(frozen=True)                                                                                                                                                              # shape/shapetracker.py:10
                    class ShapeTracker:
                      @staticmethod                                                                                                                                                                      # shape/shapetracker.py:26
                      def from_shape(shape:Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker((View.create(shape),))

                        @dataclass(frozen=True)                                                                                                                                                          # shape/view.py:85
                        class View:
                          @staticmethod                                                                                                                                                                  # shape/view.py:101
                          @functools.lru_cache(maxsize=None)
                          def create(shape:Tuple[sint, ...], strides:Optional[Tuple[sint, ...]]=None, offset:sint=0, mask:Optional[Tuple[Tuple[sint, sint], ...]]=None):
                            if not all(s >= 0 for s in shape): raise ValueError(f"Trying to create View with negative dimension: {shape=}")                                                              # shape/view.py:102
                            strides = canonicalize_strides(shape, strides) if strides else strides_for_shape(shape)                                                                                      # shape/view.py:103

                              @functools.lru_cache(maxsize=None)                                                                                                                                         # shape/view.py:13
                              def strides_for_shape(shape:Tuple[sint, ...]) -> Tuple[sint, ...]:
                                if not shape: return ()                                                                                                                                                  # shape/view.py:14
                                strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))[::-1]                                                                                # shape/view.py:15
                                return canonicalize_strides(shape, strides)                                                                                                                              # shape/view.py:16

                                  @functools.lru_cache(maxsize=None)                                                                                                                                     # shape/view.py:9
                                  def canonicalize_strides(shape:Tuple[sint, ...], strides:Tuple[sint, ...]) -> Tuple[sint, ...]:
                                    return tuple(0 if s == 1 else st for s, st in zip(shape, strides))                                                                                                   # shape/view.py:10

                            if 0 in shape: return View(shape, (0,) * len(shape), offset=0, mask=None, contiguous=True)                                                                                   # shape/view.py:105
                            if mask is not None and all(m == (0,s) for m,s in zip(mask, shape)): mask = None                                                                                             # shape/view.py:107
                            if mask and any(elim := [not (b+1 < e) for b,e in mask]):                                                                                                                    # shape/view.py:111
                            contiguous = offset == 0 and mask is None and strides == strides_for_shape(shape)                                                                                            # shape/view.py:116
                            return View(shape, strides, offset, mask, contiguous)                                                                                                                        # shape/view.py:117

                    def create_lazybuffer(device:str, st:ShapeTracker, dtype:DTypeLike, op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),                                              # lazy.py:12
                                          base:Optional[LazyBuffer]=None, enable_cache=bool(getenv("LAZYCACHE", 1))):
                      if st.size == 0: op, arg, srcs, base = MetaOps.CONST, 0, (), None                                                                                                                  # lazy.py:14

                        @dataclass(frozen=True)                                                                                                                                                          # shape/shapetracker.py:10
                        class ShapeTracker:
                          @property                                                                                                                                                                      # shape/shapetracker.py:38
                          def size(self) -> int: return self.views[-1].size()

                            @dataclass(frozen=True)                                                                                                                                                      # shape/view.py:85
                            class View:
                              @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none                                                                                           # shape/view.py:93
                              def size(self) -> int:
                                ret = prod([x.max if isinstance(x, Node) else x for x in self.shape])                                                                                                    # shape/view.py:95

                                  # NOTE: it returns int 1 if x is empty regardless of the type of x                                                                                                     # helpers.py:13
                                  def prod(x:Iterable[T]) -> Union[T,int]: return functools.reduce(operator.mul, x, 1)

                                assert isinstance(ret, int), f"{ret=} is not int"                                                                                                                        # shape/view.py:96
                                return ret                                                                                                                                                               # shape/view.py:97

                      dtype = to_dtype(dtype)                                                                                                                                                            # lazy.py:15

                        def to_dtype(dtype:DTypeLike) -> DType: return dtype if isinstance(dtype, DType) else getattr(dtypes, dtype)                                                                     # dtype.py:100

                      if op is MetaOps.CONST: arg, enable_cache = dtypes.as_const(arg, dtype) if not isinstance(arg, Variable) else arg, True                                                            # lazy.py:16
                      cache_key = (device, st, dtype, op, arg, tuple(ref(x) for x in srcs)) if base is None else (st, ref(base))                                                                         # lazy.py:18
                      if enable_cache and (rret := lazycache.get(cache_key, None)): return rret                                                                                                          # lazy.py:19
                      ret = LazyBuffer(device, st, dtype, op, arg, srcs, base=base, metadata=_METADATA.get())                                                                                            # lazy.py:21

                        class LazyBuffer:                                                                                                                                                                # lazy.py:26
                          def __init__(self, device:str, st:ShapeTracker, dtype:DTypeLike,                                                                                                               # lazy.py:27
                                       op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
                                       base:Optional[LazyBuffer]=None, metadata:Optional[Metadata]=None):
                            self.device, self.st, self.dtype, self.shape, self.size, self.metadata = device, st, to_dtype(dtype), st.shape, st.size, metadata                                            # lazy.py:30

                              @dataclass(frozen=True)                                                                                                                                                    # shape/shapetracker.py:10
                              class ShapeTracker:
                                @property                                                                                                                                                                # shape/shapetracker.py:35
                                def shape(self) -> Tuple[sint, ...]: return self.views[-1].shape

                            self._base: Optional[LazyBuffer] = None                                                                                                                                      # lazy.py:31
                            if base is None:                                                                                                                                                             # lazy.py:32
                              self.op, self.arg, self.srcs = op, arg, srcs  # this is a LazyOp, except the src is LazyBuffers and not LazyOps                                                            # lazy.py:34
                              assert self.op is not MetaOps.ASSIGN or srcs[1].base.realized is not None, "assign target must be realized"                                                                # lazy.py:35
                              if self.op is MetaOps.VIEW:                                                                                                                                                # lazy.py:37
                                self.buffer = srcs[1].base.buffer if self.op is MetaOps.ASSIGN else Buffer(device, self.size, self.dtype)                                                                # lazy.py:41

                                  class Buffer:                                                                                                                                                          # device.py:52
                                    def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:Optional[BufferOptions]=None,                                                         # device.py:53
                                                 initial_value:Optional[bytes]=None, lb_refcount=0, base:Optional[Buffer]=None, offset:int=0, preallocate=False):
                                      assert isinstance(dtype, DType)                                                                                                                                    # device.py:55
                                      if isinstance(dtype, ImageDType): options = BufferOptions(image=dtype) # TODO: image hack shouldn't be here. where should it be?                                   # device.py:56
                                      self.device, self.size, self.dtype, self.options, self.offset = device, size, dtype, options, offset                                                               # device.py:57
                                      if base is None:                                                                                                                                                   # device.py:58
                                        assert offset == 0, "base buffers can't have offset"                                                                                                             # device.py:59
                                        self._base = None                                                                                                                                                # device.py:60
                                        self._lb_refcount = lb_refcount                                                                                                                                  # device.py:61
                                        if opaque is not None: self.allocate(opaque)                                                                                                                     # device.py:62
                                        if initial_value is not None:                                                                                                                                    # device.py:63
                                      if preallocate: self.allocate()                                                                                                                                    # device.py:70

                              self.buffer.ref(1)                                                                                                                                                         # lazy.py:42

                                class Buffer:                                                                                                                                                            # device.py:52
                                  def ref(self, cnt): self.base._lb_refcount += cnt                                                                                                                      # device.py:75

                                    class Buffer:                                                                                                                                                        # device.py:52
                                      @property                                                                                                                                                          # device.py:72
                                      def base(self) -> Buffer: return self._base if self._base is not None else self

                              self.contiguous_child: Optional[Tuple[ReferenceType[LazyBuffer], ShapeTracker]] = None                                                                                     # lazy.py:43
                              self.forced_realize = False                                                                                                                                                # lazy.py:44

                      if enable_cache: lazycache[cache_key] = ret                                                                                                                                        # lazy.py:22
                      return ret                                                                                                                                                                         # lazy.py:23

            ret.buffer.allocate(x)                                                                                                                                                                       # tensor.py:54

              class Buffer:                                                                                                                                                                              # device.py:52
                def allocate(self, opaque=None) -> Buffer:                                                                                                                                               # device.py:78
                  assert not hasattr(self, '_buf'), "can't allocate already allocated buffer"                                                                                                            # device.py:79
                  self.allocator = Device[self.device].allocator                                                                                                                                         # device.py:80

                    class _Device:                                                                                                                                                                       # device.py:13
                      def __getitem__(self, ix:str) -> Compiled: return self.__get_canonicalized_item(self.canonicalize(ix))                                                                             # device.py:19

                        class _Device:                                                                                                                                                                   # device.py:13
                          @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none                                                                    # device.py:21
                          def __get_canonicalized_item(self, ix:str) -> Compiled:
                            assert ((cpn:=multiprocessing.current_process().name) == "MainProcess") or ix.split(":")[0] in ["DISK", "NPY"], \                                                            # device.py:22
                              f"can only open device {ix} from parent, not {cpn}"
                            x = ix.split(":")[0].upper()                                                                                                                                                 # device.py:24
                            ret = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "device") and x in self._devices][0](ix)  # noqa: E501 # device.py:25
                        import numpy as np                                                                                                                                                               # runtime/ops_npy.py:1
                        from tinygrad.helpers import flat_mv                                                                                                                                             # runtime/ops_npy.py:2
                        from tinygrad.device import Compiled, Allocator                                                                                                                                  # runtime/ops_npy.py:3

                              class NpyDevice(Compiled):                                                                                                                                                 # runtime/ops_npy.py:8
                                def __init__(self, device:str): super().__init__(device, NpyAllocator(), None, None, None)                                                                               # runtime/ops_npy.py:9

                                  class Compiled:                                                                                                                                                        # device.py:186
                                    def __init__(self, device:str, allocator:Allocator, renderer:Optional[Renderer], compiler:Optional[Compiler], runtime, graph=None):                                  # device.py:187
                                      self.dname, self.allocator, self.compiler, self.runtime, self.graph = device, allocator, compiler or Compiler(), runtime, graph                                    # device.py:188

                                        class Compiler:                                                                                                                                                  # device.py:176
                                          def __init__(self, cachekey:Optional[str]=None): self.cachekey = None if getenv("DISABLE_COMPILER_CACHE") else cachekey                                        # device.py:177

                                      self.renderer = renderer or Renderer()                                                                                                                             # device.py:189

                            if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()}")                                                                                                           # device.py:26

                            return ret                                                                                                                                                                   # device.py:27

                  if self._base is not None:                                                                                                                                                             # device.py:81
                    self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)                                                                                        # device.py:86
                    if not self.device.startswith("DISK"): GlobalCounters.mem_used += self.nbytes                                                                                                        # device.py:87

                      class Buffer:                                                                                                                                                                      # device.py:52
                        @property                                                                                                                                                                        # device.py:99
                        def nbytes(self): return self.size*self.dtype.itemsize

                  return self                                                                                                                                                                            # device.py:88

            del ret.srcs                                                                                                                                                                                 # tensor.py:55
            return ret                                                                                                                                                                                   # tensor.py:56

      if not isinstance(data, (LazyBuffer, MultiLazyBuffer)):                                                                                                                                            # tensor.py:141
      if isinstance(device, tuple):                                                                                                                                                                      # tensor.py:145
        self.lazydata = data if data.device == device else data.copy_to_device(device)                                                                                                                   # tensor.py:153

          class LazyBuffer:                                                                                                                                                                              # lazy.py:26
            def copy_to_device(self, device:str, force: bool = False) -> LazyBuffer:                                                                                                                     # lazy.py:119
              if self.device == device: return self                                                                                                                                                      # lazy.py:121
              if not force and self.st.contiguous and self.size == self.base.size and not self.base.realized and self.base.op is MetaOps.COPY:                                                           # lazy.py:124

                @dataclass(frozen=True)                                                                                                                                                                  # shape/shapetracker.py:10
                class ShapeTracker:
                  @property                                                                                                                                                                              # shape/shapetracker.py:29
                  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

                class LazyBuffer:                                                                                                                                                                        # lazy.py:26
                  # NOTE: this has to be a function to prevent self reference                                                                                                                            # lazy.py:63
                  @property
                  def base(self) -> LazyBuffer: return self._base if self._base is not None else self

                class LazyBuffer:                                                                                                                                                                        # lazy.py:26
                  @property                                                                                                                                                                              # lazy.py:57
                  def realized(self) -> Optional[Buffer]:
                    return self.buffer if self._base is None and not hasattr(self, 'srcs') else None                                                                                                     # lazy.py:59

              if self.is_unrealized_const():                                                                                                                                                             # lazy.py:128

                class LazyBuffer:                                                                                                                                                                        # lazy.py:26
                  def is_unrealized_const(self): return self.base.realized is None and self.base.op is MetaOps.CONST and not isinstance(self.base.arg, Variable)                                         # lazy.py:113

              if prod(self.st.shape) < prod(self.base.st.shape): return self.contiguous()._copy(device)                                                                                                  # lazy.py:132

              return self.base._copy(device)._view(self.st)                                                                                                                                              # lazy.py:135

                class LazyBuffer:                                                                                                                                                                        # lazy.py:26
                  def _copy(self, device:str) -> LazyBuffer:                                                                                                                                             # lazy.py:116
                    return create_lazybuffer(device, ShapeTracker.from_shape(self.shape), self.dtype, MetaOps.COPY, self.buffer.nbytes, (self,), enable_cache=False)                                     # lazy.py:117

                class LazyBuffer:                                                                                                                                                                        # lazy.py:26
                  def _view(self, new_st:ShapeTracker) -> LazyBuffer:                                                                                                                                    # lazy.py:208
                    if self.st.size == 0 or (new_st.views[-1].mask is not None and any((x[1]-x[0]) == 0 for x in new_st.views[-1].mask)):                                                                # lazy.py:209

                    if new_st.contiguous and self.base.shape == new_st.shape: return self.base                                                                                                           # lazy.py:211

  def _metadata_wrapper(fn):                                                                                                                                                                             # tensor.py:3224
    def _wrapper(*args, **kwargs):                                                                                                                                                                       # tensor.py:3225
      if _METADATA.get() is not None: return fn(*args, **kwargs)                                                                                                                                         # tensor.py:3226
      if TRACEMETA >= 2:                                                                                                                                                                                 # tensor.py:3228

      else: caller = ""                                                                                                                                                                                  # tensor.py:3248
      token = _METADATA.set(Metadata(name=fn.__name__, caller=caller))                                                                                                                                   # tensor.py:3250
      ret = fn(*args, **kwargs)                                                                                                                                                                          # tensor.py:3251

        class Tensor:                                                                                                                                                                                    # tensor.py:92
          def __add__(self, x) -> Tensor: return self.add(x)                                                                                                                                             # tensor.py:2740

          class Tensor:                                                                                                                                                                                  # tensor.py:92
            def add(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:                                                                                                                          # tensor.py:2490
              return F.Add.apply(*self._broadcasted(x, reverse))                                                                                                                                         # tensor.py:2508

            class Tensor:                                                                                                                                                                                # tensor.py:92
              def _broadcasted(self, y:Union[Tensor, Node, ConstType], reverse:bool=False, match_dtype:bool=True) -> Tuple[Tensor, Tensor]:                                                              # tensor.py:2466
                x: Tensor = self                                                                                                                                                                         # tensor.py:2467
                if not isinstance(y, Tensor):                                                                                                                                                            # tensor.py:2468
                  assert isinstance(y, (*get_args(ConstType), Node)), f"{type(y)=}, {y=}"                                                                                                                # tensor.py:2470
                  if isinstance(x.dtype, ImageDType) or dtypes.is_float(x.dtype) or (dtypes.is_int(x.dtype) and isinstance(y, int)): y_dtype = x.dtype                                                   # tensor.py:2471

                    class Tensor:                                                                                                                                                                        # tensor.py:92
                      @property                                                                                                                                                                          # tensor.py:184
                      def dtype(self) -> DType: return self.lazydata.dtype

                  if isinstance(y, Node): y = Tensor.from_node(y, device=x.device)                                                                                                                       # tensor.py:2473
                  else: y = Tensor(dtypes.as_const(y, y_dtype), x.device, y_dtype, requires_grad=False)                                                                                                  # tensor.py:2474

                    class Tensor:                                                                                                                                                                        # tensor.py:92
                      @property                                                                                                                                                                          # tensor.py:178
                      def device(self) -> Union[str, Tuple[str, ...]]: return self.lazydata.device

              def _metaop(op, shape:Tuple[sint,...], dtype:DType, device:Union[str, Tuple[str, ...]], arg=None, src:Tuple[LazyBuffer, ...]=()):                                                          # tensor.py:44
                if isinstance(device, str): return LazyBuffer.metaop(op, shape, dtype, device, arg, src)                                                                                                 # tensor.py:45

                if match_dtype and x.dtype != y.dtype:                                                                                                                                                   # tensor.py:2476

                if reverse: x, y = y, x                                                                                                                                                                  # tensor.py:2480
                out_shape = _broadcast_shape(x.shape, y.shape)                                                                                                                                           # tensor.py:2483

                  class Tensor:                                                                                                                                                                          # tensor.py:92
                    @property                                                                                                                                                                            # tensor.py:181
                    def shape(self) -> Tuple[sint, ...]: return self.lazydata.shape

                  def _broadcast_shape(*shapes:Tuple[sint, ...]) -> Tuple[sint, ...]:                                                                                                                    # tensor.py:89
                    return tuple(0 if 0 in nth_dim_sizes else max(nth_dim_sizes) for nth_dim_sizes in zip(*_pad_left(*shapes)))                                                                          # tensor.py:90

                      def _pad_left(*shapes:Tuple[sint, ...]) -> Tuple[Tuple[sint, ...], ...]:                                                                                                           # tensor.py:86
                        max_dim = max(len(shape) for shape in shapes)                                                                                                                                    # tensor.py:87
                        return tuple((1,) * (max_dim - len(shape)) + shape for shape in shapes)                                                                                                          # tensor.py:88

                return x._broadcast_to(out_shape), y._broadcast_to(out_shape)                                                                                                                            # tensor.py:2484

              class Tensor:                                                                                                                                                                              # tensor.py:92
                # ***** broadcasted elementwise ops *****                                                                                                                                                # tensor.py:2457
                def _broadcast_to(self, shape:Tuple[sint, ...]) -> Tensor:
                  if self.shape == shape: return self                                                                                                                                                    # tensor.py:2458

              class Tensor:                                                                                                                                                                              # tensor.py:92
                # ***** broadcasted elementwise ops *****                                                                                                                                                # tensor.py:2457
                def _broadcast_to(self, shape:Tuple[sint, ...]) -> Tensor:
                  if self.shape == shape: return self                                                                                                                                              # OLD # tensor.py:2458
                  if self.ndim > len(shape): raise ValueError(f"cannot broadcast tensor to fewer dimensions. shape={self.shape} to {shape=}")                                                            # tensor.py:2459

                    class Tensor:                                                                                                                                                                        # tensor.py:92
                      @property                                                                                                                                                                          # tensor.py:2957
                      def ndim(self) -> int:
                        return len(self.shape)                                                                                                                                                           # tensor.py:2966

                  padded, _ = _pad_left(self.shape, shape)                                                                                                                                               # tensor.py:2461

                  if any(from_ != 1 and from_ != to for from_,to in zip(padded, shape)): raise ValueError(f"cannot broadcast from shape={self.shape} to {shape=}")                                       # tensor.py:2463
                  return F.Expand.apply(self.reshape(padded), shape=shape)                                                                                                                               # tensor.py:2464

                class Tensor:                                                                                                                                                                            # tensor.py:92
                  def reshape(self, shape, *args) -> Tensor:                                                                                                                                             # tensor.py:786
                    new_shape = tuple([s if s is not None else self.shape[i] for i,s in enumerate(argfix(shape, *args))])                                                                                # tensor.py:797

                      def argfix(*x):                                                                                                                                                                    # helpers.py:20
                        if x and x[0].__class__ in (tuple, list):                                                                                                                                        # helpers.py:21
                          if len(x) != 1: raise ValueError(f"bad arg {x}")                                                                                                                               # helpers.py:22
                          return tuple(x[0])                                                                                                                                                             # helpers.py:23

                    if (c := new_shape.count(-1)) > 1: raise RuntimeError(f"only one dimension can be inferred using -1, getting {new_shape}")                                                           # tensor.py:799
                    if c: new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else s for s in new_shape])                                                                                 # tensor.py:800
                    return F.Reshape.apply(self, shape=new_shape) if new_shape != self.shape else self                                                                                                   # tensor.py:801

                      class Function:                                                                                                                                                                    # tensor.py:23
                        @classmethod                                                                                                                                                                     # tensor.py:35
                        def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
                          ctx = fxn(x[0].device, *x, metadata=_METADATA.get())                                                                                                                           # tensor.py:36

                            class Function:                                                                                                                                                              # tensor.py:23
                              def __init__(self, device:Union[str, Tuple[str, ...]], *tensors:Tensor, metadata:Optional[Metadata]=None):                                                                 # tensor.py:24
                                self.device = device                                                                                                                                                     # tensor.py:25
                                self.needs_input_grad = [t.requires_grad for t in tensors]                                                                                                               # tensor.py:26
                                self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False                                                            # tensor.py:27
                                if self.requires_grad: self.parents = tensors                                                                                                                            # tensor.py:28
                                self.metadata = metadata                                                                                                                                                 # tensor.py:29

                          ret = Tensor.__new__(Tensor)                                                                                                                                                   # tensor.py:37
                          ret.lazydata, ret.requires_grad, ret.grad = ctx.forward(*[t.lazydata for t in x], **kwargs), ctx.requires_grad, None                                                           # tensor.py:38

                            class Reshape(Function):                                                                                                                                                     # function.py:187
                              def forward(self, x:LazyBuffer, shape:Tuple[int, ...]) -> LazyBuffer:                                                                                                      # function.py:188
                                self.input_shape = x.shape                                                                                                                                               # function.py:189
                                return x.reshape(shape)                                                                                                                                                  # function.py:190

                                  class LazyBuffer:                                                                                                                                                      # lazy.py:26
                                    def reshape(self, arg:Tuple[sint, ...]): return self._view(self.st.reshape(arg))                                                                                     # lazy.py:214

                                      @dataclass(frozen=True)                                                                                                                                            # shape/shapetracker.py:10
                                      class ShapeTracker:
                                        def reshape(self, new_shape: Tuple[sint, ...]) -> ShapeTracker:                                                                                                  # shape/shapetracker.py:110
                                          if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))                # shape/shapetracker.py:111

                                            @dataclass(frozen=True)                                                                                                                                      # shape/view.py:85
                                            class View:
                                              @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none                                                                           # shape/view.py:267
                                              def reshape(self, new_shape: Tuple[sint, ...]) -> Optional[View]:
                                                if self.shape == new_shape: return self                                                                                                                  # shape/view.py:268
                                                assert all(x >= 0 for x in new_shape), f"shape can't contain negative numbers {new_shape}"                                                               # shape/view.py:270
                                                if 0 in self.shape:                                                                                                                                      # shape/view.py:271
                                                if (self_all_int := all_int(self.shape)):                                                                                                                # shape/view.py:275

                                                  assert all(isinstance(s, (int, Variable)) for s in new_shape), f"{self.shape=} -> {new_shape=} contains non (int, Variable) dim"                       # shape/view.py:276
                                                  if prod(self.shape) != prod([s if isinstance(s, int) else cast(Variable,s).val for s in new_shape]):                                                   # shape/view.py:277

                                                if new_shape == () and self.mask and any(mx==my for (mx,my) in self.mask): return None                                                                   # shape/view.py:280
                                                if self.contiguous: return View.create(new_shape)                                                                                                        # shape/view.py:283

                                      class LazyBuffer:                                                                                                                                                  # lazy.py:26
                                        def _view(self, new_st:ShapeTracker) -> LazyBuffer:                                                                                                              # lazy.py:208
                                          if self.st.size == 0 or (new_st.views[-1].mask is not None and any((x[1]-x[0]) == 0 for x in new_st.views[-1].mask)):                                    # OLD # lazy.py:209
                                          if new_st.contiguous and self.base.shape == new_st.shape: return self.base                                                                               # OLD # lazy.py:211
                                          return create_lazybuffer(self.device, new_st, self.dtype, base=self.base)                                                                                      # lazy.py:212

                                        class LazyBuffer:                                                                                                                                                # lazy.py:26
                                          def __init__(self, device:str, st:ShapeTracker, dtype:DTypeLike,                                                                                               # lazy.py:27
                                                       op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
                                                       base:Optional[LazyBuffer]=None, metadata:Optional[Metadata]=None):
                                            self.device, self.st, self.dtype, self.shape, self.size, self.metadata = device, st, to_dtype(dtype), st.shape, st.size, metadata                      # OLD # lazy.py:30
                                            self._base: Optional[LazyBuffer] = None                                                                                                                # OLD # lazy.py:31
                                            if base is None:                                                                                                                                       # OLD # lazy.py:32
                                              self.op, self.arg, self.srcs = op, arg, srcs  # this is a LazyOp, except the src is LazyBuffers and not LazyOps                                      # OLD # lazy.py:34
                                              assert self.op is not MetaOps.ASSIGN or srcs[1].base.realized is not None, "assign target must be realized"                                          # OLD # lazy.py:35
                                              if self.op is MetaOps.VIEW:                                                                                                                          # OLD # lazy.py:37
                                                self.buffer = srcs[1].base.buffer if self.op is MetaOps.ASSIGN else Buffer(device, self.size, self.dtype)                                          # OLD # lazy.py:41
                                              self.buffer.ref(1)                                                                                                                                   # OLD # lazy.py:42
                                              self.contiguous_child: Optional[Tuple[ReferenceType[LazyBuffer], ShapeTracker]] = None                                                               # OLD # lazy.py:43
                                              self.forced_realize = False                                                                                                                          # OLD # lazy.py:44
                                              assert base.base == base, "base must be a base itself"                                                                                                     # lazy.py:47

                                              self._base = base                                                                                                                                          # lazy.py:48

                          ret._ctx = ctx if ctx.requires_grad and not Tensor.no_grad else None  # used by autograd engine                                                                                # tensor.py:39
                          return ret                                                                                                                                                                     # tensor.py:40

                # NOTE: this is sum in reverse                                                                                                                                                           # function.py:179
                class Expand(Function):
                  def forward(self, x:LazyBuffer, shape:Tuple[int, ...]) -> LazyBuffer:                                                                                                                  # function.py:180
                    self.expanded_axis = tuple(i for i, (si, so) in enumerate(zip(x.shape, shape)) if si != so)                                                                                          # function.py:181
                    return x.expand(shape)                                                                                                                                                               # function.py:182

                      class LazyBuffer:                                                                                                                                                                  # lazy.py:26
                        def expand(self, arg:Tuple[sint, ...]): return self._view(self.st.expand(arg))                                                                                                   # lazy.py:216

                          @dataclass(frozen=True)                                                                                                                                                        # shape/shapetracker.py:10
                          class ShapeTracker:
                            def expand(self, new_shape: Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))                                  # shape/shapetracker.py:106

                              @dataclass(frozen=True)                                                                                                                                                    # shape/view.py:85
                              class View:
                                @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none                                                                                         # shape/view.py:239
                                def expand(self, new_shape: Tuple[sint, ...]) -> View:
                                  if len(new_shape) != len(self.shape): raise ValueError(f"expand arg {new_shape=} must have same number of dimensions as shape {self.shape=}")                          # shape/view.py:240
                                  if 0 in self.shape:                                                                                                                                                    # shape/view.py:241
                                  assert all((s == x or (s == 1 and st == 0)) for s,x,st in zip(self.shape, new_shape, self.strides)), f"can't expand {self.shape} into {new_shape}"                     # shape/view.py:244
                                  mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if s != ns else m) for m,s,ns in zip(self.mask, self.shape, new_shape)]) if self.mask else None                       # shape/view.py:246
                                  return View.create(new_shape, self.strides, self.offset, mask)                                                                                                         # shape/view.py:247

                    class LazyBuffer:                                                                                                                                                                    # lazy.py:26
                      def __del__(self):                                                                                                                                                                 # lazy.py:50
                        if hasattr(self, 'buffer'): self.buffer.ref(-1)                                                                                                                                  # lazy.py:51

            class Add(Function):                                                                                                                                                                         # function.py:118
              def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.ADD, y)                                                                                                  # function.py:119

                class LazyBuffer:                                                                                                                                                                        # lazy.py:26
                  def e(self, op:Union[MetaOps, UnaryOps, BinaryOps, TernaryOps], *in_srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:                                                            # lazy.py:137
                    srcs: List[LazyBuffer] = []                                                                                                                                                          # lazy.py:138
                    for s in (self,)+in_srcs:                                                                                                                                                            # lazy.py:139
                      if s == s.base and s.base.contiguous_child and (root:=s.base.contiguous_child[0]()) is not None:                                                                                   # lazy.py:140

                        srcs.append(s)                                                                                                                                                                   # lazy.py:143

                    assert all_same(dts:=[x.dtype.scalar() for x in (srcs[1:] if op is TernaryOps.WHERE else srcs)]), f"all dtypes must match {dts} on {op}"                                             # lazy.py:144

                      def all_same(items:Union[Tuple[T, ...], List[T]]): return all(x == items[0] for x in items)                                                                                        # helpers.py:26

                    assert all_same([x.shape for x in srcs]), f"all shapes must be the same {[x.shape for x in srcs]}"                                                                                   # lazy.py:145

                    if op is TernaryOps.WHERE: assert srcs[0].dtype == dtypes.bool, "TernaryOps.WHERE must have the first arg be bool"                                                                   # lazy.py:146
                    if op is UnaryOps.NEG: assert srcs[0].dtype != dtypes.bool, "UnaryOps.NEG does not accept dtype bool"                                                                                # lazy.py:147
                    out_dtype = dtypes.bool if op in (BinaryOps.CMPLT, BinaryOps.CMPNE) else srcs[-1].dtype                                                                                              # lazy.py:149
                    if op in python_alu and all(s.is_unrealized_unmasked_const() for s in srcs):                                                                                                         # lazy.py:152

                  class LazyBuffer:                                                                                                                                                                      # lazy.py:26
                    def is_unrealized_unmasked_const(self): return self.is_unrealized_const() and all(v.mask is None for v in self.st.views)                                                             # lazy.py:114

                    if op is UnaryOps.NEG and self.base.op is UnaryOps.NEG and self.base.realized is None: return self.base.srcs[0]                                                                      # lazy.py:154
                    if op in BinaryOps:                                                                                                                                                                  # lazy.py:155
                      x, y = self, in_srcs[0]                                                                                                                                                            # lazy.py:156
                      if op is BinaryOps.ADD:                                                                                                                                                            # lazy.py:157
                        if y.is_unrealized_unmasked_const() and y.base.arg == 0: return x                                                                                                                # lazy.py:158

                        if x.is_unrealized_unmasked_const() and x.base.arg == 0: return y                                                                                                                # lazy.py:159

                      if op is BinaryOps.MUL:                                                                                                                                                            # lazy.py:160
                    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), out_dtype, op, arg, tuple(srcs))                                                                          # lazy.py:166

      _METADATA.reset(token)                                                                                                                                                                             # tensor.py:3252
      return ret                                                                                                                                                                                         # tensor.py:3253

a.tolist()                                                                                                                                                                                               # ...s/tinygrad.tensor.tolist.py:3

  class Tensor:                                                                                                                                                                                          # tensor.py:92
    # TODO: should be Tensor.tolist() -> Union[List[ConstType], ConstType]. The List is Sequence because mypy expects memoryview.tolist() -> list[int]                                                   # tensor.py:278
    # src: https://github.com/python/mypy/blob/release-1.6/mypy/typeshed/stdlib/builtins.pyi#L803
    def tolist(self) -> Union[Sequence[ConstType], ConstType]:
      return self.data().tolist()                                                                                                                                                                        # tensor.py:287

    class Tensor:                                                                                                                                                                                        # tensor.py:92
      def data(self) -> memoryview:                                                                                                                                                                      # tensor.py:250
        assert self.dtype.fmt is not None, f"no fmt dtype for {self.dtype}"                                                                                                                              # tensor.py:259

        assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"                                                                                                                       # tensor.py:260

        return self._data().cast(self.dtype.fmt, self.shape)                                                                                                                                             # tensor.py:261

      class Tensor:                                                                                                                                                                                      # tensor.py:92
        def _data(self) -> memoryview:                                                                                                                                                                   # tensor.py:242
          if 0 in self.shape: return memoryview(bytearray(0))                                                                                                                                            # tensor.py:243

          cpu = self.cast(self.dtype.scalar()).contiguous().to("CLANG").realize()                                                                                                                        # tensor.py:245

        class Tensor:                                                                                                                                                                                    # tensor.py:92
          def cast(self, dtype:DTypeLike) -> Tensor:                                                                                                                                                     # tensor.py:3034
            return self if self.dtype == (dt:=to_dtype(dtype)) else F.Cast.apply(self, dtype=dt)                                                                                                         # tensor.py:3047

        class Tensor:                                                                                                                                                                                    # tensor.py:92
          def contiguous(self):                                                                                                                                                                          # tensor.py:1995
            return F.Contiguous.apply(self)                                                                                                                                                              # tensor.py:1999

          class Contiguous(Function):                                                                                                                                                                    # function.py:11
            def forward(self, x:LazyBuffer) -> LazyBuffer: return x.contiguous()                                                                                                                         # function.py:12

              class LazyBuffer:                                                                                                                                                                          # lazy.py:26
                def contiguous(self, allow_buffer_view=True):                                                                                                                                            # lazy.py:87
                  if not self.st.contiguous or self.size != self.base.size or self.is_unrealized_const():                                                                                                # lazy.py:88

                  self.base.forced_realize = True                                                                                                                                                        # lazy.py:92

                  return self                                                                                                                                                                            # lazy.py:93

        class Tensor:                                                                                                                                                                                    # tensor.py:92
          def to(self, device:Optional[Union[str, Tuple[str, ...]]]) -> Tensor:                                                                                                                          # tensor.py:303
            device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)                                                                 # tensor.py:307

            if device == self.device: return self                                                                                                                                                        # tensor.py:308

        class Tensor:                                                                                                                                                                                    # tensor.py:92
          def realize(self, *lst:Tensor, do_update_stats=True) -> Tensor:                                                                                                                                # tensor.py:202
            run_schedule(*self.schedule_with_vars(*lst), do_update_stats=do_update_stats)                                                                                                                # tensor.py:204

          class Tensor:                                                                                                                                                                                  # tensor.py:92
            def schedule_with_vars(self, *lst:Tensor, seen:Optional[Set[LazyBuffer]]=None) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:                                                            # tensor.py:188
              if getenv("FUZZ_SCHEDULE"):                                                                                                                                                                # tensor.py:190

              schedule, var_vals = create_schedule_with_vars(flatten([x.lazydata.lbs for x in (self,)+lst]), seen)                                                                                       # tensor.py:193

            class LazyBuffer:                                                                                                                                                                            # lazy.py:26
              # same API as multi                                                                                                                                                                        # lazy.py:67
              @property
              def lbs(self) -> List[LazyBuffer]: return [self]

                def flatten(l:Iterable[Iterable[T]]): return [item for sublist in l for item in sublist]                                                                                                 # helpers.py:34

                def create_schedule_with_vars(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:                                             # engine/schedule.py:381
                  if seen is None: seen = set()                                                                                                                                                          # engine/schedule.py:382
                  graph, in_degree = _graph_schedule(outs, seen)                                                                                                                                         # engine/schedule.py:383

                    def _graph_schedule(outs:List[LazyBuffer], seen:Set[LazyBuffer]) -> \                                                                                                                # engine/schedule.py:260
                      Tuple[DefaultDict[LBScheduleItem, List[LBScheduleItem]],  # this is the graph
                            DefaultDict[LBScheduleItem, int]]:                  # this is the in-degree of the graph
                      realizes: Dict[LazyBuffer, None] = {x.base:None for x in outs if x.base.realized is None}                                                                                          # engine/schedule.py:265

                      allbufs: Dict[LazyBuffer, None] = {}                                                                                                                                               # engine/schedule.py:266
                      simple_pads: Dict[LazyBuffer, None] = {}                                                                                                                                           # engine/schedule.py:267
                      children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)                                                                                                      # engine/schedule.py:268
                      assign_targets: Dict[LazyBuffer, LazyBuffer] = {}                                                                                                                                  # engine/schedule.py:269
                      double_reduces: Dict[LazyBuffer, None] = {}                                                                                                                                        # engine/schedule.py:270
                      for out in outs: _recurse_lb(out.base, realizes, allbufs, simple_pads, children, assign_targets, double_reduces, scheduled=True)                                                   # engine/schedule.py:271

                        def _recurse_lb(buf:LazyBuffer, realizes:Dict[LazyBuffer, None], allbufs:Dict[LazyBuffer, None], simple_pads:Dict[LazyBuffer, None],                                             # engine/schedule.py:185
                                        children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], assign_targets:Dict[LazyBuffer, LazyBuffer],
                                        double_reduces:Dict[LazyBuffer, None], scheduled=False) -> None:
                          if buf in allbufs or buf.base.realized is not None: return                                                                                                                     # engine/schedule.py:189

                          if GRAPH: log_lazybuffer(buf, scheduled)                                                                                                                                       # engine/schedule.py:190

                          if buf is not buf.base:                                                                                                                                                        # engine/schedule.py:192

                          if buf.op in ReduceOps and buf.srcs[0].base.op is buf.op and buf.srcs[0] is not buf.srcs[0].base: double_reduces[buf] = None                                                   # engine/schedule.py:206
                          allbufs[buf] = None                                                                                                                                                            # engine/schedule.py:207
                          if buf.forced_realize or buf.op in MetaOps: realizes[buf] = None                                                                                                               # engine/schedule.py:208
                          if buf.op is MetaOps.ASSIGN:                                                                                                                                                   # engine/schedule.py:209
                          if buf.op is MetaOps.COPY:                                                                                                                                                     # engine/schedule.py:213
                          if buf.op is MetaOps.VIEW: realizes[buf.srcs[0].base] = None                                                                                                                   # engine/schedule.py:216
                          for x in buf.srcs:                                                                                                                                                             # engine/schedule.py:217
                            if x.base.realized is None: children[x.base][buf] = None                                                                                                                     # engine/schedule.py:218

                            _recurse_lb(x, realizes, allbufs, simple_pads, children, assign_targets, double_reduces)                                                                                     # engine/schedule.py:219

                              def _recurse_lb(buf:LazyBuffer, realizes:Dict[LazyBuffer, None], allbufs:Dict[LazyBuffer, None], simple_pads:Dict[LazyBuffer, None],                                       # engine/schedule.py:185
                                              children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], assign_targets:Dict[LazyBuffer, LazyBuffer],
                                              double_reduces:Dict[LazyBuffer, None], scheduled=False) -> None:
                                if buf in allbufs or buf.base.realized is not None: return                                                                                                         # OLD # engine/schedule.py:189
                                if GRAPH: log_lazybuffer(buf, scheduled)                                                                                                                           # OLD # engine/schedule.py:190
                                if buf is not buf.base:                                                                                                                                            # OLD # engine/schedule.py:192
                                if buf.op in ReduceOps and buf.srcs[0].base.op is buf.op and buf.srcs[0] is not buf.srcs[0].base: double_reduces[buf] = None                                       # OLD # engine/schedule.py:206
                                allbufs[buf] = None                                                                                                                                                # OLD # engine/schedule.py:207
                                if buf.forced_realize or buf.op in MetaOps: realizes[buf] = None                                                                                                   # OLD # engine/schedule.py:208
                                if buf.op is MetaOps.ASSIGN:                                                                                                                                       # OLD # engine/schedule.py:209
                                if buf.op is MetaOps.COPY:                                                                                                                                         # OLD # engine/schedule.py:213
                                  assert buf.srcs[0].st.contiguous and buf.srcs[0].size == buf.srcs[0].base.size, "can only copy contig"                                                                 # engine/schedule.py:214

                                  realizes[buf.srcs[0].base] = None                                                                                                                                      # engine/schedule.py:215

                              def _recurse_lb(buf:LazyBuffer, realizes:Dict[LazyBuffer, None], allbufs:Dict[LazyBuffer, None], simple_pads:Dict[LazyBuffer, None],                                       # engine/schedule.py:185
                                              children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], assign_targets:Dict[LazyBuffer, LazyBuffer],
                                              double_reduces:Dict[LazyBuffer, None], scheduled=False) -> None:
                                if buf in allbufs or buf.base.realized is not None: return                                                                                                         # OLD # engine/schedule.py:189
                                if GRAPH: log_lazybuffer(buf, scheduled)                                                                                                                           # OLD # engine/schedule.py:190
                                if buf is not buf.base:                                                                                                                                            # OLD # engine/schedule.py:192
                                  if len(buf.st.views) == 1 and buf.st.views[-1].mask is not None and all_int(buf.base.st.shape) and \                                                                   # engine/schedule.py:194
                                      prod(buf.base.st.shape) >= prod([y-x for x,y in buf.st.views[-1].mask]):
                                  elif prod(buf.base.st.shape) < prod(buf.st.shape):                                                                                                                     # engine/schedule.py:198

                                    if buf.base.op is UnaryOps.CAST and isinstance(buf.base.srcs[0].dtype, ImageDType) and isinstance(buf.base.arg, ImageDType):                                         # engine/schedule.py:200

                                    else: realizes[buf.base] = None                                                                                                                                      # engine/schedule.py:202

                                  return _recurse_lb(buf.base, realizes, allbufs, simple_pads, children, assign_targets, double_reduces)                                                                 # engine/schedule.py:205

                      for p in simple_pads:                                                                                                                                                              # engine/schedule.py:274
                      reduce_for_op: Dict[LazyBuffer, LazyBuffer] = {}                                                                                                                                   # engine/schedule.py:279
                      reduce_of_const: List[LazyBuffer] = []                                                                                                                                             # engine/schedule.py:280
                      for r in allbufs:                                                                                                                                                                  # engine/schedule.py:281
                        if r.op not in ReduceOps or r in realizes: continue                                                                                                                              # engine/schedule.py:282
                      if FUSE_CONV_BW:                                                                                                                                                                   # engine/schedule.py:322

                      for r in reduce_of_const:                                                                                                                                                          # engine/schedule.py:327
                      output_groups: DefaultDict[LazyBuffer, List[LazyBuffer]] = defaultdict(list)                                                                                                       # engine/schedule.py:336
                      for buf in realizes:                                                                                                                                                               # engine/schedule.py:337
                        if buf.realized is not None or buf.op is MetaOps.CONST or buf in seen: continue                                                                                                  # engine/schedule.py:338

                        output_groups[reduce_for_op[buf] if buf in reduce_for_op and MULTIOUTPUT else buf].append(buf)                                                                                   # engine/schedule.py:339
                        if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or                                                                                            # engine/schedule.py:342
                                                                  not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):

                      prescheduled = [_lower_lazybuffer(group, realizes) for group in output_groups.values()]                                                                                            # engine/schedule.py:353

                      def _lower_lazybuffer(outs:List[LazyBuffer], realizes:Dict[LazyBuffer, None]) -> LBScheduleItem:                                                                                   # engine/schedule.py:145
                        if (out:=outs[0]).op is MetaOps.COPY and getenv("USE_COPY_KERNEL") and out.device.split(":")[0] == out.srcs[0].device.split(":")[0]:                                             # engine/schedule.py:147
                        if out.op in {MetaOps.CUSTOM, MetaOps.COPY, MetaOps.EMPTY, MetaOps.VIEW}:                                                                                                        # engine/schedule.py:151
                        reduce_info: Dict[Tuple[LazyBuffer, ShapeTracker], Tuple[ShapeTracker, Tuple[int, ...]]] = {}                                                                                    # engine/schedule.py:154
                        seen_ops: Dict[Tuple[LazyBuffer, ShapeTracker], Optional[Tuple[LazyBuffer, ShapeTracker]]] = {}                                                                                  # engine/schedule.py:155
                        for out in outs: _recurse_reduceops(out, out.st, realizes, outs, reduce_info, seen_ops)                                                                                          # engine/schedule.py:156

                          def _recurse_reduceops(buf:LazyBuffer, st:ShapeTracker, realizes:Dict[LazyBuffer, None], outs:List[LazyBuffer],                                                                # engine/schedule.py:105
                                                 reduce_info:Dict[Tuple[LazyBuffer, ShapeTracker], Tuple[ShapeTracker, Tuple[int, ...]]],
                                                 cache:Dict[Tuple[LazyBuffer, ShapeTracker], Optional[Tuple[LazyBuffer, ShapeTracker]]]) -> \
                                                   Optional[Tuple[LazyBuffer, ShapeTracker]]:
                            if (buf, st) in cache: return cache[(buf, st)]                                                                                                                               # engine/schedule.py:109
                            if buf.base.realized is not None or (buf.base in realizes and buf.base not in outs): return None                                                                             # engine/schedule.py:110

                            if buf is not buf.base: st, buf = buf.st+st, buf.base                                                                                                                        # engine/schedule.py:111

                            input_st = ShapeTracker.from_shape(buf.srcs[0].shape) if buf.op in ReduceOps else st                                                                                         # engine/schedule.py:112
                            reduce_srcs = [r for x in buf.srcs if (r:=_recurse_reduceops(x, input_st, realizes, outs, reduce_info, cache)) is not None]                                                  # engine/schedule.py:113

                            top_reduce = reduce_srcs[-1] if len(reduce_srcs) != 0 else None                                                                                                              # engine/schedule.py:114
                            if buf.op in ReduceOps:                                                                                                                                                      # engine/schedule.py:115
                            return cache.setdefault((buf, st), top_reduce)                                                                                                                               # engine/schedule.py:143

                        shape_dims = [sorted(dedup(dims)) for dims in zip(*[input_st.shape for input_st,_ in reduce_info.values()])]                                                                     # engine/schedule.py:158
                        for i,dims in enumerate(shape_dims):                                                                                                                                             # engine/schedule.py:159
                        var_vals = merge_dicts([out.st.var_vals.copy() for out in outs])                                                                                                                 # engine/schedule.py:166

                        @dataclass(frozen=True)                                                                                                                                                          # shape/shapetracker.py:10
                        class ShapeTracker:
                          @property                                                                                                                                                                      # shape/shapetracker.py:53
                          def var_vals(self) -> Dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

                            @dataclass(frozen=True)                                                                                                                                                      # shape/shapetracker.py:10
                            class ShapeTracker:
                              def vars(self) -> Set[Variable]: return set().union(*[v.vars() for v in self.views])                                                                                       # shape/shapetracker.py:50

                              @dataclass(frozen=True)                                                                                                                                                    # shape/view.py:85
                              class View:
                                @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none                                                                                                 # shape/view.py:120
                                def vars(self) -> Set[Variable]:
                                  flatten_mask = tuple(x for m in self.mask for x in m) if self.mask is not None else tuple()                                                                            # shape/view.py:121
                                  return functools.reduce(operator.or_, [x.vars() for x in self.shape+self.strides+(self.offset,)+flatten_mask if isinstance(x, Node)], set())                           # shape/view.py:122

                            def merge_dicts(ds:Iterable[Dict[T,U]]) -> Dict[T,U]:                                                                                                                        # helpers.py:41
                              assert len(kvs:=set([(k,v) for d in ds for k,v in d.items()])) == len(set(kv[0] for kv in kvs)), f"cannot merge, {kvs} contains different values for the same key"  # noqa: E501 # helpers.py:42
                              return {k:v for d in ds for k,v in d.items()}                                                                                                                              # helpers.py:43

                        assign_targets = {x.srcs[1]:x for x in outs if x.op is MetaOps.ASSIGN}                                                                                                           # engine/schedule.py:167
                        cache: Dict[Tuple[LazyBuffer, ShapeTracker], LazyOp] = {}                                                                                                                        # engine/schedule.py:168
                        ast: List[LazyOp] = []                                                                                                                                                           # engine/schedule.py:169
                        inputs: Dict[LazyBuffer, int] = {}                                                                                                                                               # engine/schedule.py:170
                        for i, out in enumerate(outs):                                                                                                                                                   # engine/schedule.py:171
                          output_st = ShapeTracker.from_shape(reduce_st(*deque(reduce_info.values(), 1).pop()) if reduce_info else out.shape)                                                            # engine/schedule.py:172

                          lop = _recursive_lazyop(out, output_st, tuple(outs), var_vals, inputs, realizes, assign_targets, reduce_info, cache=cache)                                                     # engine/schedule.py:173

                            def _recursive_lazyop(buf:LazyBuffer, st:ShapeTracker, outputs:Tuple[LazyBuffer, ...], var_vals:Dict[Variable, int], inputs:Dict[LazyBuffer, int],                           # engine/schedule.py:51
                                                  realizes:Dict[LazyBuffer, None], assign_targets:Dict[LazyBuffer, LazyBuffer],
                                                  reduce_info:Dict[Tuple[LazyBuffer, ShapeTracker], Tuple[ShapeTracker, Tuple[int, ...]]],
                                                  cache:Dict[Tuple[LazyBuffer, ShapeTracker], LazyOp]) -> LazyOp:
                              if buf is not buf.base: st, buf = buf.st+st, buf.base                                                                                                                      # engine/schedule.py:56

                              if (buf, st) in cache: return cache[(buf, st)]                                                                                                                             # engine/schedule.py:57
                              arg = buf.arg                                                                                                                                                              # engine/schedule.py:58
                              if buf.op is MetaOps.CONST:                                                                                                                                                # engine/schedule.py:61
                              if buf.realized is not None or (buf in realizes and buf not in outputs):                                                                                                   # engine/schedule.py:71

                              if buf.op in {MetaOps.CONTIGUOUS, MetaOps.ASSIGN}:                                                                                                                         # engine/schedule.py:84
                              if buf.op in ReduceOps:                                                                                                                                                    # engine/schedule.py:89
                              return cache.setdefault((buf, st), LazyOp(cast(Op,buf.op), tuple(_recursive_lazyop(x, st, outputs, var_vals, inputs, realizes, assign_targets, \                           # engine/schedule.py:97
                                  reduce_info, cache) for x in buf.srcs), arg))

                              def _recursive_lazyop(buf:LazyBuffer, st:ShapeTracker, outputs:Tuple[LazyBuffer, ...], var_vals:Dict[Variable, int], inputs:Dict[LazyBuffer, int],                         # engine/schedule.py:51
                                                    realizes:Dict[LazyBuffer, None], assign_targets:Dict[LazyBuffer, LazyBuffer],
                                                    reduce_info:Dict[Tuple[LazyBuffer, ShapeTracker], Tuple[ShapeTracker, Tuple[int, ...]]],
                                                    cache:Dict[Tuple[LazyBuffer, ShapeTracker], LazyOp]) -> LazyOp:
                                if buf is not buf.base: st, buf = buf.st+st, buf.base                                                                                                              # OLD # engine/schedule.py:56
                                if (buf, st) in cache: return cache[(buf, st)]                                                                                                                     # OLD # engine/schedule.py:57
                                arg = buf.arg                                                                                                                                                      # OLD # engine/schedule.py:58
                                if buf.op is MetaOps.CONST:                                                                                                                                        # OLD # engine/schedule.py:61
                                if buf.realized is not None or (buf in realizes and buf not in outputs):                                                                                           # OLD # engine/schedule.py:71
                                  unbound_st, st_var_vals = st.simplify().unbind()                                                                                                                       # engine/schedule.py:72

                                    @dataclass(frozen=True)                                                                                                                                              # shape/shapetracker.py:10
                                    class ShapeTracker:
                                      def simplify(self) -> ShapeTracker:                                                                                                                                # shape/shapetracker.py:97
                                        if len(self.views) >= 2 and (new_view := self.views[-2] + self.views[-1]) is not None:                                                                           # shape/shapetracker.py:98
                                        return self                                                                                                                                                      # shape/shapetracker.py:100

                                    @dataclass(frozen=True)                                                                                                                                              # shape/shapetracker.py:10
                                    class ShapeTracker:
                                      def unbind(self) -> Tuple[ShapeTracker, Dict[Variable, int]]:                                                                                                      # shape/shapetracker.py:55
                                        unbound_views, var_vals = zip(*[v.unbind() for v in self.views])                                                                                                 # shape/shapetracker.py:56

                                      @dataclass(frozen=True)                                                                                                                                            # shape/view.py:85
                                      class View:
                                        @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none                                                                                         # shape/view.py:125
                                        def unbind(self) -> Tuple[View, Dict[Variable, int]]:
                                          var_unboundvar_val = [(v, v.unbind()) for v in self.vars()]                                                                                                    # shape/view.py:126
                                          unbound_vars = {v:uv for v,(uv,_) in var_unboundvar_val}                                                                                                       # shape/view.py:127
                                          new_shape = tuple(map(substitute, self.shape))                                                                                                                 # shape/view.py:129

                                            @dataclass(frozen=True)                                                                                                                                      # shape/view.py:85
                                            class View:
                                              @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none                                                                                   # shape/view.py:125
                                              def unbind(self) -> Tuple[View, Dict[Variable, int]]:
                                                def substitute(x): return x if isinstance(x, int) else x.substitute(unbound_vars)                                                                        # shape/view.py:128

                                          new_strides = tuple(map(substitute, self.strides))                                                                                                             # shape/view.py:130

                                          new_offset = substitute(self.offset)                                                                                                                           # shape/view.py:131

                                          new_mask = tuple((substitute(x[0]), substitute(x[1])) for x in self.mask) if self.mask is not None else None                                                   # shape/view.py:132
                                          return View.create(new_shape, new_strides, new_offset, new_mask), dict(x[1] for x in var_unboundvar_val)                                                       # shape/view.py:133

                                        return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)                                                                                                 # shape/shapetracker.py:57

                                  var_vals.update(st_var_vals)                                                                                                                                           # engine/schedule.py:73
                                  if buf in assign_targets:                                                                                                                                              # engine/schedule.py:74
                                  return LazyOp(BufferOps.LOAD, (), MemBuffer(len(outputs)+inputs.setdefault(buf, len(inputs)), buf.dtype, unbound_st))                                                  # engine/schedule.py:81

                              @dataclass(frozen=True)                                                                                                                                                    # shape/shapetracker.py:10
                              class ShapeTracker:
                                def __add__(self, st:ShapeTracker) -> ShapeTracker:                                                                                                                      # shape/shapetracker.py:13
                                  ret = self                                                                                                                                                             # shape/shapetracker.py:14
                                  for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification                                                        # shape/shapetracker.py:15

                                @dataclass(frozen=True)                                                                                                                                                  # shape/view.py:85
                                class View:
                                  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none                                                                                       # shape/view.py:136
                                  def __add__(self, vm1:View) -> Optional[View]:
                                    vm2 = self                                                                                                                                                           # shape/view.py:137
                                    if vm2.contiguous: return vm1                                                                                                                                        # shape/view.py:138
                                    if vm1.contiguous and vm1.shape == vm2.shape: return vm2                                                                                                             # shape/view.py:139

                                    @dataclass(frozen=True)                                                                                                                                              # shape/shapetracker.py:10
                                    class ShapeTracker:
                                      def simplify(self) -> ShapeTracker:                                                                                                                                # shape/shapetracker.py:97
                                        if len(self.views) >= 2 and (new_view := self.views[-2] + self.views[-1]) is not None:                                                                     # OLD # shape/shapetracker.py:98
                                          return ShapeTracker(self.views[:-2] + (new_view,)).simplify()                                                                                                  # shape/shapetracker.py:99

                                  return ret                                                                                                                                                             # shape/shapetracker.py:16

                              def _recursive_lazyop(buf:LazyBuffer, st:ShapeTracker, outputs:Tuple[LazyBuffer, ...], var_vals:Dict[Variable, int], inputs:Dict[LazyBuffer, int],                         # engine/schedule.py:51
                                                    realizes:Dict[LazyBuffer, None], assign_targets:Dict[LazyBuffer, LazyBuffer],
                                                    reduce_info:Dict[Tuple[LazyBuffer, ShapeTracker], Tuple[ShapeTracker, Tuple[int, ...]]],
                                                    cache:Dict[Tuple[LazyBuffer, ShapeTracker], LazyOp]) -> LazyOp:
                                if buf is not buf.base: st, buf = buf.st+st, buf.base                                                                                                              # OLD # engine/schedule.py:56
                                if (buf, st) in cache: return cache[(buf, st)]                                                                                                                     # OLD # engine/schedule.py:57
                                arg = buf.arg                                                                                                                                                      # OLD # engine/schedule.py:58
                                if buf.op is MetaOps.CONST:                                                                                                                                        # OLD # engine/schedule.py:61
                                  unbound_st, st_var_vals = st.simplify().unbind()                                                                                                                       # engine/schedule.py:62

                                  var_vals.update(st_var_vals)                                                                                                                                           # engine/schedule.py:63
                                  if isinstance(arg, Variable):                                                                                                                                          # engine/schedule.py:64
                                  else: assert isinstance(arg, get_args(ConstType)), f"cannot create ConstBuffer with value {arg}"                                                                       # engine/schedule.py:67
                                  return LazyOp(BufferOps.CONST, (), ConstBuffer(arg, buf.dtype, unbound_st))                                                                                            # engine/schedule.py:68

                          if out.op is MetaOps.ASSIGN and out.arg:                                                                                                                                       # engine/schedule.py:174
                          output_st, vv = output_st.simplify().unbind()                                                                                                                                  # engine/schedule.py:177

                          if vv: var_vals.update(vv)                                                                                                                                                     # engine/schedule.py:178
                          ast.append(LazyOp(BufferOps.STORE, (lop,), MemBuffer(i, out.dtype, output_st)))                                                                                                # engine/schedule.py:179
                        return LBScheduleItem(LazyOp(MetaOps.KERNEL, tuple(ast)), outs, list(inputs), var_vals,                                                                                          # engine/schedule.py:180
                                              dedup([x[0].metadata for x in cache if x[0].metadata and x[0] not in inputs]))

                          def dedup(x:Iterable[T]): return list(dict.fromkeys(x))   # retains list order                                                                                                 # helpers.py:19

                            @dataclass(frozen=True)                                                                                                                                                      # helpers.py:115
                            class Metadata:
                              def __hash__(self): return hash(self.name)                                                                                                                                 # helpers.py:119

                      def _lower_lazybuffer(outs:List[LazyBuffer], realizes:Dict[LazyBuffer, None]) -> LBScheduleItem:                                                                                   # engine/schedule.py:145
                        if (out:=outs[0]).op is MetaOps.COPY and getenv("USE_COPY_KERNEL") and out.device.split(":")[0] == out.srcs[0].device.split(":")[0]:                                       # OLD # engine/schedule.py:147
                        if out.op in {MetaOps.CUSTOM, MetaOps.COPY, MetaOps.EMPTY, MetaOps.VIEW}:                                                                                                  # OLD # engine/schedule.py:151
                          return LBScheduleItem(LazyOp(out.op, (), out.arg), outs, [x.base for x in out.srcs])                                                                                           # engine/schedule.py:152

                      schedule_targets = {out:lsi for lsi in prescheduled for out in lsi.outputs}                                                                                                        # engine/schedule.py:354
                      graph: DefaultDict[LBScheduleItem, List[LBScheduleItem]] = defaultdict(list)                                                                                                       # engine/schedule.py:356
                      in_degree: DefaultDict[LBScheduleItem, int] = defaultdict(int)                                                                                                                     # engine/schedule.py:357
                      for lsi in prescheduled:                                                                                                                                                           # engine/schedule.py:358
                        if lsi not in in_degree: in_degree[lsi] = 0                                                                                                                                      # engine/schedule.py:359

                          @dataclass(frozen=True)                                                                                                                                                        # engine/schedule.py:39
                          class LBScheduleItem:
                            def __hash__(self):                                                                                                                                                          # engine/schedule.py:45
                              return hash(self.outputs[0])                                                                                                                                               # engine/schedule.py:47

                        scheduled_parents = dedup(schedule_targets[x] for x in lsi.inputs if x in schedule_targets)                                                                                      # engine/schedule.py:361

                        for x in scheduled_parents:                                                                                                                                                      # engine/schedule.py:362
                          graph[x].append(lsi)                                                                                                                                                           # engine/schedule.py:363

                          in_degree[lsi] += 1                                                                                                                                                            # engine/schedule.py:364

                        parents_assigns = dedup(schedule_targets[assign_targets[x]] for x in lsi.inputs if x in assign_targets)                                                                          # engine/schedule.py:366

                        for assign in parents_assigns:                                                                                                                                                   # engine/schedule.py:367

                      if SAVE_SCHEDULE:                                                                                                                                                                  # engine/schedule.py:371

                      return graph, in_degree                                                                                                                                                            # engine/schedule.py:377

                  if getenv("RUN_PROCESS_REPLAY") and getenv("COMPARE_SCHEDULE", 1):                                                                                                                     # engine/schedule.py:384

                  queue = deque(lsi for lsi,deg in in_degree.items() if deg == 0)                                                                                                                        # engine/schedule.py:388
                  schedule: List[ScheduleItem] = []                                                                                                                                                      # engine/schedule.py:389
                  var_vals: Dict[Variable, int] = {}                                                                                                                                                     # engine/schedule.py:390
                  kernel_number = GlobalCounters.kernel_count                                                                                                                                            # engine/schedule.py:391
                  while queue:                                                                                                                                                                           # engine/schedule.py:392
                    lsi = queue.popleft()                                                                                                                                                                # engine/schedule.py:393
                    for buf in lsi.outputs: seen.add(buf)                                                                                                                                                # engine/schedule.py:394
                    if GRAPH:                                                                                                                                                                            # engine/schedule.py:395

                    var_vals = merge_dicts([var_vals, lsi.var_vals])                                                                                                                                     # engine/schedule.py:398

                    for out in lsi.outputs: del out.srcs  # can only schedule once                                                                                                                       # engine/schedule.py:399
                    schedule.append(si:=ScheduleItem(lsi.ast, tuple(x.buffer for x in lsi.outputs+lsi.inputs if x.size != 0), lsi.metadata))                                                             # engine/schedule.py:400
                    if logops and si.ast.op is MetaOps.KERNEL and not any(i.device.startswith("DISK:") for i in si.inputs): logops.write(str(si.ast)+"\n")                                               # engine/schedule.py:401
                    for x in graph[lsi]:                                                                                                                                                                 # engine/schedule.py:402

                      in_degree[x] -= 1                                                                                                                                                                  # engine/schedule.py:403

                      if in_degree[x] == 0: queue.append(x)                                                                                                                                              # engine/schedule.py:404

                        class Buffer:                                                                                                                                                                    # device.py:52
                          def __del__(self):                                                                                                                                                             # device.py:100
                            if not hasattr(self, '_buf'): return                                                                                                                                         # device.py:101

                  if any(degree != 0 for degree in in_degree.values()) or len(in_degree) != len(schedule):                                                                                               # engine/schedule.py:407
                  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")                                                                                                     # engine/schedule.py:409

                  return schedule, var_vals                                                                                                                                                              # engine/schedule.py:410

              return memory_planner(schedule), var_vals                                                                                                                                                  # tensor.py:194

                def memory_planner(schedule:List[ScheduleItem]) -> List[ScheduleItem]:                                                                                                                   # engine/realize.py:264
                  assigned = _internal_memory_planner([si.bufs for si in schedule],                                                                                                                      # engine/realize.py:266
                                                      noopt_buffers={b for si in schedule if si.ast.op is not MetaOps.KERNEL for b in si.bufs})

                    def _internal_memory_planner(buffers:List[Union[List[Buffer], Tuple[Buffer, ...]]], noopt_buffers=None, debug_prefix="") -> Dict[Buffer, Buffer]:                                    # engine/realize.py:227
                      if getenv("NO_MEMORY_PLANNER"): return {}                                                                                                                                          # engine/realize.py:228

                      first_appearance, last_appearance = {}, {}                                                                                                                                         # engine/realize.py:229
                      for i,u in enumerate(buffers):                                                                                                                                                     # engine/realize.py:230
                        for buf in u:                                                                                                                                                                    # engine/realize.py:231
                          if buf.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue                                                            # engine/realize.py:232

                            class Buffer:                                                                                                                                                                # device.py:52
                              def is_allocated(self) -> bool: return hasattr(self, '_buf')                                                                                                               # device.py:76

                            class Buffer:                                                                                                                                                                # device.py:52
                              @property                                                                                                                                                                  # device.py:74
                              def lb_refcount(self): return self.base._lb_refcount

                      free_segs: Dict[Tuple, List[Tuple[int, int, Buffer]]] = defaultdict(list) # Dict[buffer key, Tuple[start, end, buffer to reuse on the seg]]                                        # engine/realize.py:238
                      buffer_requests = sorted([(first_appearance[buf], last_appearance[buf], buf) for buf in first_appearance.keys()], key=lambda x: -x[2].nbytes)                                      # engine/realize.py:250
                      assigned = {buf:find_replace_buffer(buf, st, en) for st, en, buf in buffer_requests}                                                                                               # engine/realize.py:251
                      for i,u in enumerate(buffers):                                                                                                                                                     # engine/realize.py:253
                        for buf in u:                                                                                                                                                                    # engine/realize.py:254
                          if buf.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue                                                            # engine/realize.py:255

                      if DEBUG >= 1 and len(ak:=dedup(x for x in assigned.keys() if x._base is None)) != len(av:=dedup(x for x in assigned.values() if x._base is None)):                                # engine/realize.py:259

                      return assigned                                                                                                                                                                    # engine/realize.py:262

                  return [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.bufs), si.metadata) for si in schedule]                                                                              # engine/realize.py:268

              def run_schedule(schedule:List[ScheduleItem], var_vals:Optional[Dict[Variable, int]]=None, do_update_stats=True):                                                                          # engine/realize.py:220
                for ei in lower_schedule(schedule):                                                                                                                                                      # engine/realize.py:221

                  def lower_schedule(schedule:List[ScheduleItem]) -> Generator[ExecItem, None, None]:                                                                                                    # engine/realize.py:205
                    while len(schedule):                                                                                                                                                                 # engine/realize.py:206
                      si = schedule.pop(0)                                                                                                                                                               # engine/realize.py:207
                      try: yield lower_schedule_item(si)                                                                                                                                                 # engine/realize.py:208

                        def lower_schedule_item(si:ScheduleItem) -> ExecItem:                                                                                                                            # engine/realize.py:189
                          assert len(set(x.device for x in si.bufs)) == 1 or si.ast.op is MetaOps.COPY or getenv("USE_COPY_KERNEL")                                                                      # engine/realize.py:190
                          if si.ast.op is MetaOps.KERNEL:                                                                                                                                                # engine/realize.py:191
                          out = si.outputs[0]                                                                                                                                                            # engine/realize.py:194

                            @dataclass(frozen=True)                                                                                                                                                      # engine/schedule.py:25
                            class ScheduleItem:
                              @property                                                                                                                                                                  # engine/schedule.py:30
                              def outputs(self) -> Tuple[Buffer, ...]:
                                return self.bufs[:len(self.ast.src)] if self.ast.op is MetaOps.KERNEL else self.bufs[0:1]                                                                                # engine/schedule.py:32

                          if si.ast.op is MetaOps.COPY:                                                                                                                                                  # engine/realize.py:195
                            kernel_type = BufferCopy                                                                                                                                                     # engine/realize.py:196
                            if hasattr(Device[out.device].allocator, 'transfer') and out.device.split(":")[0] == si.inputs[0].device.split(":")[0]:                                                      # engine/realize.py:197

                        import ctypes, subprocess, pathlib, tempfile                                                                                                                                     # runtime/ops_clang.py:1
                        from tinygrad.device import Compiled, Compiler, MallocAllocator                                                                                                                  # runtime/ops_clang.py:2
                        from tinygrad.helpers import cpu_time_execution, DEBUG, cpu_objdump                                                                                                              # runtime/ops_clang.py:3
                        from tinygrad.renderer.cstyle import ClangRenderer                                                                                                                               # runtime/ops_clang.py:4
                        from typing import Dict, List, Optional, Tuple, Union, DefaultDict, cast, Literal, Callable                                                                                      # renderer/cstyle.py:1
                        import os, math                                                                                                                                                                  # renderer/cstyle.py:2
                        from collections import defaultdict, Counter                                                                                                                                     # renderer/cstyle.py:3
                        from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps                                                                                                                         # renderer/cstyle.py:4
                        from tinygrad.helpers import strip_parens, getenv, prod, dedup                                                                                                                   # renderer/cstyle.py:5
                        from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType, ConstType                                                                                                        # renderer/cstyle.py:6
                        from tinygrad.codegen.uops import UOps, UOp                                                                                                                                      # renderer/cstyle.py:7
                        from tinygrad.renderer import Renderer, TensorCore                                                                                                                               # renderer/cstyle.py:8

                        class CStyleLanguage(Renderer):                                                                                                                                                  # renderer/cstyle.py:10
                          kernel_prefix: str = ""                                                                                                                                                        # renderer/cstyle.py:11
                          buffer_prefix: str = ""                                                                                                                                                        # renderer/cstyle.py:12
                          buffer_suffix: str = ""                                                                                                                                                        # renderer/cstyle.py:13
                          smem_align: str = ""                                                                                                                                                           # renderer/cstyle.py:14
                          smem_prefix: str = ""                                                                                                                                                          # renderer/cstyle.py:15
                          smem_prefix_for_cast: bool = True                                                                                                                                              # renderer/cstyle.py:16
                          arg_int_prefix: str = "const int"                                                                                                                                              # renderer/cstyle.py:17
                          barrier: str = ""                                                                                                                                                              # renderer/cstyle.py:18
                          code_for_workitem: Dict[Union[Literal["g"], Literal["l"], Literal["i"]], Callable] = {}                                                                                        # renderer/cstyle.py:19
                          extra_args: List[str] = []                                                                                                                                                     # renderer/cstyle.py:20
                          float4: Optional[str] = None                                                                                                                                                   # renderer/cstyle.py:21
                          uses_vload: bool = False                                                                                                                                                       # renderer/cstyle.py:22
                          uses_ptr_arithmetic: bool = False                                                                                                                                              # renderer/cstyle.py:23
                          type_map: Dict[DType, str] = {}                                                                                                                                                # renderer/cstyle.py:24
                          infinity: str = "INFINITY"                                                                                                                                                     # renderer/cstyle.py:25
                          nan: str = "NAN"                                                                                                                                                               # renderer/cstyle.py:26
                          code_for_op: Dict = {                                                                                                                                                          # renderer/cstyle.py:27
                            UnaryOps.NEG: lambda x,dtype: f"(!{x})" if dtype == dtypes.bool else f"(-{x})", UnaryOps.SQRT: lambda x,dtype: f"sqrt({x})",
                            UnaryOps.RECIP: lambda x,dtype: f"(1/{x})",
                            UnaryOps.EXP2: lambda x,dtype: f"exp2({x})", UnaryOps.LOG2: lambda x,dtype: f"log2({x})", UnaryOps.SIN: lambda x,dtype: f"sin({x})",
                            BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.MAX: lambda a,b,dtype: f"max({a},{b})",
                            BinaryOps.IDIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MUL: lambda a,b,dtype: f"({a}*{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
                            BinaryOps.CMPLT: lambda a,b,dtype: f"({a}<{b})", BinaryOps.CMPNE: lambda a,b,dtype: f"({a}!={b})", BinaryOps.XOR: lambda a,b,dtype: f"({a}^{b})",
                            BinaryOps.AND: lambda a,b,dtype: f"({a}&{b})", BinaryOps.OR: lambda a,b,dtype: f"({a}|{b})",
                            TernaryOps.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})"}

                        class ClangRenderer(CStyleLanguage):                                                                                                                                             # renderer/cstyle.py:194
                          device = "CLANG"                                                                                                                                                               # renderer/cstyle.py:195
                          float4 = "(float4)"                                                                                                                                                            # renderer/cstyle.py:196
                          has_local = False                                                                                                                                                              # renderer/cstyle.py:197
                          global_max = None                                                                                                                                                              # renderer/cstyle.py:198
                          infinity = "__builtin_inff()"                                                                                                                                                  # renderer/cstyle.py:199
                          nan = '__builtin_nanf("")'                                                                                                                                                     # renderer/cstyle.py:200
                          buffer_suffix = " restrict"                                                                                                                                                    # renderer/cstyle.py:203
                          type_map = {dtypes.bool:"_Bool", dtypes.half:"__fp16"}                                                                                                                         # renderer/cstyle.py:204
                          code_for_op = {**CStyleLanguage().code_for_op,                                                                                                                                 # renderer/cstyle.py:205
                                         UnaryOps.SQRT: lambda x,dtype: f"__builtin_sqrtl({x})" if dtype == dtypes.float64 else f"__builtin_sqrtf({x})",
                                         BinaryOps.MAX: lambda a,b,dtype: f"(({a}>{b})?{a}:{b})"}

                        class OpenCLRenderer(CStyleLanguage):                                                                                                                                            # renderer/cstyle.py:213
                          device = "GPU"                                                                                                                                                                 # renderer/cstyle.py:214
                          kernel_prefix = "__kernel "                                                                                                                                                    # renderer/cstyle.py:217
                          buffer_prefix = "__global "                                                                                                                                                    # renderer/cstyle.py:218
                          smem_align = "__attribute__ ((aligned (16))) "                                                                                                                                 # renderer/cstyle.py:219
                          smem_prefix = "__local "                                                                                                                                                       # renderer/cstyle.py:220
                          barrier = "barrier(CLK_LOCAL_MEM_FENCE);"                                                                                                                                      # renderer/cstyle.py:221
                          float4 = "(float4)"                                                                                                                                                            # renderer/cstyle.py:222
                          code_for_workitem = {"g": lambda x: f"get_group_id({x})", "l": lambda x: f"get_local_id({x})", "i": lambda x: f"get_global_id({x})"}                                           # renderer/cstyle.py:223
                          uses_vload = True                                                                                                                                                              # renderer/cstyle.py:224
                          type_map = { dtypes.uint8: "uchar", dtypes.uint32: "uint", dtypes.uint16: "ushort", dtypes.uint64: "ulong" }                                                                   # renderer/cstyle.py:225

                        class MetalRenderer(CStyleLanguage):                                                                                                                                             # renderer/cstyle.py:233
                          device = "METAL"                                                                                                                                                               # renderer/cstyle.py:234
                          shared_max = 32768                                                                                                                                                             # renderer/cstyle.py:235
                          tensor_cores = [TensorCore(dims=(8,8,8), threads=[(0,2),(1,4),(0,2),(1,2)], dtype_in=di, dtype_out=do) for (di, do) in [(dtypes.float, dtypes.float), (dtypes.half, dtypes.float), (dtypes.half, dtypes.half)]] # noqa: E501 # renderer/cstyle.py:236
                          kernel_prefix = "kernel "                                                                                                                                                      # renderer/cstyle.py:240
                          buffer_prefix = "device "                                                                                                                                                      # renderer/cstyle.py:241
                          smem_prefix = "threadgroup "                                                                                                                                                   # renderer/cstyle.py:242
                          arg_int_prefix = "constant int&"                                                                                                                                               # renderer/cstyle.py:243
                          barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"                                                                                                                   # renderer/cstyle.py:244
                          float4 = "float4"                                                                                                                                                              # renderer/cstyle.py:245
                          uses_ptr_arithmetic = True                                                                                                                                                     # renderer/cstyle.py:246
                          code_for_workitem = {"g": lambda x: f"gid.{chr(120+int(x))}", "l": lambda x: f"lid.{chr(120+int(x))}"}                                                                         # renderer/cstyle.py:247
                          extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']                                                                    # renderer/cstyle.py:249
                          type_map = {dtypes.bfloat16: "bfloat"}                                                                                                                                         # renderer/cstyle.py:250
                          code_for_op = {**CStyleLanguage().code_for_op,                                                                                                                                 # renderer/cstyle.py:251
                            BinaryOps.MAX: lambda a,b,dtype: f"(bfloat)max((float){a},(float){b})" if dtype == dtypes.bfloat16 else f"max({a},{b})",
                            UnaryOps.SQRT: lambda x,dtype: f"(bfloat)sqrt({x})" if dtype == dtypes.bfloat16 else f"sqrt({x})",
                            UnaryOps.EXP2: lambda x,dtype: f"(bfloat)exp2({x})" if dtype == dtypes.bfloat16 else f"exp2({x})",
                            UnaryOps.LOG2: lambda x,dtype: f"(bfloat)log2({x})" if dtype == dtypes.bfloat16 else f"log2({x})",
                            UnaryOps.SIN: lambda x,dtype: f"(bfloat)precise::sin({x})" if dtype == dtypes.bfloat16 else f"precise::sin({x})",}

                        code_for_op_half = {UnaryOps.RECIP: lambda x,dtype: f"hrcp({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"1/{x}",                                                       # renderer/cstyle.py:269
                                            BinaryOps.MAX: lambda a,b,dtype: f"__hmax({a},{b})" if dtype in (dtypes.half, dtypes.bfloat16) else f"max({a},{b})",
                                            UnaryOps.SQRT: lambda x,dtype: f"hsqrt({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sqrt({x})",
                                            UnaryOps.SIN: lambda x,dtype: f"hsin({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sin({x})",
                                            UnaryOps.LOG2: lambda x,dtype: f"hlog2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"log2({x})",
                                            UnaryOps.EXP2: lambda x,dtype: f"hexp2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"exp2({x})",}
                        _nms = "xyzwabcdefghijkl"                                                                                                                                                        # renderer/cstyle.py:276

                        class CUDARenderer(CStyleLanguage):                                                                                                                                              # renderer/cstyle.py:282
                          device = "CUDA"                                                                                                                                                                # renderer/cstyle.py:283
                          global_max = (2147483647, 65535, 65535)                                                                                                                                        # renderer/cstyle.py:284
                          local_max = (1024, 1024, 64)                                                                                                                                                   # renderer/cstyle.py:285
                          shared_max = 49152                                                                                                                                                             # renderer/cstyle.py:286
                          tensor_cores = [TensorCore(dims=(8,16,16), threads=[(0,2),(0,2),(1,2),(1,2),(1,2)], dtype_in=di, dtype_out=do) for (di, do) in ([(dtypes.half, dtypes.float), (dtypes.bfloat16, dtypes.float)])]  # noqa: E501 # renderer/cstyle.py:287
                          kernel_prefix = "extern \"C\" __global__ "                                                                                                                                     # renderer/cstyle.py:291
                          smem_prefix = "__shared__ "                                                                                                                                                    # renderer/cstyle.py:292
                          smem_prefix_for_cast = False                                                                                                                                                   # renderer/cstyle.py:293
                          barrier = "__syncthreads();"                                                                                                                                                   # renderer/cstyle.py:294
                          float4 = "make_float4"                                                                                                                                                         # renderer/cstyle.py:295
                          code_for_workitem = {"g": lambda x: f"blockIdx.{chr(120+int(x))}", "l": lambda x: f"threadIdx.{chr(120+int(x))}",                                                              # renderer/cstyle.py:296
                                               "i": lambda x: f"(blockIdx.{chr(120+int(x))}*blockDim.{chr(120+x)}+threadIdx.{chr(120+int(x))})"}
                          code_for_op = {**CStyleLanguage().code_for_op, **code_for_op_half}                                                                                                             # renderer/cstyle.py:298
                          type_map = {dtypes.bfloat16: "nv_bfloat16"}                                                                                                                                    # renderer/cstyle.py:299

                        code_for_op_hip = { UnaryOps.SQRT: lambda x,dtype: f"__ocml_sqrt_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",                                                    # renderer/cstyle.py:324
                                            UnaryOps.SIN: lambda x,dtype: f"__ocml_sin_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                                            UnaryOps.LOG2: lambda x,dtype: f"__ocml_log2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                                            UnaryOps.EXP2: lambda x,dtype: f"__ocml_exp2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                                            # TODO: MAX with int uses fmax_f32?
                                            BinaryOps.MAX: lambda a,b,dtype: f"__ocml_fmax_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32) }({a},{b})",}

                        class AMDRenderer(CStyleLanguage):                                                                                                                                               # renderer/cstyle.py:346
                          device = "AMD"                                                                                                                                                                 # renderer/cstyle.py:347
                          shared_max = 65536                                                                                                                                                             # renderer/cstyle.py:348
                          tensor_cores = [TensorCore(dims=(16,16,16), threads=[(0,8),(0,2),(1,2)], dtype_in=di, dtype_out=do) for (di, do) in [(dtypes.half, dtypes.float), (dtypes.half, dtypes.half)]] # noqa: E501 # renderer/cstyle.py:349
                          kernel_prefix = """extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_id(unsigned int);                                                         # renderer/cstyle.py:352
                        extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_group_id(unsigned int);
                        extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_size(unsigned int);
                        extern "C" {\n""" + "".join([
                        f"""  __attribute__((device)) __attribute__((const)) {dt} __ocml_fmax_f{n}({dt}, {dt});
                          __attribute__((device)) __attribute__((pure)) {dt} __ocml_exp2_f{n}({dt});
                          __attribute__((device)) __attribute__((pure)) {dt} __ocml_log2_f{n}({dt});
                          __attribute__((device)) __attribute__((const)) {dt} __ocml_sqrt_f{n}({dt});
                          __attribute__((device)) {dt} __ocml_sin_f{n}({dt});\n""" for dt,n in [("float",32), ("double",64), ("_Float16",16)]]) +\
                        '}\nextern "C" __attribute__((global))'
                          code_for_workitem = {"g": lambda x: f"__ockl_get_group_id({x})", "l": lambda x: f"__ockl_get_local_id({x})",                                                                   # renderer/cstyle.py:362
                                               "i": lambda x: f"(__ockl_get_group_id({x})*__ockl_get_local_size({x})+__ockl_get_local_id({x}))"}
                          code_for_op = _make_hip_code_for_op()                                                                                                                                          # renderer/cstyle.py:364

                            def _make_hip_code_for_op():                                                                                                                                                 # renderer/cstyle.py:331
                              return { k:wrapper(k,v) for k,v in {**CStyleLanguage().code_for_op, **code_for_op_hip}.items() }                                                                           # renderer/cstyle.py:339

                              def _make_hip_code_for_op():                                                                                                                                               # renderer/cstyle.py:331
                                def wrapper(key, func):                                                                                                                                                  # renderer/cstyle.py:332
                                  return cast_bf16                                                                                                                                                       # renderer/cstyle.py:338

                          smem_prefix = "__attribute__((shared))"                                                                                                                                        # renderer/cstyle.py:365
                          barrier = '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");' + '__builtin_amdgcn_s_barrier();' + \                                                                       # renderer/cstyle.py:366
                                    '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'
                          float4 = "make_float4"                                                                                                                                                         # renderer/cstyle.py:368
                          uses_ptr_arithmetic = False  # NOTE: this fixes TestLinearizerOverflowAlt                                                                                                      # renderer/cstyle.py:369
                          type_map = {dtypes.bfloat16: "hip_bfloat16"}                                                                                                                                   # renderer/cstyle.py:370

                        class NVRenderer(CUDARenderer): device = "NV"                                                                                                                                    # renderer/cstyle.py:414

                        class HIPRenderer(AMDRenderer): device = "HIP"                                                                                                                                   # renderer/cstyle.py:415

                          class ClangDevice(Compiled):                                                                                                                                                   # runtime/ops_clang.py:25
                            def __init__(self, device:str):                                                                                                                                              # runtime/ops_clang.py:26
                              from tinygrad.runtime.graph.clang import ClangGraph                                                                                                                        # runtime/ops_clang.py:27
                          from typing import List, Dict, cast                                                                                                                                            # runtime/graph/clang.py:1
                          import ctypes                                                                                                                                                                  # runtime/graph/clang.py:2
                          from tinygrad.helpers import dedup, cpu_time_execution, DEBUG                                                                                                                  # runtime/graph/clang.py:3
                          from tinygrad.engine.jit import GraphRunner, GraphException                                                                                                                    # runtime/graph/clang.py:4
                          from tinygrad.device import Buffer, Device                                                                                                                                     # runtime/graph/clang.py:5
                          from tinygrad.engine.realize import ExecItem, CompiledRunner                                                                                                                   # runtime/graph/clang.py:6
                          from tinygrad.shape.symbolic import Variable                                                                                                                                   # runtime/graph/clang.py:7
                          from tinygrad.runtime.ops_clang import ClangProgram                                                                                                                            # runtime/graph/clang.py:8
                          from tinygrad.renderer.cstyle import ClangRenderer                                                                                                                             # runtime/graph/clang.py:9
                          render_dtype = ClangRenderer().render_dtype                                                                                                                                    # runtime/graph/clang.py:10

                              super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler("compile_clang"), ClangProgram, ClangGraph)                                                       # runtime/ops_clang.py:28

                            return ExecItem(kernel_type(si.ast.arg, out.device, si.inputs[0].device), list(si.bufs))                                                                                     # engine/realize.py:199

                              @dataclass(frozen=True)                                                                                                                                                    # engine/schedule.py:25
                              class ScheduleItem:
                                @property                                                                                                                                                                # engine/schedule.py:34
                                def inputs(self) -> Tuple[Buffer, ...]:
                                  return self.bufs[len(self.ast.src):] if self.ast.op is MetaOps.KERNEL else self.bufs[1:]                                                                               # engine/schedule.py:36

                              class BufferCopy(Runner):                                                                                                                                                  # engine/realize.py:121
                                def __init__(self, total_sz, dest_device, src_device):                                                                                                                   # engine/realize.py:122
                                  if total_sz >= 1e6: name = f"{type(self).__name__[6:].lower()} {total_sz/1e6:7.2f}M, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"                                     # engine/realize.py:123
                                  else: name = f"{type(self).__name__[6:].lower()} {total_sz:8d}, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"                                                          # engine/realize.py:124
                                  super().__init__(colored(name, "yellow"), dest_device, 0, total_sz)                                                                                                    # engine/realize.py:125

                                    def colored(st, color:Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line  # noqa: E501 # helpers.py:28

                                    class Runner:                                                                                                                                                        # engine/realize.py:68
                                      def __init__(self, display_name:str, dname:str, op_estimate:sint=0, mem_estimate:sint=0, lds_estimate:Optional[sint]=None):                                        # engine/realize.py:69
                                        self.first_run, self.display_name, self.dname, self.op_estimate, self.mem_estimate, self.lds_estimate = \                                                        # engine/realize.py:70
                                          True, display_name, dname, op_estimate, mem_estimate, mem_estimate if lds_estimate is None else lds_estimate

                  if len(capturing) and CAPTURING: capturing[0].add(ei)                                                                                                                                  # engine/realize.py:222
                  ei.run(var_vals, do_update_stats=do_update_stats)                                                                                                                                      # engine/realize.py:223

                    @dataclass(frozen=True)                                                                                                                                                              # engine/realize.py:167
                    class ExecItem:
                      def run(self, var_vals:Optional[Dict[Variable, int]]=None, wait=False, jit=False, do_update_stats=True) -> Optional[float]:                                                        # engine/realize.py:171
                        bufs = [cast(Buffer, x) for x in self.bufs] if jit else [cast(Buffer, x).ensure_allocated() for x in self.bufs]                                                                  # engine/realize.py:172

                      class Buffer:                                                                                                                                                                      # device.py:52
                        def ensure_allocated(self) -> Buffer: return self.allocate() if not hasattr(self, '_buf') else self                                                                              # device.py:77

                        class LRUAllocator(Allocator):  # pylint: disable=abstract-method                                                                                                                # device.py:143
                          def alloc(self, size:int, options:Optional[BufferOptions]=None):                                                                                                               # device.py:149
                            if len(c := self.cache[(size, options)]): return c.pop()                                                                                                                     # device.py:150
                            try: return super().alloc(size, options)                                                                                                                                     # device.py:151

                              # TODO: size, dest, src are the same type. can we enforce this?                                                                                                            # device.py:132
                              class Allocator:
                                def alloc(self, size:int, options:Optional[BufferOptions]=None):                                                                                                         # device.py:133
                                  assert not isinstance(size, int) or size > 0, f"alloc size must be positve, getting {size}"                                                                            # device.py:134
                                  return self._alloc(size, options if options is not None else BufferOptions())                                                                                          # device.py:135

                                    class _MallocAllocator(LRUAllocator):                                                                                                                                # device.py:163
                                      def _alloc(self, size:int, options:BufferOptions): return (ctypes.c_uint8 * size)()                                                                                # device.py:164

                        et = self.prg(bufs, var_vals if var_vals is not None else {}, wait=wait or DEBUG >= 2)                                                                                           # engine/realize.py:173

                          class BufferCopy(Runner):                                                                                                                                                      # engine/realize.py:121
                            def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False):                                                                                          # engine/realize.py:135
                              dest, src = rawbufs[0:2]                                                                                                                                                   # engine/realize.py:136
                              assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"                                  # engine/realize.py:137
                              st = time.perf_counter()                                                                                                                                                   # engine/realize.py:138
                              self.copy(dest, src)                                                                                                                                                       # engine/realize.py:139

                                class BufferCopy(Runner):                                                                                                                                                # engine/realize.py:121
                                  def copy(self, dest, src):                                                                                                                                             # engine/realize.py:126
                                    disk_supports_fast_copyout = src.device.startswith("DISK") and hasattr(src.allocator.device, 'io_uring') and hasattr(src.allocator.device, 'fd')                     # engine/realize.py:127
                                    if src.device.startswith("DISK") and hasattr(dest.allocator, 'copy_from_disk') and disk_supports_fast_copyout and src.nbytes >= 4096:                                # engine/realize.py:128
                                    elif src.device.startswith("DISK") and hasattr(dest.allocator, 'as_buffer'):                                                                                         # engine/realize.py:130
                                      dest.copyin(src.as_buffer(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy                                                         # engine/realize.py:134

                                        class Buffer:                                                                                                                                                    # device.py:52
                                          def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:                                                                               # device.py:109
                                            if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, 'as_buffer'): return self.allocator.as_buffer(self._buf)                                 # device.py:111
                                            assert not force_zero_copy, "force zero copy was passed, but copy is required"                                                                               # device.py:112
                                            return self.copyout(memoryview(bytearray(self.nbytes)))                                                                                                      # device.py:113

                                              class Buffer:                                                                                                                                              # device.py:52
                                                def copyout(self, mv:memoryview) -> memoryview:                                                                                                          # device.py:120
                                                  mv = flat_mv(mv)                                                                                                                                       # device.py:121

                                                    def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))                                                          # helpers.py:307

                                                  assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"                                                              # device.py:122

                                                  assert self.is_allocated(), "can't copyout unallocated buffer"                                                                                         # device.py:123

                                                  self.allocator.copyout(mv, self._buf)                                                                                                                  # device.py:124

                                                    class NpyAllocator(Allocator):  # pylint: disable=abstract-method                                                                                    # runtime/ops_npy.py:5
                                                      def copyout(self, dest:memoryview, src:np.ndarray): dest[:] = flat_mv(np.require(src, requirements='C').data)                                      # runtime/ops_npy.py:6

                                                  return mv                                                                                                                                              # device.py:125

                                        class Buffer:                                                                                                                                                    # device.py:52
                                          def copyin(self, mv:memoryview):                                                                                                                               # device.py:114
                                            mv = flat_mv(mv)                                                                                                                                             # device.py:115

                                            assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"                                                                    # device.py:116

                                            assert self.is_allocated(), "can't copyin to unallocated buffer"                                                                                             # device.py:117

                                            self.allocator.copyin(self._buf, mv)                                                                                                                         # device.py:118

                                              class _MallocAllocator(LRUAllocator):                                                                                                                      # device.py:163
                                                def copyin(self, dest, src:memoryview): ctypes.memmove(dest, from_mv(src), len(src))                                                                     # device.py:166

                                                  # TODO: make this work with read only memoryviews (if possible)                                                                                        # helpers.py:296
                                                  def from_mv(mv:memoryview, to_type=ctypes.c_char):
                                                    return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents                                            # helpers.py:297

                                            return self                                                                                                                                                  # device.py:119

                              if wait:                                                                                                                                                                   # engine/realize.py:140

                        if do_update_stats:                                                                                                                                                              # engine/realize.py:174
                          GlobalCounters.kernel_count += 1                                                                                                                                               # engine/realize.py:175
                          GlobalCounters.global_ops += (op_est:=sym_infer(self.prg.op_estimate, var_vals))                                                                                               # engine/realize.py:176

                            def sym_infer(a: Union[Node, int], var_vals: Optional[Dict[Variable, int]]) -> int:                                                                                          # shape/symbolic.py:297
                              if isinstance(a, (int, float)): return a                                                                                                                                   # shape/symbolic.py:298

                          GlobalCounters.global_mem += (mem_est:=sym_infer(self.prg.mem_estimate, var_vals))                                                                                             # engine/realize.py:177

                          if et is not None: GlobalCounters.time_sum_s += et                                                                                                                             # engine/realize.py:178
                          if DEBUG >= 2:                                                                                                                                                                 # engine/realize.py:179

                          self.prg.first_run = False                                                                                                                                                     # engine/realize.py:186
                        return et                                                                                                                                                                        # engine/realize.py:187

                def lower_schedule_item(si:ScheduleItem) -> ExecItem:                                                                                                                                    # engine/realize.py:189
                  assert len(set(x.device for x in si.bufs)) == 1 or si.ast.op is MetaOps.COPY or getenv("USE_COPY_KERNEL")                                                                        # OLD # engine/realize.py:190
                  if si.ast.op is MetaOps.KERNEL:                                                                                                                                                  # OLD # engine/realize.py:191
                    runner = get_runner(si.outputs[0].device, si.ast)                                                                                                                                    # engine/realize.py:192

                      def get_runner(dname:str, ast:LazyOp) -> CompiledRunner:                                                                                                                           # engine/realize.py:150
                        ckey = (dname, ast, BEAM.value, NOOPT.value, False)                                                                                                                              # engine/realize.py:151
                        if cret:=method_cache.get(ckey): return cret                                                                                                                                     # engine/realize.py:152

                          @dataclass(frozen=True, eq=False)                                                                                                                                              # ops.py:55
                          class LazyOp:
                            def __hash__(self): return self.hash                                                                                                                                         # ops.py:82

                              @dataclass(frozen=True, eq=False)                                                                                                                                          # ops.py:55
                              class LazyOp:
                                @functools.cached_property                                                                                                                                               # ops.py:81
                                def hash(self): return hash((self.op, self.src, self.arg))

                        bkey = (dname.split(":")[0], ast, BEAM.value, NOOPT.value, True)                                                                                                                 # engine/realize.py:153
                        if bret:=method_cache.get(bkey):                                                                                                                                                 # engine/realize.py:154

                          prg: Program = get_kernel(Device[dname].renderer, ast).to_program()                                                                                                            # engine/realize.py:157

                            def get_kernel(renderer:Renderer, ast:LazyOp) -> Kernel:                                                                                                                     # engine/realize.py:17
                              if DEBUG >= 5:                                                                                                                                                             # engine/realize.py:18

                              k = Kernel(ast, opts=renderer).required_optimizations()                                                                                                                    # engine/realize.py:20

                                class Kernel:                                                                                                                                                            # codegen/kernel.py:54
                                  def __init__(self, *ast:LazyOp, opts:Optional[Renderer]=None):                                                                                                         # codegen/kernel.py:55
                                    if len(ast) > 1 or ast[0].op is BufferOps.STORE:                                                                                                                     # codegen/kernel.py:56
                                      assert len(ast) == 1 and ast[0].op is MetaOps.KERNEL                                                                                                               # codegen/kernel.py:60
                                      self.ast = ast[0]                                                                                                                                                  # codegen/kernel.py:61
                                    self.opts = opts if opts is not None else Device[Device.DEFAULT].renderer                                                                                            # codegen/kernel.py:63
                                    try: lazyop_sts_map = verify_lazyop(self.ast)                                                                                                                        # codegen/kernel.py:64

                                      # the living definition of LazyOps                                                                                                                                 # ops.py:139
                                      def verify_lazyop(ast:LazyOp) -> Dict[LazyOp, ShapeTracker]:
                                        assert ast.op is MetaOps.KERNEL, "must be SINK"                                                                                                                  # ops.py:140
                                        sts: Dict[LazyOp, ShapeTracker] = {}                                                                                                                             # ops.py:141
                                        for i, out in enumerate(ast.src):                                                                                                                                # ops.py:163
                                          assert out.arg.idx == i, f"unexpected output buffer idx {out.arg.idx} != {i}"                                                                                  # ops.py:164
                                          assert out.op is BufferOps.STORE, f"kernels must have stores as the output, got {out.op}"                                                                      # ops.py:165
                                          assert out.arg.st.size == ast.src[-1].arg.st.size, f"outputs must have the same size, got {out.arg.st.size}"                                                   # ops.py:166

                                          assert_valid(out, out.arg.st)                                                                                                                                  # ops.py:167

                                            # the living definition of LazyOps                                                                                                                           # ops.py:139
                                            def verify_lazyop(ast:LazyOp) -> Dict[LazyOp, ShapeTracker]:
                                              def assert_valid(op:LazyOp, st:ShapeTracker):                                                                                                              # ops.py:142
                                                if op in sts: return                                                                                                                                     # ops.py:143

                                                if op.op is BufferOps.LOAD and op.arg.idx < 0:                                                                                                           # ops.py:145
                                                for x in op.src: assert_valid(x, st)                                                                                                                     # ops.py:148

                                              # the living definition of LazyOps                                                                                                                         # ops.py:139
                                              def verify_lazyop(ast:LazyOp) -> Dict[LazyOp, ShapeTracker]:
                                                def assert_valid(op:LazyOp, st:ShapeTracker):                                                                                                            # ops.py:142
                                                  if op in sts: return                                                                                                                             # OLD # ops.py:143
                                                  if op.op is BufferOps.LOAD and op.arg.idx < 0:                                                                                                   # OLD # ops.py:145
                                                  for x in op.src: assert_valid(x, st)                                                                                                             # OLD # ops.py:148
                                                  if op.op in ReduceOps:                                                                                                                                 # ops.py:150
                                                    st = op.arg.st if op.op in BufferOps else sts[op.src[0]]                                                                                             # ops.py:157
                                                    for x in op.src:                                                                                                                                     # ops.py:158
                                                  sts[op] = st                                                                                                                                           # ops.py:162

                                                  # the living definition of LazyOps                                                                                                                     # ops.py:139
                                                  def verify_lazyop(ast:LazyOp) -> Dict[LazyOp, ShapeTracker]:
                                                    def assert_valid(op:LazyOp, st:ShapeTracker):                                                                                                        # ops.py:142
                                                      if op in sts: return                                                                                                                         # OLD # ops.py:143
                                                      if op.op is BufferOps.LOAD and op.arg.idx < 0:                                                                                               # OLD # ops.py:145
                                                      for x in op.src: assert_valid(x, st)                                                                                                         # OLD # ops.py:148
                                                      if op.op in ReduceOps:                                                                                                                       # OLD # ops.py:150
                                                        st = op.arg.st if op.op in BufferOps else sts[op.src[0]]                                                                                   # OLD # ops.py:157
                                                        for x in op.src:                                                                                                                           # OLD # ops.py:158
                                                          if sts[x].shape != st.shape:                                                                                                                   # ops.py:159

                                        shape_dims = [sorted(dedup(dims)) for dims in zip(*[x.shape for x in sts.values()])]                                                                             # ops.py:168

                                        assert all(len(x) == 1 or (len(x) == 2 and x[0] == 1) for x in shape_dims), f"shapes must have either 1 or n in each dimension, {shape_dims}"                    # ops.py:169
                                        return sts                                                                                                                                                       # ops.py:170

                                    self.reduceops = dedup([x for x in ordered_lazyops(self.ast) if x.op in ReduceOps])                                                                                  # codegen/kernel.py:72

                                      class Kernel:                                                                                                                                                      # codegen/kernel.py:54
                                        def __init__(self, *ast:LazyOp, opts:Optional[Renderer]=None):                                                                                                   # codegen/kernel.py:55
                                          @functools.lru_cache(None)                                                                                                                                     # codegen/kernel.py:71
                                          def ordered_lazyops(op): return dedup([item for x in op.src for item in ordered_lazyops(x)] + [op])

                                    self.vars = self.ast.vars()                                                                                                                                          # codegen/kernel.py:74

                                      @dataclass(frozen=True, eq=False)                                                                                                                                  # ops.py:55
                                      class LazyOp:
                                        def vars(self) -> List[Variable]:                                                                                                                                # ops.py:85
                                          extract_vars = [x.arg.st.vars() for x in self.lazyops if x.op in BufferOps]                                                                                    # ops.py:86

                                            @dataclass(frozen=True, eq=False)                                                                                                                            # ops.py:55
                                            class LazyOp:
                                              @functools.cached_property                                                                                                                                 # ops.py:84
                                              def lazyops(self) -> List[LazyOp]: return dedup([self] + [item for x in self.src for item in x.lazyops])

                                          const_vars = [x.arg.val for x in self.lazyops if x.op is BufferOps.CONST and isinstance(x.arg.val, Variable)]                                                  # ops.py:87
                                          return sorted(set.union(*extract_vars, set(const_vars)), key=lambda v: v.expr)                                                                                 # ops.py:88

                                    self.bufs: List[Union[MemBuffer, ConstBuffer]] = dedup([x.arg for x in self.ast.lazyops if x.op in BufferOps])                                                       # codegen/kernel.py:75

                                    earlybufs = [x.arg for reduceop in self.reduceops for x in reduceop.lazyops if x.op in BufferOps]                                                                    # codegen/kernel.py:78
                                    self.full_buf_index: int = self.bufs.index(earlybufs[0]) if earlybufs else 0                                                                                         # codegen/kernel.py:79
                                    self.sts: List[ShapeTracker] = [x.st for x in self.bufs]                                                                                                             # codegen/kernel.py:83
                                    for x in self.reduceops:                                                                                                                                             # codegen/kernel.py:87
                                    reduce = list(enumerate(zip(self.full_shape, self.output_shape)))                                                                                                    # codegen/kernel.py:92

                                      class Kernel:                                                                                                                                                      # codegen/kernel.py:54
                                        @property                                                                                                                                                        # codegen/kernel.py:157
                                        def full_shape(self) -> Tuple[sint, ...]: return self.sts[self.full_buf_index].shape

                                      class Kernel:                                                                                                                                                      # codegen/kernel.py:54
                                        @property                                                                                                                                                        # codegen/kernel.py:154
                                        def output_shape(self) -> Tuple[sint, ...]: return self.sts[0].shape

                                    permute = tuple([i for i,(s,n) in reduce if s == n] + [i for i,(s,n) in reduce if s != n])                                                                           # codegen/kernel.py:93
                                    self.reshape_and_permute(None, permute)                                                                                                                              # codegen/kernel.py:94

                                      class Kernel:                                                                                                                                                      # codegen/kernel.py:54
                                        # apply reshape and permute to all shapetrackers                                                                                                                 # codegen/kernel.py:204
                                        def reshape_and_permute(self, new_shape_fxn, axis):
                                          new_sts = []                                                                                                                                                   # codegen/kernel.py:205
                                          for st in self.sts:                                                                                                                                            # codegen/kernel.py:206
                                            if new_shape_fxn is not None: st = st.reshape(tuple(new_shape_fxn(st.shape)))                                                                                # codegen/kernel.py:207
                                            if axis is not None: st = st.permute(tuple(axis))                                                                                                            # codegen/kernel.py:208

                                              @dataclass(frozen=True)                                                                                                                                    # shape/shapetracker.py:10
                                              class ShapeTracker:
                                                def permute(self, axis: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))                       # shape/shapetracker.py:107

                                                  @dataclass(frozen=True)                                                                                                                                # shape/view.py:85
                                                  class View:
                                                    @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none                                                                     # shape/view.py:250
                                                    def permute(self, axis: Tuple[int, ...]) -> View:
                                                      assert sorted(axis) == list(range(len(self.shape))), f"invalid permutation {axis} of len {len(self.shape)}"                                        # shape/view.py:251
                                                      return View.create(tuple(self.shape[a] for a in axis), tuple(self.strides[a] for a in axis), self.offset,                                          # shape/view.py:252
                                                                         tuple(self.mask[a] for a in axis) if self.mask is not None else None)

                                            new_sts.append(st)                                                                                                                                           # codegen/kernel.py:209

                                          self.sts = new_sts                                                                                                                                             # codegen/kernel.py:210

                                    self.applied_opts: List[Opt] = []                                                                                                                                    # codegen/kernel.py:97
                                    self.group_for_reduces: int = 0                                                                                                                                      # codegen/kernel.py:98
                                    self.upcasted: int = 0                                                                                                                                               # codegen/kernel.py:99
                                    self.local_dims: int = 0                                                                                                                                             # codegen/kernel.py:100
                                    self.tensor_core: Optional[TensorCore] = None                                                                                                                        # codegen/kernel.py:101
                                    self.tensor_core_opts: Optional[TensorCoreOptions] = None                                                                                                            # codegen/kernel.py:102
                                    self.use_tensor_cores: int = 0                                                                                                                                       # codegen/kernel.py:103
                                    self.bufs_for_tensor_core: Dict[LazyOp, Tuple[int, int]] = {}                                                                                                        # codegen/kernel.py:105
                                    self.dont_use_locals: bool = False                                                                                                                                   # codegen/kernel.py:106
                                    self.simplify_ones()                                                                                                                                                 # codegen/kernel.py:109

                                      class Kernel:                                                                                                                                                      # codegen/kernel.py:54
                                        def simplify_ones(self) -> bool:                                                                                                                                 # codegen/kernel.py:231
                                          if self.shape_len == 0: return False                                                                                                                           # codegen/kernel.py:234

                                            class Kernel:                                                                                                                                                # codegen/kernel.py:54
                                              @property                                                                                                                                                  # codegen/kernel.py:163
                                              def shape_len(self) -> int: return len(self.sts[0].shape)

                                          all_ones = [s==1 for s in self.full_shape]                                                                                                                     # codegen/kernel.py:235

                                          self.local_dims -= sum(all_ones[self.first_reduce-self.local_dims:self.first_reduce])                                                                          # codegen/kernel.py:236

                                            class Kernel:                                                                                                                                                # codegen/kernel.py:54
                                              @property                                                                                                                                                  # codegen/kernel.py:144
                                              def first_reduce(self) -> int:
                                                return [x!=y for x,y in zip(self.sts[0].shape[:self.first_upcast]+(0,), self.full_shape[:self.first_upcast]+(1,))].index(True)                           # codegen/kernel.py:145

                                                  class Kernel:                                                                                                                                          # codegen/kernel.py:54
                                                    @property                                                                                                                                            # codegen/kernel.py:148
                                                    def first_upcast(self) -> int: return self.shape_len-self.upcasted

                                          self.upcasted -= sum(all_ones[self.first_upcast:]) # TODO: no necessary since upcasted axis can't be un-upcasted                                               # codegen/kernel.py:237

                                          self.reshape_and_permute(lambda shape: [x for i,x in enumerate(shape) if not all_ones[i]], None)                                                               # codegen/kernel.py:238

                                          return any(all_ones)                                                                                                                                           # codegen/kernel.py:239

                                    self.simplify_merge_adjacent()                                                                                                                                       # codegen/kernel.py:110

                                      class Kernel:                                                                                                                                                      # codegen/kernel.py:54
                                        def simplify_merge_adjacent(self):                                                                                                                               # codegen/kernel.py:241
                                          if self.shape_len == 0: return                                                                                                                                 # codegen/kernel.py:242

                                          shapes, strides = [x.shape for x in self.sts], [x.real_strides() for x in self.sts]                                                                            # codegen/kernel.py:243

                                        @dataclass(frozen=True)                                                                                                                                          # shape/shapetracker.py:10
                                        class ShapeTracker:
                                          # NOTE: if a stride is not always valid, it will be None                                                                                                       # shape/shapetracker.py:60
                                          def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]:
                                            if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides                                                                       # shape/shapetracker.py:61

                                          if isinstance(self.bufs[0].dtype, ImageDType):                                                                                                                 # codegen/kernel.py:246
                                          rets = [[(s[0], st[0])] for s,st in zip(shapes, strides)]                                                                                                      # codegen/kernel.py:262
                                          for i in range(1, len(shapes[0])):                                                                                                                             # codegen/kernel.py:263
                                          for i,x in enumerate(rets[:len(self.sts)]): self.sts[i] = self.sts[i].reshape(tuple([y[0] for y in x]))                                                        # codegen/kernel.py:276

                                class Kernel:                                                                                                                                                            # codegen/kernel.py:54
                                  def required_optimizations(self) -> Kernel:                                                                                                                            # codegen/kernel.py:483
                                    if self.bufs[0].dtype.__class__ is ImageDType:                                                                                                                       # codegen/kernel.py:484
                                    return self                                                                                                                                                          # codegen/kernel.py:489

                              if not NOOPT:                                                                                                                                                              # engine/realize.py:21

                                if not (used_tensor_cores:=k.apply_tensor_cores(getenv("TC", 1))): k.hand_coded_optimizations()                                                                          # engine/realize.py:22

                                  class Kernel:                                                                                                                                                          # codegen/kernel.py:54
                                    def apply_tensor_cores(self, use_tensor_cores=1, extra_opts:Optional[List[Opt]]=None, axis:int=0, tc_opt:Optional[int]=None) -> bool:                                # codegen/kernel.py:348
                                      if tc_opt is None: tc_opt = TC_OPT.value                                                                                                                           # codegen/kernel.py:363
                                      if not self.opts.tensor_cores and use_tensor_cores != 2: return False                                                                                              # codegen/kernel.py:364

                                  class Kernel:                                                                                                                                                          # codegen/kernel.py:54
                                    def hand_coded_optimizations(self) -> Kernel:                                                                                                                        # codegen/kernel.py:491
                                      self.required_optimizations()                                                                                                                                      # codegen/kernel.py:492

                                      MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)                 # codegen/kernel.py:495

                                      if self.opts.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \                                   # codegen/kernel.py:496
                                          self.reduceop is not None and self.reduceop.op is ReduceOps.SUM and len(self.full_shape) >= 2 and self.opts.has_shared and \
                                          (mulop:=self.reduceop.src[0]).op is BinaryOps.MUL and mulop.src[0].op is BufferOps.LOAD and mulop.src[1].op is BufferOps.LOAD:
                                      if self.opts.has_local and self.opts.has_shared and all_int(self.sts[0].shape[:self.first_reduce]):                                                                # codegen/kernel.py:512
                                      for buf_index,buf in enumerate(self.bufs):                                                                                                                         # codegen/kernel.py:531
                                        unit_stride_axes_mul_4 = [i for i in self.sts[buf_index].unit_stride_axes(ignore_valid=True) if self.sts[buf_index].shape[i]%4 == 0]                             # codegen/kernel.py:532

                                          @dataclass(frozen=True)                                                                                                                                        # shape/shapetracker.py:10
                                          class ShapeTracker:
                                            def unit_stride_axes(self, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]                    # shape/shapetracker.py:76

                                        if buf.dtype.__class__ is ImageDType:                                                                                                                            # codegen/kernel.py:533

                                      if self.group_for_reduces: return self                                                                                                                             # codegen/kernel.py:542
                                      to_upcast: List[int] = []                                                                                                                                          # codegen/kernel.py:551
                                      for axis in range(self.first_reduce):                                                                                                                              # codegen/kernel.py:553

                                        if isinstance(self.full_shape[axis], int) and self.full_shape[axis] <= 7 and any(st.axis_is_masked(axis) for st in self.sts) and \                               # codegen/kernel.py:556
                                          prod(self.full_shape[self.first_upcast:]) * prod(self.full_shape[j] for j in to_upcast) * self.full_shape[axis] <= 7 * 7:

                                    @dataclass(frozen=True)                                                                                                                                              # shape/shapetracker.py:10
                                    class ShapeTracker:
                                      def axis_is_masked(self, axis:int) -> bool:                                                                                                                        # shape/shapetracker.py:93
                                        _, valid = self.expr_idxs()                                                                                                                                      # shape/shapetracker.py:94

                                          @dataclass(frozen=True)                                                                                                                                        # shape/shapetracker.py:10
                                          class ShapeTracker:
                                            def expr_idxs(self, idxs:Optional[Iterable[Node]]=None) -> Tuple[Node, Node]:                                                                                # shape/shapetracker.py:78
                                              idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)] if idxs is None else list(idxs)                                                      # shape/shapetracker.py:79

                                            class Variable(Node):                                                                                                                                        # shape/symbolic.py:110
                                              def __new__(cls, *args):                                                                                                                                   # shape/symbolic.py:111
                                                expr, nmin, nmax = args                                                                                                                                  # shape/symbolic.py:112
                                                assert nmin >= 0 and nmin <= nmax, f"invalid Variable {expr=} {nmin=} {nmax=}"                                                                           # shape/symbolic.py:113
                                                if nmin == nmax: return NumNode(nmin)                                                                                                                    # shape/symbolic.py:114
                                                return super().__new__(cls)                                                                                                                              # shape/symbolic.py:115

                                            class Variable(Node):                                                                                                                                        # shape/symbolic.py:110
                                              def __init__(self, expr:str, nmin:int, nmax:sint):                                                                                                         # shape/symbolic.py:119
                                                self.expr, self.min, self.max = expr, nmin, nmax                                                                                                         # shape/symbolic.py:120
                                                self._val: Optional[int] = None                                                                                                                          # shape/symbolic.py:121

                                              idx, valid = self.views[-1].expr(idxs)                                                                                                                     # shape/shapetracker.py:80

                                                @dataclass(frozen=True)                                                                                                                                  # shape/view.py:85
                                                class View:
                                                  def expr(self, idxs:List[Node], valid:Optional[Node]=None) -> Tuple[Node, Node]:                                                                       # shape/view.py:316
                                                    assert len(idxs) == len(self.shape), f"need an idx for all dimensions {idxs} vs {self.shape}"                                                        # shape/view.py:317
                                                    iexpr: List[Node] = [NumNode(self.offset) if isinstance(self.offset, int) else self.offset]                                                          # shape/view.py:318

                                                      class NumNode(Node):                                                                                                                               # shape/symbolic.py:136
                                                        def __init__(self, num:int):                                                                                                                     # shape/symbolic.py:137
                                                          assert isinstance(num, int), f"{num} is not an int"                                                                                            # shape/symbolic.py:138
                                                          self.b:int = num                                                                                                                               # shape/symbolic.py:139
                                                          self.min, self.max = num, num                                                                                                                  # shape/symbolic.py:140

                                                    vexpr: List[Node] = [valid] if valid is not None else []                                                                                             # shape/view.py:319
                                                    for idx,sh,st,m in zip(idxs, self.shape, self.strides, self.mask if self.mask is not None else [None]*len(self.shape)):                              # shape/view.py:320
                                                      if sh != 1 and st != 0: iexpr.append(idx*st)                                                                                                       # shape/view.py:321

                                                        class Node:                                                                                                                                      # shape/symbolic.py:10
                                                          def __mul__(self, b:Union[Node, int]):                                                                                                         # shape/symbolic.py:41
                                                            if b == 0: return NumNode(0)                                                                                                                 # shape/symbolic.py:42
                                                            if b == 1: return self                                                                                                                       # shape/symbolic.py:43

                                                      if m is not None: vexpr += [create_ge_node(idx, m[0]), create_lt_node(idx, m[1])]  # idx >= m[0], idx < m[1]                                       # shape/view.py:322
                                                    return Node.sum(iexpr), Node.ands(vexpr)                                                                                                             # shape/view.py:323

                                                      class Node:                                                                                                                                        # shape/symbolic.py:10
                                                        @staticmethod                                                                                                                                    # shape/symbolic.py:83
                                                        def sum(nodes:List[Node]) -> Node:
                                                          nodes = [x for x in nodes if x.max or x.min]                                                                                                   # shape/symbolic.py:84
                                                          if not nodes: return NumNode(0)                                                                                                                # shape/symbolic.py:85
                                                          if len(nodes) == 1: return nodes[0]                                                                                                            # shape/symbolic.py:86

                                                      class Node:                                                                                                                                        # shape/symbolic.py:10
                                                        @staticmethod                                                                                                                                    # shape/symbolic.py:99
                                                        def ands(nodes:List[Node]) -> Node:
                                                          if not nodes: return NumNode(1)                                                                                                                # shape/symbolic.py:100

                                              for view in reversed(self.views[0:-1]):                                                                                                                    # shape/shapetracker.py:81
                                              assert not isinstance(idx.min, int) or idx.min >= -2**31, f"idx.min too small. {idx=}, {idx.min=}"                                                         # shape/shapetracker.py:89
                                              assert not isinstance(idx.max, int) or idx.max < 2**31, f"idx.max too big. {idx=}, {idx.max=}"                                                             # shape/shapetracker.py:90
                                              return idx, valid                                                                                                                                          # shape/shapetracker.py:91

                                        return f'idx{axis}' in [v.expr for v in valid.vars()]                                                                                                            # shape/shapetracker.py:95

                                          class Node:                                                                                                                                                    # shape/symbolic.py:10
                                            def vars(self) -> Set[Variable]: return set()                                                                                                                # shape/symbolic.py:18

                                      for axis in to_upcast[::-1]: self.apply_opt(Opt(OptOps.UPCAST, axis, 0))                                                                                           # codegen/kernel.py:560
                                      upcasted_axis = set()                                                                                                                                              # codegen/kernel.py:563
                                      while prod(self.sts[0].shape[:self.first_reduce]) >= 1024:                                                                                                         # codegen/kernel.py:564

                                      if self.first_reduce < self.first_upcast and (prod(self.full_shape[self.first_upcast:]) <= 4 or not any(r for _,_,r in self.upcasted_axis(self.full_buf_index))) and (self.upcasted == 0 or prod(self.full_shape[-self.upcasted:]) < 64):  # noqa: E501 # codegen/kernel.py:578

                                      for splits in [4]:                                                                                                                                                 # codegen/kernel.py:592
                                        if self.upcasted == 0 and self.full_unupcasted_shape and self.full_unupcasted_shape[-1] % splits == 0:                                                           # codegen/kernel.py:593

                                          class Kernel:                                                                                                                                                  # codegen/kernel.py:54
                                            @property                                                                                                                                                    # codegen/kernel.py:160
                                            def full_unupcasted_shape(self) -> Tuple[sint, ...]: return self.full_shape[:self.first_upcast]

                                      if self.opts.has_local:                                                                                                                                            # codegen/kernel.py:598
                                      return self                                                                                                                                                        # codegen/kernel.py:616

                                if BEAM >= 1:                                                                                                                                                            # engine/realize.py:23

                              if logkerns is not None: logkerns.writelines([f"{(k.ast, k.applied_opts)}\n"])                                                                                             # engine/realize.py:62
                              if DEBUG >= 5: print((k.ast, k.applied_opts)) # print here to show final applied_opts for all kernels instead of just in beam_search                                       # engine/realize.py:63

                              return k                                                                                                                                                                   # engine/realize.py:64

                            class Kernel:                                                                                                                                                                # codegen/kernel.py:54
                              def to_program(self, name_override:Optional[str]=None) -> Program:                                                                                                         # codegen/kernel.py:737
                                self.linearize()                                                                                                                                                         # codegen/kernel.py:738

                                  class Kernel:                                                                                                                                                          # codegen/kernel.py:54
                                    def linearize(self) -> Kernel:                                                                                                                                       # codegen/kernel.py:722
                                      modified_ast = self.get_optimized_ast()                                                                                                                            # codegen/kernel.py:723

                                        class Kernel:                                                                                                                                                    # codegen/kernel.py:54
                                          def get_optimized_ast(self) -> LazyOp:                                                                                                                         # codegen/kernel.py:633
                                            return fixup_ast(self.ast)                                                                                                                                   # codegen/kernel.py:718

                                              class Kernel:                                                                                                                                              # codegen/kernel.py:54
                                                def get_optimized_ast(self) -> LazyOp:                                                                                                                   # codegen/kernel.py:633
                                                  # set the shapetrackers to the optimized ones, fixup reduceop                                                                                          # codegen/kernel.py:637
                                                  # transformed to the final LazyOp
                                                  @functools.lru_cache(None)
                                                  def fixup_ast(op:LazyOp, apply_to_st=None) -> LazyOp:
                                                    if op.op in BufferOps:                                                                                                                               # codegen/kernel.py:638
                                                    elif op.op in ReduceOps:                                                                                                                             # codegen/kernel.py:645
                                                    elif op.op is MetaOps.KERNEL:                                                                                                                        # codegen/kernel.py:713
                                                      arg = KernelInfo(self.local_dims, self.upcasted, self.dont_use_locals)                                                                             # codegen/kernel.py:714
                                                    return LazyOp(op.op, tuple(fixup_ast(x, apply_to_st) for x in op.src), arg)                                                                          # codegen/kernel.py:717

                                                class Kernel:                                                                                                                                            # codegen/kernel.py:54
                                                  def get_optimized_ast(self) -> LazyOp:                                                                                                                 # codegen/kernel.py:633
                                                    # set the shapetrackers to the optimized ones, fixup reduceop                                                                                        # codegen/kernel.py:637
                                                    # transformed to the final LazyOp
                                                    @functools.lru_cache(None)
                                                    def fixup_ast(op:LazyOp, apply_to_st=None) -> LazyOp:
                                                      if op.op in BufferOps:                                                                                                                       # OLD # codegen/kernel.py:638
                                                        if isinstance(op.arg, MemBuffer) and op.arg.idx < 0:                                                                                             # codegen/kernel.py:639
                                                          idx = self.bufs.index(op.arg)                                                                                                                  # codegen/kernel.py:643
                                                          arg = replace(op.arg, st=self.sts[idx] if apply_to_st is None else apply_to_st(self.sts[idx]))                                                 # codegen/kernel.py:644

                                                  class Kernel:                                                                                                                                          # codegen/kernel.py:54
                                                    def get_optimized_ast(self) -> LazyOp:                                                                                                               # codegen/kernel.py:633
                                                      # set the shapetrackers to the optimized ones, fixup reduceop                                                                                      # codegen/kernel.py:637
                                                      # transformed to the final LazyOp
                                                      @functools.lru_cache(None)
                                                      def fixup_ast(op:LazyOp, apply_to_st=None) -> LazyOp:
                                                        if op.op in BufferOps:                                                                                                                     # OLD # codegen/kernel.py:638
                                                        elif op.op in ReduceOps:                                                                                                                   # OLD # codegen/kernel.py:645
                                                        elif op.op is MetaOps.KERNEL:                                                                                                              # OLD # codegen/kernel.py:713
                                                          arg = KernelInfo(self.local_dims, self.upcasted, self.dont_use_locals)                                                                   # OLD # codegen/kernel.py:714
                                                          arg = op.arg                                                                                                                                   # codegen/kernel.py:716

                                      if DEBUG >= 3:                                                                                                                                                     # codegen/kernel.py:725

                                      verify_lazyop(modified_ast)                                                                                                                                        # codegen/kernel.py:729

                                      self.uops:UOpGraph = UOpGraph(lazyop_to_uop(modified_ast, self.opts), self.opts)                                                                                   # codegen/kernel.py:732

                                        def lazyop_to_uop(ast:LazyOp, opts:Renderer) -> UOp: return IndependentLowerer().lower(ast, opts)                                                                # codegen/lowerer.py:215

                                          class IndependentLowerer:                                                                                                                                      # codegen/lowerer.py:108
                                            def lower(self, ast:LazyOp, opts:Renderer) -> UOp:                                                                                                           # codegen/lowerer.py:109
                                              self.output_count = len(ast.src)                                                                                                                           # codegen/lowerer.py:110
                                              ki = ast.arg if isinstance(ast.arg, KernelInfo) else KernelInfo()                                                                                          # codegen/lowerer.py:112
                                              full_shape = ast.full_shape                                                                                                                                # codegen/lowerer.py:114

                                                @dataclass(frozen=True, eq=False)                                                                                                                        # ops.py:55
                                                class LazyOp:
                                                  @functools.cached_property                                                                                                                             # ops.py:74
                                                  def full_shape(self) -> Tuple[sint, ...]:
                                                    if len(self.src) == 0 and self.op in BufferOps: return self.arg.st.shape                                                                             # ops.py:75
                                                    return tuple(max(x) for x in zip(*[x.full_shape for x in self.src]))                                                                                 # ops.py:76

                                              first_upcasted = len(full_shape)-ki.upcasted                                                                                                               # codegen/lowerer.py:115
                                              first_reduce = [x!=y for x,y in zip(ast.src[0].arg.st.shape[:first_upcasted]+(0,), full_shape[:first_upcasted]+(1,))].index(True)                          # codegen/lowerer.py:117

                                              local_loads = [x for x in ast.lazyops if x.op is BufferOps.LOAD and x.arg.idx == -1]                                                                       # codegen/lowerer.py:118

                                              group_for_reduces = sum([x!=y for x,y in zip(                                                                                                              # codegen/lowerer.py:120
                                                local_loads[0].arg.st.shape[first_reduce:first_upcasted], ast.src[0].arg.st.shape[first_reduce:first_upcasted])]) if local_loads else 0
                                              global_dims = first_reduce-ki.local_dims                                                                                                                   # codegen/lowerer.py:122
                                              if opts.has_local:                                                                                                                                         # codegen/lowerer.py:124
                                                self.idxs = [UOp(UOps.RANGE, dtypes.bigint, (UOp.const(dtypes.bigint, 0), variable_to_uop(g)), (i, False))                                               # codegen/lowerer.py:134
                                                             for i,g in enumerate(full_shape[:first_reduce])]

                                            def variable_to_uop(x, ctx=None) -> UOp: return UOp.const(dtypes.bigint, x) if isinstance(x, int) else x.render(render_ops, ctx)                             # codegen/lowerer.py:14

                                              self.idxs += [UOp(UOps.RANGE, dtypes.bigint, (UOp.const(dtypes.bigint, 0), variable_to_uop(g)), (i, True))                                                 # codegen/lowerer.py:138
                                                for i,g in enumerate(full_shape[first_reduce+group_for_reduces:first_upcasted], start=first_reduce+group_for_reduces)]
                                              for i,g in enumerate(full_shape[first_upcasted:], start=first_upcasted):                                                                                   # codegen/lowerer.py:142
                                              self.ridxs = self.idxs[:]                                                                                                                                  # codegen/lowerer.py:147
                                              for a in range(first_reduce, first_reduce+group_for_reduces):                                                                                              # codegen/lowerer.py:148
                                              self.uop_cache: Dict[LazyOp, UOp] = {}                                                                                                                     # codegen/lowerer.py:151
                                              return self.to_uop(ast)                                                                                                                                    # codegen/lowerer.py:152

                                                class IndependentLowerer:                                                                                                                                # codegen/lowerer.py:108
                                                  def to_uop(self, x:LazyOp) -> UOp:                                                                                                                     # codegen/lowerer.py:154
                                                    if uop:=self.uop_cache.get(x, None): return uop                                                                                                      # codegen/lowerer.py:155

                                                    ret = self._to_uop(x)                                                                                                                                # codegen/lowerer.py:156

                                                      class IndependentLowerer:                                                                                                                          # codegen/lowerer.py:108
                                                        def _to_uop(self, x:LazyOp) -> UOp:                                                                                                              # codegen/lowerer.py:160
                                                          if x.op in BufferOps:                                                                                                                          # codegen/lowerer.py:161
                                                          in_uops = tuple(self.to_uop(y) for y in x.src)                                                                                                 # codegen/lowerer.py:191

                                                        class IndependentLowerer:                                                                                                                        # codegen/lowerer.py:108
                                                          def _to_uop(self, x:LazyOp) -> UOp:                                                                                                            # codegen/lowerer.py:160
                                                            if x.op in BufferOps:                                                                                                                  # OLD # codegen/lowerer.py:161
                                                              idx, valid = st_to_uops(x.arg.st, self.ridxs if x.op is BufferOps.LOAD and x.arg.idx == -1 else self.idxs,                                 # codegen/lowerer.py:162
                                                                x.arg.dtype.base if isinstance(x.arg.dtype, ImageDType) and (not isinstance(x.arg, MemBuffer) or x.arg.idx == -1) else x.arg.dtype)

                                                                def st_to_uops(st:ShapeTracker, idxs:List[UOp], dtype:DType) -> Tuple[UOp, UOp]:                                                         # codegen/lowerer.py:65
                                                                  if getenv("SYMBOLIC_DIFF"):                                                                                                            # codegen/lowerer.py:66

                                                                  return st_to_uops_graph(st, idxs, dtype) if getenv("UOP_IS_SYMBOLIC") else st_to_uops_symbolic(st, idxs, dtype)                        # codegen/lowerer.py:80

                                                                    # TODO: this is the old one, delete when ready                                                                                       # codegen/lowerer.py:51
                                                                    def st_to_uops_symbolic(st:ShapeTracker, idxs:List[UOp], dtype:DType) -> Tuple[UOp, UOp]:
                                                                      fake_idxs = [Variable(f"__idx{i}", 0, s-1) for i,s in enumerate(st.shape)]                                                         # codegen/lowerer.py:52

                                                                      idx, valid = st.expr_idxs(fake_idxs)                                                                                               # codegen/lowerer.py:53

                                                                      ctx = dict(zip(fake_idxs, idxs))                                                                                                   # codegen/lowerer.py:54

                                                                        class Node:                                                                                                                      # shape/symbolic.py:10
                                                                          def __hash__(self): return hash(self.key)                                                                                      # shape/symbolic.py:27

                                                                            class Node:                                                                                                                  # shape/symbolic.py:10
                                                                              @functools.cached_property                                                                                                 # shape/symbolic.py:24
                                                                              def key(self) -> str: return self.render(ctx="DEBUG")

                                                                                class Node:                                                                                                              # shape/symbolic.py:10
                                                                                  def render(self, ops=None, ctx=None) -> Any:                                                                           # shape/symbolic.py:14
                                                                                    if ops is None: ops = render_python                                                                                  # shape/symbolic.py:15
                                                                                    assert self.__class__ in (Variable, NumNode) or self.min != self.max                                                 # shape/symbolic.py:16
                                                                                    return ops[type(self)](self, ops, ctx)                                                                               # shape/symbolic.py:17

                                                                      uvalid = valid.render(render_ops, ctx)                                                                                             # codegen/lowerer.py:55

                                                                      if isinstance(dtype, ImageDType):                                                                                                  # codegen/lowerer.py:56
                                                                        uidx = idx.render(render_ops, ctx)                                                                                               # codegen/lowerer.py:60

                                                                      if uvalid.op is UOps.CONST: uvalid = UOp.const(dtypes.bool, uvalid.arg)                                                            # codegen/lowerer.py:61

                                                                      assert uvalid.dtype == dtypes.bool                                                                                                 # codegen/lowerer.py:62
                                                                      return uidx, uvalid                                                                                                                # codegen/lowerer.py:63

                                                              has_valid = valid.op is not UOps.CONST or valid.arg is not True                                                                            # codegen/lowerer.py:165
                                                              if x.op is BufferOps.CONST:                                                                                                                # codegen/lowerer.py:166
                                                              if x.arg.idx < 0:                                                                                                                          # codegen/lowerer.py:169
                                                                buf = UOp(UOps.DEFINE_GLOBAL, x.arg.dtype if isinstance(x.arg.dtype, ImageDType) else PtrDType(x.arg.dtype), (), x.arg.idx)              # codegen/lowerer.py:173

                                                                  # @dataclass(frozen=True, init=False, repr=False, eq=False)                                                                            # dtype.py:31
                                                                  class PtrDType(DType):
                                                                    def __init__(self, dt:DType): super().__init__(dt.priority, dt.itemsize, dt.name, dt.fmt, dt.count)                                  # dtype.py:32

                                                              if x.op is BufferOps.LOAD:                                                                                                                 # codegen/lowerer.py:174
                                                              if x.arg.idx >= 0:                                                                                                                         # codegen/lowerer.py:185
                                                                for oidx, ridx in zip(self.idxs, self.ridxs):                                                                                            # codegen/lowerer.py:186
                                                                  if oidx != ridx: valid = valid * oidx.eq(0)                                                                                            # codegen/lowerer.py:187
                                                                has_valid = valid.op is not UOps.CONST or valid.arg is not True                                                                          # codegen/lowerer.py:188
                                                              return UOp(UOps.STORE, None, (buf, idx, self.to_uop(x.src[0])) + ((valid,) if has_valid else ()))                                          # codegen/lowerer.py:189

                                                          class IndependentLowerer:                                                                                                                      # codegen/lowerer.py:108
                                                            def _to_uop(self, x:LazyOp) -> UOp:                                                                                                          # codegen/lowerer.py:160
                                                              if x.op in BufferOps:                                                                                                                # OLD # codegen/lowerer.py:161
                                                                  barrier = (UOp(UOps.BARRIER, None, (self.to_uop(x.src[0]),)),) if len(x.src) else ()                                                   # codegen/lowerer.py:175
                                                                  load_dtype = x.arg.dtype.scalar()                                                                                                      # codegen/lowerer.py:176

                                                                  if idx.dtype == dtypes.int.vec(3):                                                                                                     # codegen/lowerer.py:177

                                                                  return UOp(UOps.LOAD, load_dtype, (buf, idx) + ((UOp.const(load_dtype, 0), valid) if has_valid else ()) + barrier)                     # codegen/lowerer.py:183

                                                          class IndependentLowerer:                                                                                                                      # codegen/lowerer.py:108
                                                            def to_uop(self, x:LazyOp) -> UOp:                                                                                                           # codegen/lowerer.py:154
                                                              if uop:=self.uop_cache.get(x, None): return uop                                                                                      # OLD # codegen/lowerer.py:155
                                                              ret = self._to_uop(x)                                                                                                                # OLD # codegen/lowerer.py:156
                                                              self.uop_cache[x] = ret                                                                                                                    # codegen/lowerer.py:157

                                                              return ret                                                                                                                                 # codegen/lowerer.py:158

                                                          class IndependentLowerer:                                                                                                                      # codegen/lowerer.py:108
                                                            def _to_uop(self, x:LazyOp) -> UOp:                                                                                                          # codegen/lowerer.py:160
                                                              if x.op in BufferOps:                                                                                                                # OLD # codegen/lowerer.py:161
                                                                  dtype = x.arg.dtype.base if isinstance(x.arg.dtype, ImageDType) else x.arg.dtype                                                       # codegen/lowerer.py:167
                                                                  return valid.where(UOp.const(dtype, x.arg.val), UOp.const(dtype, 0))                                                                   # codegen/lowerer.py:168

                                                          class IndependentLowerer:                                                                                                                      # codegen/lowerer.py:108
                                                            def _to_uop(self, x:LazyOp) -> UOp:                                                                                                          # codegen/lowerer.py:160
                                                              if x.op in BufferOps:                                                                                                                # OLD # codegen/lowerer.py:161
                                                              in_uops = tuple(self.to_uop(y) for y in x.src)                                                                                       # OLD # codegen/lowerer.py:191
                                                                idx, valid = st_to_uops(x.arg.st, self.ridxs if x.op is BufferOps.LOAD and x.arg.idx == -1 else self.idxs,                         # OLD # codegen/lowerer.py:162
                                                                  x.arg.dtype.base if isinstance(x.arg.dtype, ImageDType) and (not isinstance(x.arg, MemBuffer) or x.arg.idx == -1) else x.arg.dtype)
                                                                has_valid = valid.op is not UOps.CONST or valid.arg is not True                                                                    # OLD # codegen/lowerer.py:165
                                                                if x.op is BufferOps.CONST:                                                                                                        # OLD # codegen/lowerer.py:166
                                                                if x.arg.idx < 0:                                                                                                                  # OLD # codegen/lowerer.py:169
                                                                  buf = UOp(UOps.DEFINE_GLOBAL, x.arg.dtype if isinstance(x.arg.dtype, ImageDType) else PtrDType(x.arg.dtype), (), x.arg.idx)      # OLD # codegen/lowerer.py:173
                                                                if x.op is BufferOps.LOAD:                                                                                                         # OLD # codegen/lowerer.py:174
                                                                if x.arg.idx >= 0:                                                                                                                 # OLD # codegen/lowerer.py:185
                                                                  for oidx, ridx in zip(self.idxs, self.ridxs):                                                                                    # OLD # codegen/lowerer.py:186
                                                                    if oidx != ridx: valid = valid * oidx.eq(0)                                                                                    # OLD # codegen/lowerer.py:187
                                                                  has_valid = valid.op is not UOps.CONST or valid.arg is not True                                                                  # OLD # codegen/lowerer.py:188
                                                                return UOp(UOps.STORE, None, (buf, idx, self.to_uop(x.src[0])) + ((valid,) if has_valid else ()))                                  # OLD # codegen/lowerer.py:189
                                                                  barrier = (UOp(UOps.BARRIER, None, (self.to_uop(x.src[0]),)),) if len(x.src) else ()                                             # OLD # codegen/lowerer.py:175
                                                                  load_dtype = x.arg.dtype.scalar()                                                                                                # OLD # codegen/lowerer.py:176
                                                                  if idx.dtype == dtypes.int.vec(3):                                                                                               # OLD # codegen/lowerer.py:177
                                                                  return UOp(UOps.LOAD, load_dtype, (buf, idx) + ((UOp.const(load_dtype, 0), valid) if has_valid else ()) + barrier)               # OLD # codegen/lowerer.py:183
                                                                  dtype = x.arg.dtype.base if isinstance(x.arg.dtype, ImageDType) else x.arg.dtype                                                 # OLD # codegen/lowerer.py:167
                                                                  return valid.where(UOp.const(dtype, x.arg.val), UOp.const(dtype, 0))                                                             # OLD # codegen/lowerer.py:168
                                                              if x.op is MetaOps.KERNEL: return UOp(UOps.SINK, src=in_uops)                                                                              # codegen/lowerer.py:192
                                                              if x.op is UnaryOps.CAST: return UOp(UOps.CAST, x.arg.scalar(), in_uops)                                                                   # codegen/lowerer.py:193
                                                              if x.op is UnaryOps.BITCAST: return UOp(UOps.BITCAST, x.arg.scalar(), in_uops)                                                             # codegen/lowerer.py:194
                                                              if x.op in ReduceOps:                                                                                                                      # codegen/lowerer.py:195
                                                              return in_uops[0].alu(x.op, *in_uops[1:])                                                                                                  # codegen/lowerer.py:213

                                        class UOpGraph:                                                                                                                                                  # codegen/uopgraph.py:497
                                          def __init__(self, sink:Union[UOp, List[UOp]], opts:Optional[Renderer]=None):                                                                                  # codegen/uopgraph.py:498
                                            self.sink: UOp = sink if isinstance(sink, UOp) else UOp(UOps.SINK, None, tuple(sink))                                                                        # codegen/uopgraph.py:499
                                            assert self.sink.op is UOps.SINK, f"sink isn't sink, it's {self.sink.op}"                                                                                    # codegen/uopgraph.py:500
                                            self._uops: Optional[List[UOp]] = None                                                                                                                       # codegen/uopgraph.py:502
                                            self.opts = opts                                                                                                                                             # codegen/uopgraph.py:503
                                            self.folder = constant_folder                                                                                                                                # codegen/uopgraph.py:504
                                            if TRANSCENDENTAL >= 2 or (opts is not None and TRANSCENDENTAL >= 1 and opts.device in {"CLANG", "LLVM"}):                                                   # codegen/uopgraph.py:505

                                              self.folder = self.folder + transcendental_folding                                                                                                         # codegen/uopgraph.py:506

                                                class PatternMatcher:                                                                                                                                    # codegen/uops.py:194
                                                  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none                                                                               # codegen/uops.py:205
                                                  def __add__(self, more:PatternMatcher): return PatternMatcher(self.patterns+more.patterns)

                                      if DEBUG >= 5: self.uops.print()                                                                                                                                   # codegen/kernel.py:733

                                      if getenv("GRAPHUOPS"): self.uops.graph()                                                                                                                          # codegen/kernel.py:734

                                      return self                                                                                                                                                        # codegen/kernel.py:735

                                self.uops.linearize(self.opts.extra_matcher)                                                                                                                             # codegen/kernel.py:739

                                  class UOpGraph:                                                                                                                                                        # codegen/uopgraph.py:497
                                    def linearize(self, extra_pm:Optional[PatternMatcher]=None, skip_check=False) -> UOpGraph:                                                                           # codegen/uopgraph.py:524
                                      acc_number = 0                                                                                                                                                     # codegen/uopgraph.py:526
                                      sink = graph_rewrite(self.sink, self.folder)                                                                                                                       # codegen/uopgraph.py:532

                                        def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:                                                                                                           # codegen/uopgraph.py:486
                                          nodes: Dict[Tuple, UOp] = {}                                                                                                                                   # codegen/uopgraph.py:487
                                          replace: Dict[UOp, UOp] = {}                                                                                                                                   # codegen/uopgraph.py:488
                                          return __inner_rewrite(sink)                                                                                                                                   # codegen/uopgraph.py:495

                                            def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:                                                                                                       # codegen/uopgraph.py:486
                                              def __inner_rewrite(n:UOp) -> UOp:                                                                                                                         # codegen/uopgraph.py:489
                                                if n in replace: return replace[n]                                                                                                                       # codegen/uopgraph.py:490
                                                replace_source = (n.op, n.dtype, tuple(__inner_rewrite(y) for y in n.src), n.arg)                                                                        # codegen/uopgraph.py:491

                                              def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:                                                                                                     # codegen/uopgraph.py:486
                                                def __inner_rewrite(n:UOp) -> UOp:                                                                                                                       # codegen/uopgraph.py:489
                                                  if n in replace: return replace[n]                                                                                                               # OLD # codegen/uopgraph.py:490
                                                  replace_source = (n.op, n.dtype, tuple(__inner_rewrite(y) for y in n.src), n.arg)                                                                # OLD # codegen/uopgraph.py:491
                                                  if found := nodes.get(replace_source): replace[n] = found                                                                                              # codegen/uopgraph.py:492

                                                    # @dataclass(frozen=True, init=False, repr=False, eq=False)                                                                                          # dtype.py:31
                                                    class PtrDType(DType):
                                                      def __hash__(self): return super().__hash__()                                                                                                      # dtype.py:33

                                                  else: nodes[replace_source] = replace[n] = found = __inner_rewrite(new_x) if (new_x := pm.rewrite(x:=UOp(*replace_source))) else x                     # codegen/uopgraph.py:493

                                                    class PatternMatcher:                                                                                                                                # codegen/uops.py:194
                                                      def rewrite(self, uop:UOp) -> Optional[UOp]:                                                                                                       # codegen/uops.py:207
                                                        for p,fxn in itertools.chain(self.pdict[(uop.op, uop.arg)], self.pdict[(uop.op, None)]):                                                         # codegen/uops.py:208
                                                        return None                                                                                                                                      # codegen/uops.py:210

                                                  return found                                                                                                                                           # codegen/uopgraph.py:494

                                              class PatternMatcher:                                                                                                                                      # codegen/uops.py:194
                                                def rewrite(self, uop:UOp) -> Optional[UOp]:                                                                                                             # codegen/uops.py:207
                                                  for p,fxn in itertools.chain(self.pdict[(uop.op, uop.arg)], self.pdict[(uop.op, None)]):                                                         # OLD # codegen/uops.py:208
                                                    if (matches := _match(uop, p, {})) and (ret:=fxn(**matches[0])) is not None: return ret # NOTE: if it returns None, we keep trying to match          # codegen/uops.py:209

                                                      def _match(uop:UOp, pat:UPat, store:Dict[str, UOp]) -> List[Dict[str, UOp]]:                                                                       # codegen/uops.py:180
                                                        if (pat.name is not None and store.setdefault(pat.name, uop) is not uop) or \                                                                    # codegen/uops.py:181
                                                           (pat.dtype is not None and uop.dtype not in pat.dtype) or \
                                                           (pat.arg is not None and pat.arg != uop.arg) or \
                                                           (pat.op is not None and uop.op not in pat.op): return []
                                                        if pat.src is None: return [store]                                                                                                               # codegen/uops.py:185

                                              def _match(uop:UOp, pat:UPat, store:Dict[str, UOp]) -> List[Dict[str, UOp]]:                                                                               # codegen/uops.py:180
                                                if (pat.name is not None and store.setdefault(pat.name, uop) is not uop) or \                                                                      # OLD # codegen/uops.py:181
                                                   (pat.dtype is not None and uop.dtype not in pat.dtype) or \
                                                   (pat.arg is not None and pat.arg != uop.arg) or \
                                                   (pat.op is not None and uop.op not in pat.op): return []
                                                if pat.src is None: return [store]                                                                                                                 # OLD # codegen/uops.py:185
                                                res: List[Dict[str, UOp]] = []                                                                                                                           # codegen/uops.py:186
                                                for vp in pat.src:                                                                                                                                       # codegen/uops.py:187
                                                  if pat.allowed_len != 0 and len(uop.src) != pat.allowed_len: return []                                                                                 # codegen/uops.py:188

                                              def _match(uop:UOp, pat:UPat, store:Dict[str, UOp]) -> List[Dict[str, UOp]]:                                                                               # codegen/uops.py:180
                                                if (pat.name is not None and store.setdefault(pat.name, uop) is not uop) or \                                                                      # OLD # codegen/uops.py:181
                                                   (pat.dtype is not None and uop.dtype not in pat.dtype) or \
                                                   (pat.arg is not None and pat.arg != uop.arg) or \
                                                   (pat.op is not None and uop.op not in pat.op): return []
                                                if pat.src is None: return [store]                                                                                                                 # OLD # codegen/uops.py:185
                                                res: List[Dict[str, UOp]] = []                                                                                                                     # OLD # codegen/uops.py:186
                                                for vp in pat.src:                                                                                                                                 # OLD # codegen/uops.py:187
                                                  if pat.allowed_len != 0 and len(uop.src) != pat.allowed_len: return []                                                                           # OLD # codegen/uops.py:188
                                                  new_stores = [store.copy()]                                                                                                                            # codegen/uops.py:189
                                                  for uu, vv in zip(uop.src, vp): new_stores = [rstore for nstore in new_stores for rstore in _match(uu, vv, nstore)]                                    # codegen/uops.py:190

                                                  res.extend(new_stores)                                                                                                                                 # codegen/uops.py:191
                                                return res                                                                                                                                               # codegen/uops.py:192

                                              @dataclass(frozen=True, eq=False)                                                                                                                          # codegen/uops.py:32
                                              class UOp:
                                                @functools.cached_property                                                                                                                               # codegen/uops.py:111
                                                def vmin(self) -> UOp: return x if (x:=self._min_max[0]) is not None and not math.isnan(x.arg) else self.sconst(dtypes.min(cast(DType, self.dtype)))

                                                  @dataclass(frozen=True, eq=False)                                                                                                                      # codegen/uops.py:32
                                                  class UOp:
                                                    @functools.cached_property                                                                                                                           # codegen/uops.py:115
                                                    def _min_max(self) -> Tuple[Optional[UOp], Optional[UOp]]:
                                                      if self.op is UOps.DEFINE_VAR: return self.src[0], self.src[1] if isinstance(self.src[1].arg, int) else None                                       # codegen/uops.py:117
                                                      if self.op is UOps.RANGE: return self.src[0], self.const(self.src[1].arg-1) if isinstance(self.src[1].arg, int) else None                          # codegen/uops.py:118
                                                      if self.op is UOps.SPECIAL: return self.const(0), self.const(self.arg[1]-1) if isinstance(self.arg[1], int) else None                              # codegen/uops.py:120
                                                      if self.op is UOps.CONST: return self, self                                                                                                        # codegen/uops.py:121
                                                      if self.op is UOps.ALU and cast(DType, self.dtype).count == 1:                                                                                     # codegen/uops.py:122
                                                        s0,s1 = [cast(UOp, self.src[i] if i < len(self.src) else None) for i in range(2)]                                                                # codegen/uops.py:123
                                                        if self.arg is UnaryOps.NEG and self.dtype != dtypes.bool and not dtypes.is_unsigned(cast(DType, self.dtype)):                                   # codegen/uops.py:124
                                                        if self.arg is BinaryOps.ADD: return self.sconst(s0.vmin.arg+s1.vmin.arg), self.sconst(s0.vmax.arg+s1.vmax.arg)                                  # codegen/uops.py:126

                                                    @dataclass(frozen=True, eq=False)                                                                                                                    # codegen/uops.py:32
                                                    class UOp:
                                                      @functools.cached_property                                                                                                                         # codegen/uops.py:115
                                                      def _min_max(self) -> Tuple[Optional[UOp], Optional[UOp]]:
                                                        if self.op is UOps.DEFINE_VAR: return self.src[0], self.src[1] if isinstance(self.src[1].arg, int) else None                               # OLD # codegen/uops.py:117
                                                        if self.op is UOps.RANGE: return self.src[0], self.const(self.src[1].arg-1) if isinstance(self.src[1].arg, int) else None                  # OLD # codegen/uops.py:118
                                                        if self.op is UOps.SPECIAL: return self.const(0), self.const(self.arg[1]-1) if isinstance(self.arg[1], int) else None                      # OLD # codegen/uops.py:120
                                                        if self.op is UOps.CONST: return self, self                                                                                                # OLD # codegen/uops.py:121
                                                        if self.op is UOps.ALU and cast(DType, self.dtype).count == 1:                                                                             # OLD # codegen/uops.py:122
                                                          s0,s1 = [cast(UOp, self.src[i] if i < len(self.src) else None) for i in range(2)]                                                        # OLD # codegen/uops.py:123
                                                          if self.arg is UnaryOps.NEG and self.dtype != dtypes.bool and not dtypes.is_unsigned(cast(DType, self.dtype)):                           # OLD # codegen/uops.py:124
                                                          if self.arg is BinaryOps.ADD: return self.sconst(s0.vmin.arg+s1.vmin.arg), self.sconst(s0.vmax.arg+s1.vmax.arg)                          # OLD # codegen/uops.py:126
                                                        return None, None                                                                                                                                # codegen/uops.py:140

                                                    class dtypes:                                                                                                                                        # dtype.py:38
                                                      @staticmethod                                                                                                                                      # dtype.py:56
                                                      def min(dtype:DType):
                                                        if dtypes.is_int(dtype): return 0 if dtypes.is_unsigned(dtype) else -2**(dtype.itemsize*8-1)                                                     # dtype.py:57

                                                    @dataclass(frozen=True, eq=False)                                                                                                                    # codegen/uops.py:32
                                                    class UOp:
                                                      def sconst(self:Union[UOp, DType, None], b:ConstType|Variable):                                                                                    # codegen/uops.py:73
                                                        return UOp._const(cast(DType, self.dtype if isinstance(self, UOp) else self).scalar() if self is not None else self, b)                          # codegen/uops.py:74

                                                          @dataclass(frozen=True, eq=False)                                                                                                              # codegen/uops.py:32
                                                          class UOp:
                                                            @functools.cached_property                                                                                                                   # codegen/uops.py:113
                                                            def vmax(self) -> UOp: return x if (x:=self._min_max[1]) is not None and not math.isnan(x.arg) else self.sconst(dtypes.max(cast(DType, self.dtype)))

                                                              class dtypes:                                                                                                                              # dtype.py:38
                                                                @staticmethod                                                                                                                            # dtype.py:60
                                                                def max(dtype:DType):
                                                                  if dtypes.is_int(dtype): return (2**(dtype.itemsize*8-(0 if dtypes.is_unsigned(dtype) else 1)))-1                                      # dtype.py:61

                                      UOpGraph.cnt += 1                                                                                                                                                  # codegen/uopgraph.py:535
                                      if UOpGraph.cnt != getenv("DEBUG_EXPAND", 0):                                                                                                                      # codegen/uopgraph.py:536

                                        sink = graph_rewrite(sink, self.folder+expander+float4_folding if self.opts is not None and self.opts.supports_float4 else self.folder+expander)                 # codegen/uopgraph.py:537

                                    def do_expand(root:UOp):                                                                                                                                             # codegen/uopgraph.py:363
                                      expands = [x for x in root.src if x.op is UOps.EXPAND]                                                                                                             # codegen/uopgraph.py:364
                                      if len(expands) == 0: return None                                                                                                                                  # codegen/uopgraph.py:365

                                    def create_gate(root:UOp) -> Optional[UOp]:                                                                                                                          # codegen/uopgraph.py:433
                                      return None if len(root.src) == 3 or (ret:=_gate_srcs(root, root.src[3])) is root else ret                                                                         # codegen/uopgraph.py:438

                                    def fold_expanded(ex, buf):                                                                                                                                          # codegen/uopgraph.py:14
                                      if buf.dtype != PtrDType(dtypes.float) and buf.dtype != PtrDType(dtypes.half) and not isinstance(buf.dtype, ImageDType): return None                               # codegen/uopgraph.py:15

                                        # @dataclass(frozen=True, init=False, repr=False, eq=False)                                                                                                      # dtype.py:31
                                        class PtrDType(DType):
                                          def __ne__(self, dt): return not (self == dt)                                                                                                                  # dtype.py:35

                                            # @dataclass(frozen=True, init=False, repr=False, eq=False)                                                                                                  # dtype.py:31
                                            class PtrDType(DType):
                                              def __eq__(self, dt): return self.priority==dt.priority and self.itemsize==dt.itemsize and self.name==dt.name and self.count==dt.count                     # dtype.py:34

                                        sink = graph_rewrite(sink, self.folder+expander+reducer)                                                                                                         # codegen/uopgraph.py:538

                                    def no_vectorized_alu(alu):                                                                                                                                          # codegen/uopgraph.py:427
                                      if alu.dtype.count == 1: return None                                                                                                                               # codegen/uopgraph.py:428

                                    def delete_redundant_gates(root:UOp) -> Optional[UOp]:                                                                                                               # codegen/uopgraph.py:460
                                      if len(root.src) == 3 or (gate:=find_gate(root)) is None or gate.src[0] is not root.src[3]: return None                                                            # codegen/uopgraph.py:465

                                      if extra_pm: sink = graph_rewrite(sink, self.folder+extra_pm)                                                                                                      # codegen/uopgraph.py:541
                                      children: Dict[UOp, List[UOp]] = {}                                                                                                                                # codegen/uopgraph.py:545
                                      in_degree: Dict[UOp, int] = {}                                                                                                                                     # codegen/uopgraph.py:546
                                      get_children_dfs(sink, children, in_degree)                                                                                                                        # codegen/uopgraph.py:547

                                        def get_children_dfs(u:UOp, children:Dict[UOp, List[UOp]], in_degree:Dict[UOp, int]):                                                                            # codegen/uopgraph.py:478
                                          if u in children: return                                                                                                                                       # codegen/uopgraph.py:479
                                          children[u] = []                                                                                                                                               # codegen/uopgraph.py:480
                                          for x in u.src:                                                                                                                                                # codegen/uopgraph.py:481
                                            get_children_dfs(x, children, in_degree)                                                                                                                     # codegen/uopgraph.py:482

                                          def get_children_dfs(u:UOp, children:Dict[UOp, List[UOp]], in_degree:Dict[UOp, int]):                                                                          # codegen/uopgraph.py:478
                                            if u in children: return                                                                                                                               # OLD # codegen/uopgraph.py:479
                                            children[u] = []                                                                                                                                       # OLD # codegen/uopgraph.py:480
                                            for x in u.src:                                                                                                                                        # OLD # codegen/uopgraph.py:481
                                              get_children_dfs(x, children, in_degree)                                                                                                             # OLD # codegen/uopgraph.py:482
                                            in_degree[u] = len(u.src)                                                                                                                                    # codegen/uopgraph.py:484

                                              def get_children_dfs(u:UOp, children:Dict[UOp, List[UOp]], in_degree:Dict[UOp, int]):                                                                      # codegen/uopgraph.py:478
                                                if u in children: return                                                                                                                           # OLD # codegen/uopgraph.py:479
                                                children[u] = []                                                                                                                                   # OLD # codegen/uopgraph.py:480
                                                for x in u.src:                                                                                                                                    # OLD # codegen/uopgraph.py:481
                                                  get_children_dfs(x, children, in_degree)                                                                                                         # OLD # codegen/uopgraph.py:482
                                                  children[x].append(u)                                                                                                                                  # codegen/uopgraph.py:483

                                      scope_children = {p:get_recursive_children(p, END_FOR_UOP[p.op][0]) for p in reversed(in_degree) if p.op in END_FOR_UOP}                                           # codegen/uopgraph.py:555

                                    class UOpGraph:                                                                                                                                                      # codegen/uopgraph.py:497
                                      def linearize(self, extra_pm:Optional[PatternMatcher]=None, skip_check=False) -> UOpGraph:                                                                         # codegen/uopgraph.py:524
                                        @functools.lru_cache(None)                                                                                                                                       # codegen/uopgraph.py:550
                                        def get_recursive_children(x:UOp, end:UOps, include_self=False) -> Set[UOp]:
                                          if x.op is UOps.SINK: return set()                                                                                                                             # codegen/uopgraph.py:551
                                          return set.union({x} if include_self else set(), *([get_recursive_children(u, end, True) for u in children[x] if x.op is not end]))                            # codegen/uopgraph.py:552

                                      queue:List[Tuple[int, UOp]] = []                                                                                                                                   # codegen/uopgraph.py:557
                                      for u in children:                                                                                                                                                 # codegen/uopgraph.py:565
                                        if in_degree[u] == 0: push(u)                                                                                                                                    # codegen/uopgraph.py:566

                                          class UOpGraph:                                                                                                                                                # codegen/uopgraph.py:497
                                            def linearize(self, extra_pm:Optional[PatternMatcher]=None, skip_check=False) -> UOpGraph:                                                                   # codegen/uopgraph.py:524
                                              def push(u:UOp):                                                                                                                                           # codegen/uopgraph.py:558
                                                priority = 0                                                                                                                                             # codegen/uopgraph.py:559
                                                for l, ss in scope_children.items():                                                                                                                     # codegen/uopgraph.py:561
                                                  if l.op is UOps.RANGE and u in ss: priority -= l.arg[0]*1000 + l.arg[1]                                                                                # codegen/uopgraph.py:562
                                                heapq.heappush(queue, (priority, u))                                                                                                                     # codegen/uopgraph.py:563

                                    @dataclass(frozen=True, eq=False)                                                                                                                                    # codegen/uops.py:32
                                    class UOp:
                                      def __lt__(self, x:UOp): return self.cmp_tuple < x.cmp_tuple                                                                                                       # codegen/uops.py:45

                                        @dataclass(frozen=True, eq=False)                                                                                                                                # codegen/uops.py:32
                                        class UOp:
                                          @functools.cached_property                                                                                                                                     # codegen/uops.py:41
                                          def cmp_tuple(self):
                                            return (self.op.value, (self.arg if self.op is not UOps.DEFINE_VAR else self.arg.expr) if self.op is not UOps.ALU else \                                     # codegen/uops.py:43
                                                    self.arg.value, self.dtype, self.src)

                                      scope_end: Dict[UOp, UOp] = {}                                                                                                                                     # codegen/uopgraph.py:568
                                      self._uops = []                                                                                                                                                    # codegen/uopgraph.py:569
                                      while queue:                                                                                                                                                       # codegen/uopgraph.py:570
                                        p,x = heapq.heappop(queue)                                                                                                                                       # codegen/uopgraph.py:571

                                        if DEBUG >= 7: print(f"{p:5d}",x)                                                                                                                                # codegen/uopgraph.py:572

                                        if x in scope_children: scope_end[x] = x                                                                                                                         # codegen/uopgraph.py:573
                                        if x.op is UOps.DEFINE_ACC:                                                                                                                                      # codegen/uopgraph.py:574
                                        else: self._uops.append(x)                                                                                                                                       # codegen/uopgraph.py:577
                                        for u, ss in scope_children.items():                                                                                                                             # codegen/uopgraph.py:578
                                          if x in ss:                                                                                                                                                    # codegen/uopgraph.py:579
                                        for u in children[x]:                                                                                                                                            # codegen/uopgraph.py:582
                                          in_degree[u] -= 1                                                                                                                                              # codegen/uopgraph.py:583
                                          if in_degree[u] == 0: push(u)                                                                                                                                  # codegen/uopgraph.py:584

                                            ss.remove(x)                                                                                                                                                 # codegen/uopgraph.py:580
                                            if len(ss) == 0: scope_end[u] = x                                                                                                                            # codegen/uopgraph.py:581

                                        for u in children[x]:                                                                                                                                      # OLD # codegen/uopgraph.py:582
                                          in_degree[u] -= 1                                                                                                                                        # OLD # codegen/uopgraph.py:583
                                          if in_degree[u] == 0: push(u)                                                                                                                            # OLD # codegen/uopgraph.py:584
                                      for u, x in scope_end.items(): self._uops.insert(self._uops.index(x)+1, UOp(END_FOR_UOP[u.op][1], None, (u,)))                                                     # codegen/uopgraph.py:587
                                      if not skip_check:                                                                                                                                                 # codegen/uopgraph.py:590
                                        bad_ops = dedup([x.op for x in self._uops if x.op in {UOps.EXPAND, UOps.CONTRACT, UOps.REDUCE}])                                                                 # codegen/uopgraph.py:591

                                        try:                                                                                                                                                             # codegen/uopgraph.py:592
                                          type_verify(self.uops)                                                                                                                                         # codegen/uopgraph.py:593

                                            class UOpGraph:                                                                                                                                              # codegen/uopgraph.py:497
                                              @property                                                                                                                                                  # codegen/uopgraph.py:513
                                              def uops(self) -> List[UOp]:
                                                if self._uops is None: self.linearize()                                                                                                                  # codegen/uopgraph.py:514
                                                return cast(List[UOp], self._uops)                                                                                                                       # codegen/uopgraph.py:515

                                            def type_verify(uops):                                                                                                                                       # codegen/uops.py:212
                                              for u in uops:                                                                                                                                             # codegen/uops.py:213
                                                uop, arg, src, dtype = u.op, u.arg, u.src, u.dtype                                                                                                       # codegen/uops.py:214
                                                if uop in {UOps.CONST, UOps.DEFINE_ACC}:                                                                                                                 # codegen/uops.py:215
                                                if uop in {UOps.CAST, UOps.BITCAST, UOps.VECTORIZE}: assert arg is None and dtype is not None # type is the output type, not an arg                      # codegen/uops.py:220
                                                if uop is UOps.CAST: assert dtype.count == 1 and len(src) == 1                                                                                           # codegen/uops.py:221
                                                if uop is UOps.VECTORIZE:                                                                                                                                # codegen/uops.py:222
                                                if uop is UOps.LOAD and len(src) > 3 and src[3].op is UOps.ALU: assert src[3].dtype == dtypes.bool and src[2].dtype == dtype                             # codegen/uops.py:225
                                                if uop is UOps.GEP: assert dtype == src[0].dtype.scalar(), f"GEP of {src[0].dtype=} should be {src[0].dtype.scalar()} != {dtype}"                        # codegen/uops.py:226
                                                if uop is UOps.STORE:                                                                                                                                    # codegen/uops.py:227
                                                if uop is UOps.ALU:                                                                                                                                      # codegen/uops.py:230
                                                  if uop is UOps.CONST:                                                                                                                                  # codegen/uops.py:216
                                                    assert dtype is not None and dtype == dtype.scalar(), f"consts should be scalar, got {dtype}"                                                        # codegen/uops.py:217

                                                    assert type(arg) is type(dtypes.as_const(arg, dtype)), f"type of {arg=} does not match {dtype}"                                                      # codegen/uops.py:218

                                                  if uop is UOps.DEFINE_ACC: assert dtype is not None and src[0].dtype == dtype, f"dtype mismatch {src[0].dtype=} != {dtype=}"                           # codegen/uops.py:219

                                                if uop in {UOps.CAST, UOps.BITCAST, UOps.VECTORIZE}: assert arg is None and dtype is not None # type is the output type, not an arg                # OLD # codegen/uops.py:220
                                                if uop is UOps.CAST: assert dtype.count == 1 and len(src) == 1                                                                                     # OLD # codegen/uops.py:221
                                                if uop is UOps.VECTORIZE:                                                                                                                          # OLD # codegen/uops.py:222
                                                if uop is UOps.LOAD and len(src) > 3 and src[3].op is UOps.ALU: assert src[3].dtype == dtypes.bool and src[2].dtype == dtype                       # OLD # codegen/uops.py:225
                                                if uop is UOps.GEP: assert dtype == src[0].dtype.scalar(), f"GEP of {src[0].dtype=} should be {src[0].dtype.scalar()} != {dtype}"                  # OLD # codegen/uops.py:226
                                                if uop is UOps.STORE:                                                                                                                              # OLD # codegen/uops.py:227
                                                if uop is UOps.ALU:                                                                                                                                # OLD # codegen/uops.py:230
                                                  if arg in UnaryOps: assert dtype == src[0].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=}"                                                  # codegen/uops.py:231
                                                  elif arg in {BinaryOps.CMPLT, BinaryOps.CMPNE}:                                                                                                        # codegen/uops.py:232
                                                  elif arg is BinaryOps.IDIV:                                                                                                                            # codegen/uops.py:235
                                                  elif arg in {BinaryOps.SHL, BinaryOps.SHR}:                                                                                                            # codegen/uops.py:238
                                                  elif arg in BinaryOps: assert dtype == src[0].dtype == src[1].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=} != {src[1].dtype=}"            # codegen/uops.py:241
                                                  assert dtype is None, f"{uop} dtype must be None, got {dtype}"                                                                                         # codegen/uops.py:228
                                                  if len(src) == 4: assert src[3].dtype == dtypes.bool, f"gate dtype mismatch {src[3].dtype} != {dtypes.bool}"                                           # codegen/uops.py:229

                                          assert self._uops[-1].op is UOps.SINK, f"didn't end with SINK, ended with {self._uops[-1]}"                                                                    # codegen/uopgraph.py:594
                                          assert len(bad_ops) == 0, f"bad UOps left in list: {bad_ops}"                                                                                                  # codegen/uopgraph.py:595
                                          assert len(all_stores := [x.src[0:2]+x.src[3:] for x in self._uops if x.op is UOps.STORE and x.src[0].op is not UOps.DEFINE_LOCAL]) \                          # codegen/uopgraph.py:598
                                            == len(dedup(all_stores)), "repeated stores in uops"

                                      self._uops = self._uops[:-1]                                                                                                                                       # codegen/uopgraph.py:606
                                      return self                                                                                                                                                        # codegen/uopgraph.py:607

                                src = self.opts.render(name:=to_function_name(ansiname:=(name_override if name_override is not None else self.name)), self.uops.uops)                                    # codegen/kernel.py:740

                                  class Kernel:                                                                                                                                                          # codegen/kernel.py:54
                                    @functools.cached_property                                                                                                                                           # codegen/kernel.py:622
                                    def name(self) -> str:
                                      name = ("r" if self.reduceop else ("C" if all(x.op in BufferOps for x in self.ast.lazyops) else "E")) + \                                                          # codegen/kernel.py:624
                                                   (f"{len(self.ast.src)}_" if len(self.ast.src) > 1 else "_") + \
                                                   colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])

                                        class Kernel:                                                                                                                                                    # codegen/kernel.py:54
                                          @property                                                                                                                                                      # codegen/kernel.py:151
                                          def reduceop(self) -> Optional[LazyOp]: return self.reduceops[0] if len(self.reduceops) > 0 else None

                                        class Kernel:                                                                                                                                                    # codegen/kernel.py:54
                                          # there's eight chunks of the shape                                                                                                                            # codegen/kernel.py:182
                                          # blue   -- global dims
                                          # cyan   -- local dims (warp ones first)
                                          #  *** self.first_reduce
                                          # green  -- reduce-local dims
                                          # white  -- reduce-late upcasted dim (self.upcast_in_mid_reduce_axes)
                                          # red    -- reduce loops
                                          #  *** self.upcasted
                                          # purple -- reduce upcasted
                                          # yellow -- normal upcasted dimensions
                                          def colors(self) -> List[str]:
                                            colors = ["blue"] * self.global_dims if not self.dont_use_locals else ["BLUE"] * self.global_dims                                                            # codegen/kernel.py:184

                                              class Kernel:                                                                                                                                              # codegen/kernel.py:54
                                                @property                                                                                                                                                # codegen/kernel.py:170
                                                def global_dims(self) -> int: return self.first_reduce-self.local_dims

                                            colors += ["cyan"] * self.local_dims                                                                                                                         # codegen/kernel.py:186
                                            colors += ["white" if i in self.upcast_in_mid_reduce_axes else "green" for i in range(self.first_reduce, self.first_reduce + self.group_for_reduces)]  # noqa: E501 # codegen/kernel.py:188

                                            colors += ["red"] * (self.first_upcast - (self.first_reduce + self.group_for_reduces))                                                                       # codegen/kernel.py:190

                                            colors += ["magenta" if self.full_shape[i] != self.sts[0].shape[i] else "yellow" for i in range(self.first_upcast, self.shape_len)]                          # codegen/kernel.py:192

                                            assert len(colors) == self.shape_len, "colors size mismatch"                                                                                                 # codegen/kernel.py:193

                                            return colors                                                                                                                                                # codegen/kernel.py:194

                                      Kernel.kernel_cnt[(function_name := to_function_name(name))] += 1                                                                                                  # codegen/kernel.py:629

                                        @functools.lru_cache(maxsize=None)                                                                                                                               # helpers.py:77
                                        def to_function_name(s:str): return ''.join([c if c in (string.ascii_letters+string.digits+'_') else f'{ord(c):02X}' for c in ansistrip(s)])

                                          def ansistrip(s:str): return re.sub('\x1b\\[(K|.*?m)', '', s)                                                                                                  # helpers.py:31

                                      suffix = f"{'n'+str(Kernel.kernel_cnt[function_name]-1)}" if Kernel.kernel_cnt[function_name] > 1 else ""                                                          # codegen/kernel.py:630
                                      return name+colored(suffix, 'BLACK')                                                                                                                               # codegen/kernel.py:631

                                  class CStyleLanguage(Renderer):                                                                                                                                        # renderer/cstyle.py:10
                                    def render(self, name:str, uops:List[UOp]) -> str:                                                                                                                   # renderer/cstyle.py:97
                                      kernel = []                                                                                                                                                        # renderer/cstyle.py:98
                                      bufs: Dict[UOp, Tuple[str, Tuple[DType, bool]]] = {}                                                                                                               # renderer/cstyle.py:99
                                      depth = 1                                                                                                                                                          # renderer/cstyle.py:100
                                      c: DefaultDict[str, int] = defaultdict(int)                                                                                                                        # renderer/cstyle.py:103
                                      r: Dict[UOp, str] = {}                                                                                                                                             # renderer/cstyle.py:104
                                      child_count = Counter(v for ru in uops for v in ru.src)                                                                                                            # renderer/cstyle.py:113
                                      seen_vars = set()                                                                                                                                                  # renderer/cstyle.py:115
                                      for u in uops:                                                                                                                                                     # renderer/cstyle.py:116
                                        uop,dtype,src,args = u.op,u.dtype,u.src,u.arg                                                                                                                    # renderer/cstyle.py:117
                                        if uop is UOps.IF:                                                                                                                                               # renderer/cstyle.py:119
                                        elif uop is UOps.BARRIER: kk(self.barrier)                                                                                                                       # renderer/cstyle.py:122
                                        elif uop in {UOps.ENDRANGE, UOps.ENDIF}:                                                                                                                         # renderer/cstyle.py:123
                                        elif uop is UOps.STORE:                                                                                                                                          # renderer/cstyle.py:126
                                          assert dtype is not None, f"None dtype for uop {uop}"                                                                                                          # renderer/cstyle.py:133
                                          if uop is UOps.RANGE:                                                                                                                                          # renderer/cstyle.py:134
                                          elif uop is UOps.ALU:                                                                                                                                          # renderer/cstyle.py:137
                                          elif uop is UOps.SPECIAL:                                                                                                                                      # renderer/cstyle.py:146
                                          elif uop is UOps.DEFINE_VAR:                                                                                                                                   # renderer/cstyle.py:149
                                          elif uop is UOps.LOAD:                                                                                                                                         # renderer/cstyle.py:154
                                          elif uop is UOps.PHI:                                                                                                                                          # renderer/cstyle.py:159
                                          elif uop in {UOps.CAST, UOps.BITCAST, UOps.VECTORIZE}:                                                                                                         # renderer/cstyle.py:162
                                          elif uop is UOps.DEFINE_LOCAL:                                                                                                                                 # renderer/cstyle.py:172
                                          elif uop is UOps.DEFINE_GLOBAL:                                                                                                                                # renderer/cstyle.py:175
                                            bufs[u] = (nm:=f"data{args}", (dtype, False))                                                                                                                # renderer/cstyle.py:176
                                            r[u] = nm                                                                                                                                                    # renderer/cstyle.py:177
                                          elif uop is UOps.WMMA: kk(f"{self.render_dtype(dtype)} {ssa('wmma',u)} = __{args[0]}({r[src[0]]}, {r[src[1]]}, {r[src[2]]});")                                 # renderer/cstyle.py:178
                                          elif uop is UOps.DEFINE_ACC: kk(f"{self.render_dtype(dtype)} {ssa('acc',u)} = {r[src[0]]};")                                                                   # renderer/cstyle.py:179
                                          elif uop is UOps.CONST: r[u] = self.render_const(args, dtype) if args >= 0 else f"({self.render_const(args, dtype)})"                                          # renderer/cstyle.py:180

                                            class CStyleLanguage(Renderer):                                                                                                                              # renderer/cstyle.py:10
                                              # returns a str expression of the const with the given type                                                                                                # renderer/cstyle.py:49
                                              def render_const(self, x:ConstType, dtype:DType) -> str:
                                                assert dtype.count == 1, f"consts should be scalar, got {dtype}"                                                                                         # renderer/cstyle.py:50
                                                if math.isnan(x): val = self.nan                                                                                                                         # renderer/cstyle.py:51
                                                elif math.isinf(x): val = ("-" if x < 0 else "") + self.infinity                                                                                         # renderer/cstyle.py:52
                                                elif dtype == dtypes.bool: val = "1" if x else "0"                                                                                                       # renderer/cstyle.py:53
                                                elif dtype == dtypes.float: val = f"{x}f"                                                                                                                # renderer/cstyle.py:54
                                                elif dtype == dtypes.uint64: val = f"{x}ULL"                                                                                                             # renderer/cstyle.py:55
                                                else: val = str(x)                                                                                                                                       # renderer/cstyle.py:56
                                                return (self.render_cast(val, dtype) if dtype not in [dtypes.float, dtypes.int, dtypes.bool] else val)                                                   # renderer/cstyle.py:57

                                            kk(f"for (int {(expr := ssa('ridx',u))} = {r[src[0]]}; {expr} < {r[src[1]]}; {expr}++) {{")                                                                  # renderer/cstyle.py:135

                                              class CStyleLanguage(Renderer):                                                                                                                            # renderer/cstyle.py:10
                                                def render(self, name:str, uops:List[UOp]) -> str:                                                                                                       # renderer/cstyle.py:97
                                                  def ssa(prefix:str, u:Optional[UOp]=None):                                                                                                             # renderer/cstyle.py:106
                                                    ret = f"{prefix}{c[prefix]}"                                                                                                                         # renderer/cstyle.py:108
                                                    if u is not None: r[u] = ret                                                                                                                         # renderer/cstyle.py:109
                                                    c[prefix] += 1                                                                                                                                       # renderer/cstyle.py:110
                                                    return ret                                                                                                                                           # renderer/cstyle.py:111

                                              class CStyleLanguage(Renderer):                                                                                                                            # renderer/cstyle.py:10
                                                def render(self, name:str, uops:List[UOp]) -> str:                                                                                                       # renderer/cstyle.py:97
                                                  def kk(s): kernel.append("  "*depth+s)                                                                                                                 # renderer/cstyle.py:101

                                            depth += 1                                                                                                                                                   # renderer/cstyle.py:136
                                          elif uop is UOps.ALU:                                                                                                                                    # OLD # renderer/cstyle.py:137
                                          elif uop is UOps.SPECIAL:                                                                                                                                # OLD # renderer/cstyle.py:146
                                          elif uop is UOps.DEFINE_VAR:                                                                                                                             # OLD # renderer/cstyle.py:149
                                          elif uop is UOps.LOAD:                                                                                                                                   # OLD # renderer/cstyle.py:154
                                            val = self.render_load(dtype, r[src[0]], src[0].dtype, strip_parens(r[src[1]]), src[0].op is UOps.DEFINE_LOCAL)                                              # renderer/cstyle.py:155

                                              def strip_parens(fst:str): return fst[1:-1] if fst[0] == '(' and fst[-1] == ')' and fst[1:-1].find('(') <= fst[1:-1].find(')') else fst                    # helpers.py:37

                                              class CStyleLanguage(Renderer):                                                                                                                            # renderer/cstyle.py:10
                                                # returns a str expression of the loaded value with the output type                                                                                      # renderer/cstyle.py:60
                                                def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
                                                  if isinstance(buf_dtype, ImageDType):                                                                                                                  # renderer/cstyle.py:61
                                                  if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and output_dtype.scalar() != dtypes.float16:                                               # renderer/cstyle.py:64
                                                  if output_dtype.count > 1:                                                                                                                             # renderer/cstyle.py:66
                                                  return f"*({buf_name}+{idx})" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}]"                                                                    # renderer/cstyle.py:68

                                            if len(src) > 3 and src[3].op is UOps.ALU: val = self.code_for_op[TernaryOps.WHERE](r[src[3]], val, r[src[2]], dtype)                                        # renderer/cstyle.py:157
                                            kk(f"{self.render_dtype(dtype)} {ssa('val',u)} = {val};")                                                                                                    # renderer/cstyle.py:158

                                              class CStyleLanguage(Renderer):                                                                                                                            # renderer/cstyle.py:10
                                                def render_dtype(self, var_dtype:DType) -> str:                                                                                                          # renderer/cstyle.py:94
                                                  return self.type_map.get(scalar:=var_dtype.scalar(), scalar.name) + (str(var_dtype.count) if (var_dtype.count) > 1 else "")                            # renderer/cstyle.py:95

                                            if args in {BinaryOps.ADD,BinaryOps.MUL,BinaryOps.XOR}: operands = [strip_parens(r[v]) if v.arg == args else r[v]for v in src]                               # renderer/cstyle.py:139
                                            val = self.code_for_op[args](*operands, dtype)                                                                                                               # renderer/cstyle.py:141
                                            assert child_count[u] != 0, f"childless ALU op found {u}"                                                                                                    # renderer/cstyle.py:142
                                            if child_count[u] <= 1 and args is not BinaryOps.MAX and not getenv("EXPAND_SSA"): r[u] = val                                                                # renderer/cstyle.py:144

                                          assert src[0].dtype is not None and src[2].dtype is not None                                                                                                   # renderer/cstyle.py:127
                                          if src[0].op is UOps.DEFINE_GLOBAL: bufs[src[0]] = (bufs[src[0]][0], (bufs[src[0]][1][0], True))                                                               # renderer/cstyle.py:129
                                          rendered_store = self.render_store(r[src[0]], src[0].dtype, r[src[2]], src[2].dtype, strip_parens(r[src[1]]), src[0].op is UOps.DEFINE_LOCAL)                  # renderer/cstyle.py:130

                                            class CStyleLanguage(Renderer):                                                                                                                              # renderer/cstyle.py:10
                                              # returns a str statement that does the store                                                                                                              # renderer/cstyle.py:82
                                              def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, local=False) -> str:
                                                if isinstance(buf_dtype, ImageDType):                                                                                                                    # renderer/cstyle.py:83
                                                if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and var_dtype.scalar() != dtypes.float16:                                                    # renderer/cstyle.py:86
                                                if var_dtype.count > 1:                                                                                                                                  # renderer/cstyle.py:88
                                                return f"*({buf_name}+{idx}) = {var_name};" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}] = {var_name};"                                          # renderer/cstyle.py:91

                                          kk(f"if ({r[src[3]]}) {{ {rendered_store} }}" if len(src) > 3 else rendered_store)                                                                             # renderer/cstyle.py:131

                                          depth -= 1                                                                                                                                                     # renderer/cstyle.py:124
                                          kk("}")                                                                                                                                                        # renderer/cstyle.py:125

                                        elif uop is UOps.STORE:                                                                                                                                    # OLD # renderer/cstyle.py:126
                                          assert dtype is not None, f"None dtype for uop {uop}"                                                                                                    # OLD # renderer/cstyle.py:133
                                          if uop is UOps.RANGE:                                                                                                                                    # OLD # renderer/cstyle.py:134
                                          elif uop is UOps.ALU:                                                                                                                                    # OLD # renderer/cstyle.py:137
                                          elif uop is UOps.SPECIAL:                                                                                                                                # OLD # renderer/cstyle.py:146
                                          elif uop is UOps.DEFINE_VAR:                                                                                                                             # OLD # renderer/cstyle.py:149
                                          elif uop is UOps.LOAD:                                                                                                                                   # OLD # renderer/cstyle.py:154
                                          elif uop is UOps.PHI:                                                                                                                                    # OLD # renderer/cstyle.py:159
                                          elif uop in {UOps.CAST, UOps.BITCAST, UOps.VECTORIZE}:                                                                                                   # OLD # renderer/cstyle.py:162
                                          elif uop is UOps.DEFINE_LOCAL:                                                                                                                           # OLD # renderer/cstyle.py:172
                                          elif uop is UOps.DEFINE_GLOBAL:                                                                                                                          # OLD # renderer/cstyle.py:175
                                            bufs[u] = (nm:=f"data{args}", (dtype, False))                                                                                                          # OLD # renderer/cstyle.py:176
                                            r[u] = nm                                                                                                                                              # OLD # renderer/cstyle.py:177
                                          elif uop is UOps.WMMA: kk(f"{self.render_dtype(dtype)} {ssa('wmma',u)} = __{args[0]}({r[src[0]]}, {r[src[1]]}, {r[src[2]]});")                           # OLD # renderer/cstyle.py:178
                                          elif uop is UOps.DEFINE_ACC: kk(f"{self.render_dtype(dtype)} {ssa('acc',u)} = {r[src[0]]};")                                                             # OLD # renderer/cstyle.py:179
                                          elif uop is UOps.CONST: r[u] = self.render_const(args, dtype) if args >= 0 else f"({self.render_const(args, dtype)})"                                    # OLD # renderer/cstyle.py:180
                                            kk(f"for (int {(expr := ssa('ridx',u))} = {r[src[0]]}; {expr} < {r[src[1]]}; {expr}++) {{")                                                            # OLD # renderer/cstyle.py:135
                                            depth += 1                                                                                                                                             # OLD # renderer/cstyle.py:136
                                            val = self.render_load(dtype, r[src[0]], src[0].dtype, strip_parens(r[src[1]]), src[0].op is UOps.DEFINE_LOCAL)                                        # OLD # renderer/cstyle.py:155
                                            if len(src) > 3 and src[3].op is UOps.ALU: val = self.code_for_op[TernaryOps.WHERE](r[src[3]], val, r[src[2]], dtype)                                  # OLD # renderer/cstyle.py:157
                                            kk(f"{self.render_dtype(dtype)} {ssa('val',u)} = {val};")                                                                                              # OLD # renderer/cstyle.py:158
                                            if args in {BinaryOps.ADD,BinaryOps.MUL,BinaryOps.XOR}: operands = [strip_parens(r[v]) if v.arg == args else r[v]for v in src]                         # OLD # renderer/cstyle.py:139
                                            val = self.code_for_op[args](*operands, dtype)                                                                                                         # OLD # renderer/cstyle.py:141
                                            assert child_count[u] != 0, f"childless ALU op found {u}"                                                                                              # OLD # renderer/cstyle.py:142
                                            if child_count[u] <= 1 and args is not BinaryOps.MAX and not getenv("EXPAND_SSA"): r[u] = val                                                          # OLD # renderer/cstyle.py:144
                                          assert src[0].dtype is not None and src[2].dtype is not None                                                                                             # OLD # renderer/cstyle.py:127
                                          if src[0].op is UOps.DEFINE_GLOBAL: bufs[src[0]] = (bufs[src[0]][0], (bufs[src[0]][1][0], True))                                                         # OLD # renderer/cstyle.py:129
                                          rendered_store = self.render_store(r[src[0]], src[0].dtype, r[src[2]], src[2].dtype, strip_parens(r[src[1]]), src[0].op is UOps.DEFINE_LOCAL)            # OLD # renderer/cstyle.py:130
                                          kk(f"if ({r[src[3]]}) {{ {rendered_store} }}" if len(src) > 3 else rendered_store)                                                                       # OLD # renderer/cstyle.py:131
                                      return self.render_kernel(name, kernel, list(bufs.values()), uops)                                                                                                 # renderer/cstyle.py:189

                                        class ClangRenderer(CStyleLanguage):                                                                                                                             # renderer/cstyle.py:194
                                          def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:                                                                                # renderer/cstyle.py:209
                                            prefix = [_make_clang_dtype(self, dtype) for dtype in dedup(uop.dtype for uop in uops if uop.dtype is not None and uop.dtype.count>1)]                       # renderer/cstyle.py:210

                                            return super().render_kernel(function_name, kernel, bufs, uops, prefix)                                                                                      # renderer/cstyle.py:211

                                              class CStyleLanguage(Renderer):                                                                                                                            # renderer/cstyle.py:10
                                                def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:               # renderer/cstyle.py:71
                                                  tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,(dtype,_) in bufs) else ""  # noqa: E501 # renderer/cstyle.py:72
                                                  buftypes = [(name,f"{'write_only' if mutable else 'read_only'} image2d_t" if dtype.name.startswith('image') else                                       # renderer/cstyle.py:73
                                                              ("" if mutable else "const ")+self.buffer_prefix+self.render_dtype(dtype)+"*"+self.buffer_suffix if isinstance(dtype, PtrDType) else
                                                              self.arg_int_prefix if dtype == dtypes.int else None) for name,(dtype,mutable) in bufs]

                                                  prg = ''.join([f"{self.kernel_prefix}void {self.get_kernel_modifier(uops)}{function_name}(",] +                                                        # renderer/cstyle.py:76
                                                  [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
                                                  [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])

                                                    class CStyleLanguage(Renderer):                                                                                                                      # renderer/cstyle.py:10
                                                      def get_kernel_modifier(self, uops:List[UOp]) -> str: return ""                                                                                    # renderer/cstyle.py:70

                                                  return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"                                                                                         # renderer/cstyle.py:79

                                if getenv("RUN_PROCESS_REPLAY"):                                                                                                                                         # codegen/kernel.py:742
                                mem_bytes = sum(max(x.arg.dtype.itemsize * x.arg.st.real_size() for x in group) for _, group in                                                                          # codegen/kernel.py:748
                                  itertools.groupby([x for x in self.ast.lazyops if x.op in BufferOps and isinstance(x.arg, MemBuffer) and x.arg.idx >= 0],
                                                    key=lambda x: (x.op, x.arg.idx)))

                              @dataclass(frozen=True)                                                                                                                                                    # shape/shapetracker.py:10
                              class ShapeTracker:
                                def real_size(self) -> int:                                                                                                                                              # shape/shapetracker.py:40
                                  if 0 in self.shape: return 0                                                                                                                                           # shape/shapetracker.py:41

                                  idx, valid = self.expr_idxs()                                                                                                                                          # shape/shapetracker.py:42

                                  if not valid: return 0                                                                                                                                                 # shape/shapetracker.py:43

                                    class Node:                                                                                                                                                          # shape/symbolic.py:10
                                      def __bool__(self): return not (self.max == self.min == 0)                                                                                                         # shape/symbolic.py:28

                                  ret = idx.max                                                                                                                                                          # shape/shapetracker.py:45
                                  if not isinstance(ret, int): ret = ret.max  # might be represent by symbolic shape, take one more max for int max                                                      # shape/shapetracker.py:46
                                  assert isinstance(ret, int), f"ret must be integer, {ret=} isn't"                                                                                                      # shape/shapetracker.py:47
                                  return ret+1                                                                                                                                                           # shape/shapetracker.py:48

                                return Program(ansiname, src, self.opts.device, self.uops.uops, mem_estimate=mem_bytes,                                                                                  # codegen/kernel.py:751
                                               global_size=[1,1,1] if self.opts.has_local else None, local_size=[1,1,1] if self.opts.has_local else None)

                                  @dataclass                                                                                                                                                             # renderer/__init__.py:18
                                  class Program:
                                    def __post_init__(self):                                                                                                                                             # renderer/__init__.py:33
                                      if not self._ran_post_init and self.uops is not None:                                                                                                              # renderer/__init__.py:34
                                        for u in self.uops:                                                                                                                                              # renderer/__init__.py:36
                                          if u.op is UOps.DEFINE_VAR: self.vars.append(u.arg)                                                                                                            # renderer/__init__.py:37
                                          if u.op is UOps.DEFINE_GLOBAL: self.globals.append(u.arg)                                                                                                      # renderer/__init__.py:38
                                          if u.op is UOps.STORE: self.outs.extend([x.arg for x in u.src[0].sparents if x.op is UOps.DEFINE_GLOBAL])                                                      # renderer/__init__.py:39
                                          if u.op is UOps.SPECIAL:                                                                                                                                       # renderer/__init__.py:40

                                            @dataclass(frozen=True, eq=False)                                                                                                                            # codegen/uops.py:32
                                            class UOp:
                                              @property  # parents with self                                                                                                                             # codegen/uops.py:92
                                              def sparents(self) -> Set[UOp]: return set([self]).union(self.parents)

                                                @dataclass(frozen=True, eq=False)                                                                                                                        # codegen/uops.py:32
                                                class UOp:
                                                  @functools.cached_property                                                                                                                             # codegen/uops.py:90
                                                  def parents(self) -> Set[UOp]: return set.union(set(self.src), *[x.parents for x in self.src])

                                        self.vars = sorted(self.vars, key=lambda v: v.expr)                                                                                                              # renderer/__init__.py:49
                                        self.outs = sorted(dedup(self.outs))                                                                                                                             # renderer/__init__.py:50

                                        self._ran_post_init = True                                                                                                                                       # renderer/__init__.py:51

                          if getenv("FUZZ_UOPS"):                                                                                                                                                        # engine/realize.py:158

                          method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, dname=dname))                                                                                      # engine/realize.py:161

                            class CompiledRunner(Runner):                                                                                                                                                # engine/realize.py:79
                              def __init__(self, p:Program, precompiled:Optional[bytes]=None):                                                                                                           # engine/realize.py:80
                                if DEBUG >= 4: print(p.src)                                                                                                                                              # engine/realize.py:81

                                self.p:Program = p                                                                                                                                                       # engine/realize.py:82
                                self.lib:bytes = precompiled if precompiled is not None else Device[p.dname].compiler.compile_cached(p.src)                                                              # engine/realize.py:83

                                  class Compiler:                                                                                                                                                        # device.py:176
                                    def compile_cached(self, src:str) -> bytes:                                                                                                                          # device.py:179
                                      if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:                                                                                    # device.py:180

                                        def diskcache_get(table:str, key:Union[Dict, str, int]) -> Any:                                                                                                  # helpers.py:225
                                          if CACHELEVEL == 0: return None                                                                                                                                # helpers.py:226
                                          if isinstance(key, (str,int)): key = {"key": key}                                                                                                              # helpers.py:227
                                          conn = db_connection()                                                                                                                                         # helpers.py:228

                                            def db_connection():                                                                                                                                         # helpers.py:209
                                              if _db_connection is None:                                                                                                                                 # helpers.py:211
                                                os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)                                                                                                 # helpers.py:212
                                                _db_connection = sqlite3.connect(CACHEDB, timeout=60, isolation_level="IMMEDIATE")                                                                       # helpers.py:213
                                                with contextlib.suppress(sqlite3.OperationalError): _db_connection.execute("PRAGMA journal_mode=WAL").fetchone()                                         # helpers.py:216
                                                if DEBUG >= 7: _db_connection.set_trace_callback(print)                                                                                                  # helpers.py:217

                                              return _db_connection                                                                                                                                      # helpers.py:218

                                          cur = conn.cursor()                                                                                                                                            # helpers.py:229
                                          try:                                                                                                                                                           # helpers.py:230
                                            res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))                          # helpers.py:231
                                          if (val:=res.fetchone()) is not None: return pickle.loads(val[0])                                                                                              # helpers.py:234

                                      return lib                                                                                                                                                         # device.py:184

                                self.clprg = Device[p.dname].runtime(p.function_name, self.lib)                                                                                                          # engine/realize.py:84

                                  @dataclass                                                                                                                                                             # renderer/__init__.py:18
                                  class Program:
                                    @functools.cached_property                                                                                                                                           # renderer/__init__.py:64
                                    def function_name(self) -> str: return to_function_name(self.name)

                                  class ClangProgram:                                                                                                                                                    # runtime/ops_clang.py:14
                                    def __init__(self, name:str, lib:bytes):                                                                                                                             # runtime/ops_clang.py:15
                                      if DEBUG >= 6: cpu_objdump(lib)                                                                                                                                    # runtime/ops_clang.py:16

                                      self.name, self.lib = name, lib                                                                                                                                    # runtime/ops_clang.py:17
                                      with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:                                                                                                 # runtime/ops_clang.py:19
                                        pathlib.Path(cached_file_path.name).write_bytes(lib)                                                                                                             # runtime/ops_clang.py:20
                                        self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]                                                                                                         # runtime/ops_clang.py:21

                                super().__init__(p.name, p.dname, p.op_estimate, p.mem_estimate, p.lds_estimate)                                                                                         # engine/realize.py:85

                                  @dataclass                                                                                                                                                             # renderer/__init__.py:18
                                  class Program:
                                    @property                                                                                                                                                            # renderer/__init__.py:54
                                    def op_estimate(self) -> sint: return self._ops_lds[0]

                                      @dataclass                                                                                                                                                         # renderer/__init__.py:18
                                      class Program:
                                        @functools.cached_property                                                                                                                                       # renderer/__init__.py:58
                                        def _ops_lds(self) -> Tuple[sint, sint]: return (0,0) if self.uops is None else flops_mem(self.uops, ignore_indexing=True)

                                          def flops_mem(uops:List[UOp], ignore_indexing=False) -> Tuple[sint, sint]:                                                                                     # codegen/uops.py:257
                                            flops: sint = 0                                                                                                                                              # codegen/uops.py:258
                                            mem: sint = 0                                                                                                                                                # codegen/uops.py:259
                                            mults: sint = 1                                                                                                                                              # codegen/uops.py:260
                                            mult_stack: List[sint] = []                                                                                                                                  # codegen/uops.py:261
                                            dont_count: Set[UOp] = set()                                                                                                                                 # codegen/uops.py:262
                                            if ignore_indexing:                                                                                                                                          # codegen/uops.py:263
                                              for u in uops:                                                                                                                                             # codegen/uops.py:264
                                                if u.op is UOps.LOAD:                                                                                                                                    # codegen/uops.py:265
                                                elif u.op is UOps.STORE:                                                                                                                                 # codegen/uops.py:268
                                                elif u.op is UOps.IF:                                                                                                                                    # codegen/uops.py:271
                                                  dont_count = dont_count.union(u.src[1].sparents)                                                                                                       # codegen/uops.py:266

                                                  if len(u.src) > 3: dont_count = dont_count.union(u.src[2].sparents)                                                                                    # codegen/uops.py:267
                                                elif u.op is UOps.STORE:                                                                                                                           # OLD # codegen/uops.py:268
                                                  dont_count = dont_count.union(u.src[1].sparents)                                                                                                       # codegen/uops.py:269

                                                  if len(u.src) > 3: dont_count = dont_count.union(u.src[3].sparents)                                                                                    # codegen/uops.py:270
                                                elif u.op is UOps.IF:                                                                                                                              # OLD # codegen/uops.py:271
                                            for u in uops:                                                                                                                                               # codegen/uops.py:273
                                              if u.op is UOps.RANGE:                                                                                                                                     # codegen/uops.py:274
                                              elif u.op is UOps.ENDRANGE:                                                                                                                                # codegen/uops.py:277
                                              elif u.op is UOps.SPECIAL:                                                                                                                                 # codegen/uops.py:279
                                              elif u.op is UOps.LOAD:                                                                                                                                    # codegen/uops.py:281
                                              elif u.op is UOps.STORE:                                                                                                                                   # codegen/uops.py:284
                                              elif u.op is UOps.ALU and u not in dont_count:                                                                                                             # codegen/uops.py:287
                                              elif u.op is UOps.WMMA and u not in dont_count:                                                                                                            # codegen/uops.py:290
                                                mult_stack.append(mults)                                                                                                                                 # codegen/uops.py:275
                                                mults *= uop_alu_resolve(u.src[1] - u.src[0])                                                                                                            # codegen/uops.py:276

                                                  def uop_alu_resolve(u:UOp) -> sint:                                                                                                                    # codegen/uops.py:247
                                                    if u.op in {UOps.CONST, UOps.DEFINE_VAR}: return u.arg                                                                                               # codegen/uops.py:248
                                                    if u.op is UOps.ALU: return exec_alu(u.arg, cast(DType,u.dtype), tuple(map(uop_alu_resolve, u.src)))                                                 # codegen/uops.py:249

                                                    def exec_alu(op:Op, dtype:DType, operands): return truncate.get(dtype, lambda x: x)(python_alu[op](*operands))                                       # ops.py:134

                                              elif u.op is UOps.ENDRANGE:                                                                                                                          # OLD # codegen/uops.py:277
                                              elif u.op is UOps.SPECIAL:                                                                                                                           # OLD # codegen/uops.py:279
                                              elif u.op is UOps.LOAD:                                                                                                                              # OLD # codegen/uops.py:281
                                                assert u.dtype is not None                                                                                                                               # codegen/uops.py:282
                                                mem += u.dtype.itemsize * mults                                                                                                                          # codegen/uops.py:283
                                              elif u.op is UOps.STORE:                                                                                                                             # OLD # codegen/uops.py:284
                                              elif u.op is UOps.ALU and u not in dont_count:                                                                                                       # OLD # codegen/uops.py:287
                                                assert u.dtype is not None                                                                                                                               # codegen/uops.py:288
                                                flops += (mults * (2 if u.arg == TernaryOps.MULACC else 1)) * u.dtype.count                                                                              # codegen/uops.py:289
                                                assert u.src[2].dtype is not None                                                                                                                        # codegen/uops.py:285
                                                mem += u.src[2].dtype.itemsize * mults                                                                                                                   # codegen/uops.py:286
                                                mults = mult_stack.pop(-1)                                                                                                                               # codegen/uops.py:278
                                              elif u.op is UOps.SPECIAL:                                                                                                                           # OLD # codegen/uops.py:279
                                              elif u.op is UOps.LOAD:                                                                                                                              # OLD # codegen/uops.py:281
                                              elif u.op is UOps.STORE:                                                                                                                             # OLD # codegen/uops.py:284
                                              elif u.op is UOps.ALU and u not in dont_count:                                                                                                       # OLD # codegen/uops.py:287
                                              elif u.op is UOps.WMMA and u not in dont_count:                                                                                                      # OLD # codegen/uops.py:290
                                                assert u.dtype is not None                                                                                                                         # OLD # codegen/uops.py:282
                                                mem += u.dtype.itemsize * mults                                                                                                                    # OLD # codegen/uops.py:283
                                                assert u.dtype is not None                                                                                                                         # OLD # codegen/uops.py:288
                                                flops += (mults * (2 if u.arg == TernaryOps.MULACC else 1)) * u.dtype.count                                                                        # OLD # codegen/uops.py:289
                                                assert u.src[2].dtype is not None                                                                                                                  # OLD # codegen/uops.py:285
                                                mem += u.src[2].dtype.itemsize * mults                                                                                                             # OLD # codegen/uops.py:286
                                            return flops, mem                                                                                                                                            # codegen/uops.py:293

                                  @dataclass                                                                                                                                                             # renderer/__init__.py:18
                                  class Program:
                                    @property                                                                                                                                                            # renderer/__init__.py:56
                                    def lds_estimate(self) -> sint: return self._ops_lds[1]

                        return ret                                                                                                                                                                       # engine/realize.py:162

                    return ExecItem(runner, [si.bufs[x] for x in runner.p.globals], si.metadata)                                                                                                         # engine/realize.py:193

                class CompiledRunner(Runner):                                                                                                                                                            # engine/realize.py:79
                  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False) -> Optional[float]:                                                                                 # engine/realize.py:89
                    global_size, local_size = self.p.launch_dims(var_vals)                                                                                                                               # engine/realize.py:90

                      @dataclass                                                                                                                                                                         # renderer/__init__.py:18
                      class Program:
                        def launch_dims(self, var_vals:Dict[Variable, int]):                                                                                                                             # renderer/__init__.py:66
                          global_size = [sym_infer(sz, var_vals) for sz in self.global_size] if self.global_size is not None else None                                                                   # renderer/__init__.py:67
                          local_size = [sym_infer(sz, var_vals) for sz in self.local_size] if self.local_size is not None else None                                                                      # renderer/__init__.py:68
                          return global_size, local_size                                                                                                                                                 # renderer/__init__.py:69

                    if global_size is not None and local_size is None and all_int(self.p.global_size): # type: ignore[arg-type]                                                                          # engine/realize.py:91
                    lra = {}                                                                                                                                                                             # engine/realize.py:97
                    if global_size:                                                                                                                                                                      # engine/realize.py:98
                    if local_size:                                                                                                                                                                       # engine/realize.py:101
                    return self.clprg(*[x._buf for x in rawbufs], **lra, vals=tuple(var_vals[k] for k in self.p.vars), wait=wait)                                                                        # engine/realize.py:104

                      class ClangProgram:                                                                                                                                                                # runtime/ops_clang.py:14
                        def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)                                                           # runtime/ops_clang.py:23

                          def cpu_time_execution(cb, enable):                                                                                                                                            # helpers.py:283
                            if enable: st = time.perf_counter()                                                                                                                                          # helpers.py:284
                            cb()                                                                                                                                                                         # helpers.py:285

                            if enable: return time.perf_counter()-st                                                                                                                                     # helpers.py:286

              class Buffer:                                                                                                                                                                              # device.py:52
                def __del__(self):                                                                                                                                                                       # device.py:100
                  if not hasattr(self, '_buf'): return                                                                                                                                             # OLD # device.py:101
                  if self._base is None:                                                                                                                                                                 # device.py:102
                    if not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes                                                                                                        # device.py:103

                    self.allocator.free(self._buf, self.nbytes, self.options)                                                                                                                            # device.py:104

                      class LRUAllocator(Allocator):  # pylint: disable=abstract-method                                                                                                                  # device.py:143
                        def free(self, opaque:Any, size:int, options:Optional[BufferOptions]=None):                                                                                                      # device.py:159
                          if getenv("LRU", 1) and (options is None or not options.nolru): self.cache[(size, options)].append(opaque)                                                                     # device.py:160

            return self                                                                                                                                                                                  # tensor.py:205

          buf = cast(Buffer, cast(LazyBuffer, cpu.lazydata).base.realized)                                                                                                                               # tensor.py:246

          if self.device != "CLANG": buf.options = BufferOptions(nolru=True)                                                                                                                             # tensor.py:247

          return buf.as_buffer(allow_zero_copy=True if self.device != "CLANG" else False)                                                                                                                # tensor.py:248

        class _MallocAllocator(LRUAllocator):                                                                                                                                                            # device.py:163
          def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src, len(dest))                                                                                                         # device.py:167

