<p align="center">
  <img alt="MumPy" src="https://github.com/parsiad/mumpy/blob/master/logo.png?raw=true">
</p>

![](https://github.com/parsiad/mumpy/actions/workflows/tox.yml/badge.svg)
![](https://github.com/parsiad/mumpy/actions/workflows/api_coverage.yml/badge.svg)
<a href="https://github.com/parsiad/mumpy"><img alt="GitHub" src="https://img.shields.io/badge/github-%23121011.svg?logo=github"></a>

**MumPy = Metal + NumPy** in the same way that **CuPY = CUDA + NumPy**.
More precisely, MumPy is a NumPy interface for Apple Silicon (similar to how CuPy is a NumPy interface for CUDA).
It exists to

- enable devs familiar with NumPy to leverage their Apple Silicon hardware
- serve as a drop-in replacement for existing NumPy-based code

A mental model of the hierarchy is

```text
Apple Silicon (hardware)
├── Metal (GPU API) ──┐
└── MLX (array library)
    └── MumPy (NumPy-style interface)
```

For API coverage, see [`API_COVERAGE.md`](https://github.com/parsiad/mumpy/blob/master/API_COVERAGE.md).

🤖 A majority of this library is vibe coded.
It is tested and type-checked, but expect rough edges for now.

## Usage

```shell
pip install git+https://github.com/parsiad/mumpy.git
```

MumPy uses a familiar NumPy-style API:

```pycon
>>> import mumpy as np
>>> rng = np.random.default_rng(0)
>>> a = rng.normal(size=(4, 4))
>>> b = rng.normal(size=(4,))
>>> x = np.linalg.solve(a, b)
>>> y = x.mean()
```

It uses special wrappers around [MLX](https://github.com/ml-explore/mlx) arrays:

```pycon
>>> type(x).__name__
'MumPyArray'
>>> type(y).__name__
'MumPyScalar'
>>> y.item()
-0.9347759930825212
>>> x.dtype
mlx.core.float64
>>> raw = x.mx  # Raw MLX array
>>> type(raw).__name__
'array'
```

As seen above, default dtypes match NumPy (float64) to maximize compatibility.
You should generally opt into lower precision to take advantage of Apple Silicon hardware:

```pycon
>>> x32 = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
>>> x32.dtype
mlx.core.float32
```

When developing, it's helpful to see when MumPy had to hand work to NumPy (a "fallback" in MumPy terminology) instead of running natively in MLX.
Reducing fallbacks usually improves performance and makes execution more predictable.
You can do this with a fallback counter:

```pycon
>>> np.fallbacks.reset_counts()
>>> with np.fallbacks.capture_counts(reset=True) as cap:
...     _ = np.lexsort(([1, 1, 0], [3, 2, 1]))
>>> cap.delta
{'core.lexsort:numpy': 1}
```

Note that the above does not track cases where MumPy still uses MLX but runs on CPU.

## Design philosophy

As demonstrated above, MumPy aims to make NumPy the default mental model while using MLX as the execution backend.
That means:

- NumPy-style function names and signatures where practical.
- NumPy-style default dtypes for explicit MumPy APIs (for example, `int64` / `float64` on 64-bit platforms) unless a backend limitation forces a documented exception.
- MumPy-owned wrapper return types (`MumPyArray` / `MumPyScalar`) so both function-style and method-style calls can go through the same compatibility logic.

### Execution model (CPU vs GPU)

On Apple Silicon, CPU and GPU share unified memory, so crossing between CPU and GPU is usually cheaper than on discrete-GPU systems.
However, it is still not free: synchronization, new allocations, dtype conversions (for example `float64` casts), and memory bandwidth can all add measurable cost.

MumPy runs on top of MLX, which uses an operation-centric device model, not a PyTorch-style per-array `.to(...)` workflow.
In practice, what matters most is where an operation executes (CPU or GPU).
MumPy may route some operations to MLX CPU paths to preserve NumPy-like behavior (especially around default dtypes such as `float64`/`int64`).
This is different from a NumPy fallback: the operation still runs in MLX, just on CPU instead of GPU.

If you are optimizing for performance, prefer explicit `float32` / `int32` dtypes.

### Wrappers and the MLX escape hatch

MumPy-owned wrappers allow calls like `x.mean()` and `x.astype(...)` to use MumPy's compatibility rules (dtype parity, CPU routing where needed, and fallback behavior).
When you want direct MLX behavior (outside MumPy's compatibility layer), use the raw array escape hatch: `x.mx` gives you the underlying `mlx.core.array`.

### Compatibility layers

MumPy prefers MLX-native implementations on common paths, but it also provides broad NumPy-style coverage via compatibility fallbacks.

- Many less commonly used APIs in `mumpy`, `mumpy.linalg`, `mumpy.fft`, and `mumpy.random` are available through dynamic compatibility lookup.
- Many of those names are intentionally callable via attribute lookup but excluded from curated `__all__`, `dir()`, and `from ... import *` exports.
- Fallback results are converted back into MumPy/MLX when representable.
- If a fallback result cannot be represented in MLX (for example, some string/object arrays), MumPy returns a NumPy/Python object directly.

### Practical consequences

These are the main places where backend realities still matter:

- Default complex parity is a documented exception today: MumPy defaults to `complex64` because current MLX support does not provide `complex128` in the target environment.
- MLX arrays are immutable, so NumPy features that depend on writable views or in-place `out=` mutation cannot be mirrored exactly.
- `empty` / `empty_like` currently return zero-filled arrays because MLX does not expose uninitialized allocations.
- Some MLX linear algebra operations are CPU-only; MumPy routes those calls to CPU automatically.
- NumPy fallback wrapping may incur copies and host/device boundary costs.

### Deliberate fallback choices

Some behaviors are intentionally implemented via NumPy fallback today because MLX is not yet a good parity match:

- Non-`C` order paths (`F`, `A`, `K`) are routed through NumPy where needed.
- `lexsort` and `sort_complex` remain NumPy fallback implementations.
- `piecewise` is MLX-native for scalar/broadcastable callable outputs, while NumPy-style subset-callable semantics fall back to NumPy for parity.
- `mumpy.random.poisson` and `mumpy.random.binomial` are MLX-native, but large-parameter paths use fast approximation branches (still reproducible under `seed()` / `Generator`) to avoid very slow sampling loops.

## Development conventions

- Members private to a Python file have a leading underscore (for example, `_private_func`).
- Files not meant to be imported directly by users have a leading underscore (for example, `_impl.py`).
- User-visible members are only exported in `__init__.py` files (for example, `mumpy/fft/__init__.py` makes members in `mumpy/fft/_fft.py` visible).
