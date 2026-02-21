import builtins
import secrets
from contextlib import suppress
from functools import wraps
from typing import Any, cast

import mlx.core as mx

from .. import _bridge as bridge
from .. import _core as core
from .. import _dynamic_exports as dynamic_exports

_CPU_STREAM = getattr(mx, "cpu", None)
_FALLBACK_ATTR_CACHE: dict[str, Any] = {}
_NUMPY_EXPORT_EXCEPTIONS = (NotImplementedError, bridge.FallbackDisabledError)
_scalar_bool = core._scalar_bool
_prod_int = core._prod_int
_normalize_axis_index = core._normalize_axis_index


def _tolist_list(values: mx.array) -> list[Any]:
    return cast("list[Any]", values.tolist())


def _normalize_size(size: Any | None) -> tuple[int, ...] | None:
    if size is None:
        return None
    if isinstance(size, int):
        return (size,)
    if isinstance(size, tuple):
        return size
    if isinstance(size, list):
        return tuple(size)
    msg = "size must be an int, tuple, list, or None"
    raise TypeError(msg)


def _shape_for_mlx(size: Any | None) -> tuple[int, ...] | list[int]:
    shape = _normalize_size(size)
    if shape is None:
        return []
    return shape


def _sample_shape(size: Any | None, *params: Any) -> tuple[int, ...] | list[int]:
    shape = _normalize_size(size)
    if shape is not None:
        return shape
    shapes: list[tuple[int, ...]] = []
    for p in params:
        if p is None:
            continue
        arr = _asarray(p)
        if arr.ndim > 0:
            shapes.append(tuple(int(s) for s in arr.shape))
    if not shapes:
        return []
    return tuple(mx.broadcast_shapes(*shapes))


def _uniform_open01(size: Any | None = None, *params: Any, key: mx.array | None = None) -> mx.array:
    shape = _sample_shape(size, *params)
    u = mx.random.uniform(shape=shape) if key is None else mx.random.uniform(shape=shape, key=key)
    return mx.clip(u, 1e-7, 1.0 - 1e-7)


def _validate_positive(name: str, x: Any) -> None:
    if _scalar_bool(core.any(core.less_equal(_asarray(x), 0))):
        msg = f"{name} must be > 0"
        raise ValueError(msg)


def _validate_probability(name: str, p: Any) -> None:
    arr = _asarray(p)
    bad = core.logical_or(core.less_equal(arr, 0), core.greater(arr, 1))
    if _scalar_bool(core.any(bad)):
        msg = f"{name} must be in (0, 1]"
        raise ValueError(msg)


def _validate_probability_closed(name: str, p: Any) -> None:
    arr = _asarray(p)
    bad = core.logical_or(core.less(arr, 0), core.greater(arr, 1))
    if _scalar_bool(core.any(bad)):
        msg = f"{name} must be in [0, 1]"
        raise ValueError(msg)


def _shape_tuple(shape: Any | None) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, tuple):
        return tuple(int(s) for s in shape)
    if isinstance(shape, list):
        return tuple(int(s) for s in shape)
    msg = "shape must be a tuple, list, or None"
    raise TypeError(msg)


def _sample_shape_tuple(size: Any | None, *params: Any) -> tuple[int, ...]:
    return _shape_tuple(_sample_shape(size, *params))


def _rand_uniform(shape: tuple[int, ...], key: mx.array | None = None) -> mx.array:
    if key is None:
        return mx.random.uniform(shape=shape)
    return mx.random.uniform(shape=shape, key=key)


def _rand_normal(shape: tuple[int, ...], key: mx.array | None = None) -> mx.array:
    if key is None:
        return mx.random.normal(shape=shape)
    return mx.random.normal(shape=shape, key=key)


def _rand_int(low: int, high: int, shape: tuple[int, ...], key: mx.array | None = None) -> mx.array:
    if key is None:
        return mx.random.randint(low, high, shape=shape, dtype=mx.int32)
    return mx.random.randint(low, high, shape=shape, dtype=mx.int32, key=key)


def _normalize_prob_vector(p: Any, n: int) -> mx.array:
    p_arr = core.ravel(_asarray(p)).astype(mx.float32)
    if p_arr.ndim != 1 or int(p_arr.shape[0]) != int(n):
        msg = "a and p must have same size"
        raise ValueError(msg)
    if _scalar_bool(core.any(core.less(p_arr, 0))):
        msg = "probabilities are not non-negative"
        raise ValueError(msg)
    total = core.sum(p_arr)
    if _scalar_bool(core.less_equal(total, 0)):
        msg = "probabilities do not sum to a positive value"
        raise ValueError(msg)
    return p_arr / total


def _categorical_from_probs(p: mx.array, shape: tuple[int, ...], key: mx.array | None = None) -> mx.array:
    logits = mx.log(p)
    if key is None:
        return mx.random.categorical(logits, shape=shape)
    return mx.random.categorical(logits, shape=shape, key=key)


def _poisson_small_exact(lam_b: mx.array, mask: mx.array, key: mx.array | None = None) -> mx.array:
    shape = tuple(int(s) for s in lam_b.shape)
    counts = core.zeros(shape, dtype=mx.int32)
    if not _scalar_bool(core.any(mask)):
        return counts
    thresh = mx.exp(-lam_b)
    prod_u = core.ones(shape, dtype=mx.float32)
    active = mask
    key_work = key
    while _scalar_bool(core.any(active)):
        subkey = None
        if key_work is not None:
            key_work, subkey = _next_key_from(key_work)
        u = _uniform_open01(size=shape, key=subkey)
        prod_u = core.where(active, prod_u * u, prod_u)
        still = core.greater(prod_u, thresh)
        inc = core.logical_and(active, still)
        counts = counts + inc.astype(mx.int32)
        active = inc
    return counts


def _binomial_small_exact(n_i: mx.array, p_b: mx.array, mask: mx.array, key: mx.array | None = None) -> mx.array:
    shape = tuple(int(s) for s in n_i.shape)
    counts = core.zeros(shape, dtype=mx.int32)
    if not _scalar_bool(core.any(mask)):
        return counts
    max_n = int(core.max(core.where(mask, n_i, 0)).item())
    key_work = key
    for i in range(max_n):
        subkey = None
        if key_work is not None:
            key_work, subkey = _next_key_from(key_work)
        u = _uniform_open01(size=shape, key=subkey)
        active = core.logical_and(mask, core.greater(n_i, i))
        success = core.logical_and(active, core.less(u, p_b))
        counts = counts + success.astype(mx.int32)
    return counts


def _poisson_impl(lam: Any = 1.0, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    shape = _sample_shape_tuple(size, lam)
    if _prod_int(shape) == 0:
        return core.zeros(shape, dtype=mx.int32)
    lam_arr = _asarray(lam).astype(mx.float32)
    lam_b = mx.broadcast_to(lam_arr, shape)
    if _scalar_bool(core.any(core.less(lam_b, 0))):
        msg = "lam must be >= 0"
        raise ValueError(msg)
    counts = core.zeros(shape, dtype=mx.int32)
    lam_cur = lam_b
    active = core.greater(lam_cur, 0)
    key_work = key

    while _scalar_bool(core.any(core.logical_and(active, core.greater_equal(lam_cur, 10.0)))):
        large_mask = core.logical_and(active, core.greater_equal(lam_cur, 10.0))
        subkey_g = subkey_b = None
        if key_work is not None:
            key_work, subkey_g = _next_key_from(key_work)
            key_work, subkey_b = _next_key_from(key_work)
        m = mx.floor(lam_cur * 0.875).astype(mx.int32)
        m_safe = core.where(large_mask, m, 1)
        m_f = m_safe.astype(mx.float32)
        x = _gamma_impl(m_f, scale=1.0, size=None, key=subkey_g)
        take_m = core.logical_and(large_mask, core.less(x, lam_cur))
        counts = counts + core.where(take_m, m, 0)
        lam_cur = core.where(take_m, lam_cur - x, lam_cur)

        stop_mask = core.logical_and(large_mask, core.logical_not(take_m))
        if _scalar_bool(core.any(stop_mask)):
            p_bin = core.where(stop_mask, lam_cur / x, 0.5)
            n_bin = core.where(stop_mask, m_safe - 1, 0)
            tail = _binomial_impl(n_bin, p_bin, size=None, key=subkey_b)
            counts = counts + core.where(stop_mask, tail, 0)
            active = core.logical_and(active, core.logical_not(stop_mask))

    if _scalar_bool(core.any(active)):
        subkey_small = None
        if key_work is not None:
            key_work, subkey_small = _next_key_from(key_work)
        small_counts = _poisson_small_exact(lam_cur, active, key=subkey_small)
        counts = counts + core.where(active, small_counts, 0)
    return counts


def _binomial_impl(n: Any, p: Any, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    shape = _sample_shape_tuple(size, n, p)
    if _prod_int(shape) == 0:
        return core.zeros(shape, dtype=mx.int32)
    n_arr = _asarray(n).astype(mx.float32)
    p_arr = _asarray(p).astype(mx.float32)
    n_b = mx.broadcast_to(n_arr, shape)
    p_b = mx.broadcast_to(p_arr, shape)
    _validate_probability_closed("p", p_b)
    if _scalar_bool(core.any(core.less(n_b, 0))):
        msg = "n must be >= 0"
        raise ValueError(msg)
    if _scalar_bool(core.any(core.not_equal(n_b, mx.floor(n_b)))):
        msg = "n must be an integer"
        raise ValueError(msg)
    n_i = n_b.astype(mx.int32)
    flip = core.greater(p_b, 0.5)
    p_use = core.where(flip, 1.0 - p_b, p_b)

    counts = core.zeros(shape, dtype=mx.int32)
    k_acc = core.zeros(shape, dtype=mx.int32)
    n_cur = n_i
    p_cur = p_use.astype(mx.float32)
    active = core.greater(n_cur, 0)
    key_work = key

    small_n = 64
    while _scalar_bool(core.any(active)):
        small_mask = core.logical_and(active, core.less_equal(n_cur, small_n))
        if _scalar_bool(core.any(small_mask)):
            subkey_small = None
            if key_work is not None:
                key_work, subkey_small = _next_key_from(key_work)
            small_counts = _binomial_small_exact(n_cur, p_cur, small_mask, key=subkey_small)
            k_acc = k_acc + core.where(small_mask, small_counts, 0)
            active = core.logical_and(active, core.logical_not(small_mask))

        if not _scalar_bool(core.any(active)):
            break

        large_mask = active
        subkey_beta = None
        if key_work is not None:
            key_work, subkey_beta = _next_key_from(key_work)

        a = (n_cur // 2) + 1
        b = n_cur - a + 1
        a_safe = core.where(large_mask, a, 1)
        b_safe = core.where(large_mask, b, 1)
        x = _beta_impl(a_safe.astype(mx.float32), b_safe.astype(mx.float32), size=None, key=subkey_beta)
        go_left = core.logical_and(large_mask, core.greater_equal(x, p_cur))
        go_right = core.logical_and(large_mask, core.logical_not(go_left))

        # Left branch: keep lower half, no offset.
        n_cur = core.where(go_left, a - 1, n_cur)
        p_cur = core.where(go_left, p_cur / x, p_cur)

        # Right branch: skip the guaranteed successes in the lower half.
        k_acc = k_acc + core.where(go_right, a, 0)
        n_cur = core.where(go_right, b - 1, n_cur)
        p_cur = core.where(go_right, (p_cur - x) / (1.0 - x), p_cur)

        p_cur = core.where(core.less(p_cur, 0.0), 0.0, p_cur)
        p_cur = core.where(core.greater(p_cur, 1.0), 1.0, p_cur)

    counts = k_acc
    return core.where(flip, n_i - counts, counts)


def _standard_gamma_sample(alpha: mx.array, key: mx.array | None = None) -> mx.array:
    a = _asarray(alpha).astype(mx.float32)
    _validate_positive("shape", a)
    shape = tuple(int(s) for s in a.shape)
    if _prod_int(shape) == 0:
        return core.zeros(shape, dtype=mx.float32)
    key_work = key
    subkey = None
    if key_work is not None:
        key_work, subkey = _next_key_from(key_work)
    u_adj = _uniform_open01(size=shape, key=subkey)
    lt1 = core.less(a, 1.0)
    a_eff = core.where(lt1, a + 1.0, a)
    d = a_eff - (1.0 / 3.0)
    c = 1.0 / mx.sqrt(9.0 * d)
    out = core.zeros(shape, dtype=mx.float32)
    done = core.zeros(shape, dtype=mx.bool_)
    while not _scalar_bool(core.all(done)):
        kx = ku = None
        if key_work is not None:
            key_work, kx = _next_key_from(key_work)
            key_work, ku = _next_key_from(key_work)
        x = _rand_normal(shape, key=kx)
        v = 1.0 + c * x
        v3 = v * v * v
        good_v = core.greater(v, 0)
        v3_safe = core.where(good_v, v3, 1.0)
        u = _uniform_open01(size=shape, key=ku)
        x2 = x * x
        cond1 = core.less(u, 1.0 - 0.0331 * x2 * x2)
        cond2 = core.less(mx.log(u), 0.5 * x2 + d * (1.0 - v3_safe + mx.log(v3_safe)))
        accept = core.logical_and(good_v, core.logical_or(cond1, cond2))
        accept = core.logical_and(core.logical_not(done), accept)
        out = core.where(accept, d * v3_safe, out)
        done = core.logical_or(done, accept)
    factor = core.where(lt1, mx.power(u_adj, 1.0 / a), 1.0)
    return out * factor


def _standard_gamma_impl(shape_param: Any, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    shape = _sample_shape_tuple(size, shape_param)
    alpha = _asarray(shape_param).astype(mx.float32)
    alpha_b = mx.broadcast_to(alpha, shape)
    return _standard_gamma_sample(alpha_b, key=key)


def _gamma_impl(shape_param: Any, scale: Any = 1.0, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    shape = _sample_shape_tuple(size, shape_param, scale)
    alpha = mx.broadcast_to(_asarray(shape_param).astype(mx.float32), shape)
    scale_b = mx.broadcast_to(_asarray(scale).astype(mx.float32), shape)
    _validate_positive("shape", alpha)
    _validate_positive("scale", scale_b)
    return _standard_gamma_sample(alpha, key=key) * scale_b


def _chisquare_impl(df: Any, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    df_arr = _asarray(df).astype(mx.float32)
    _validate_positive("df", df_arr)
    return 2.0 * _gamma_impl(df_arr * 0.5, scale=1.0, size=size, key=key)


def _beta_impl(a: Any, b: Any, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    shape = _sample_shape_tuple(size, a, b)
    aa = mx.broadcast_to(_asarray(a).astype(mx.float32), shape)
    bb = mx.broadcast_to(_asarray(b).astype(mx.float32), shape)
    _validate_positive("a", aa)
    _validate_positive("b", bb)
    k1 = k2 = None
    if key is not None:
        key_mid, k1 = _next_key_from(key)
        _, k2 = _next_key_from(key_mid)
    ga = _standard_gamma_sample(aa, key=k1)
    gb = _standard_gamma_sample(bb, key=k2)
    return ga / (ga + gb)


def _dirichlet_impl(alpha: Any, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    alpha_vec = core.ravel(_asarray(alpha)).astype(mx.float32)
    if alpha_vec.ndim != 1:
        msg = "alpha must be 1-dimensional"
        raise ValueError(msg)
    if int(alpha_vec.shape[0]) == 0:
        msg = "alpha must be non-empty"
        raise ValueError(msg)
    _validate_positive("alpha", alpha_vec)
    size_shape = _normalize_size(size)
    prefix = () if size_shape is None else tuple(int(s) for s in size_shape)
    final_shape = (*prefix, int(alpha_vec.shape[0]))
    if _prod_int(prefix) == 0:
        return core.zeros(final_shape, dtype=mx.float32)
    alpha_b = mx.broadcast_to(alpha_vec, final_shape)
    g = _standard_gamma_sample(alpha_b, key=key)
    return g / core.sum(g, axis=-1, keepdims=True)


def _standard_t_impl(df: Any, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    shape = _sample_shape_tuple(size, df)
    if _prod_int(shape) == 0:
        return core.zeros(shape, dtype=mx.float32)
    df_b = mx.broadcast_to(_asarray(df).astype(mx.float32), shape)
    _validate_positive("df", df_b)
    kz = kc = None
    if key is not None:
        key_mid, kz = _next_key_from(key)
        _, kc = _next_key_from(key_mid)
    z = _rand_normal(shape, key=kz)
    chi2 = _chisquare_impl(df_b, size=None, key=kc)
    return z / mx.sqrt(chi2 / df_b)


def _f_impl(dfnum: Any, dfden: Any, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    shape = _sample_shape_tuple(size, dfnum, dfden)
    if _prod_int(shape) == 0:
        return core.zeros(shape, dtype=mx.float32)
    dfn = mx.broadcast_to(_asarray(dfnum).astype(mx.float32), shape)
    dfd = mx.broadcast_to(_asarray(dfden).astype(mx.float32), shape)
    _validate_positive("dfnum", dfn)
    _validate_positive("dfden", dfd)
    k1 = k2 = None
    if key is not None:
        key_mid, k1 = _next_key_from(key)
        _, k2 = _next_key_from(key_mid)
    x1 = _chisquare_impl(dfn, size=None, key=k1) / dfn
    x2 = _chisquare_impl(dfd, size=None, key=k2) / dfd
    return x1 / x2


def _multinomial_impl(n: int, pvals: Any, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    n_int = int(n)
    if n_int < 0:
        msg = "n must be >= 0"
        raise ValueError(msg)
    p_flat = core.ravel(_asarray(pvals))
    if p_flat.ndim != 1 or int(p_flat.shape[0]) == 0:
        msg = "pvals must be a non-empty 1-D sequence"
        raise ValueError(msg)
    probs = _normalize_prob_vector(p_flat, int(p_flat.shape[0]))
    k = int(probs.shape[0])
    size_shape = _normalize_size(size)
    prefix = () if size_shape is None else tuple(int(s) for s in size_shape)
    sample_count = _prod_int(prefix) if prefix else 1
    if sample_count == 0:
        counts = core.zeros((0, k), dtype=mx.int32)
    elif n_int == 0:
        counts = core.zeros((sample_count, k), dtype=mx.int32)
    else:
        draws = _categorical_from_probs(probs, (sample_count, n_int), key=key).astype(mx.int32)
        cols: list[mx.array] = [core.sum(core.equal(draws, cls).astype(mx.int32), axis=1) for cls in range(k)]
        counts = core.stack(cols, axis=1)
    if size_shape is None:
        return counts[0]
    return core.reshape(counts, (*prefix, k))


def _negative_binomial_impl(n: Any, p: Any, size: Any | None = None, key: mx.array | None = None) -> mx.array:
    shape = _sample_shape_tuple(size, n, p)
    if _prod_int(shape) == 0:
        return core.zeros(shape, dtype=mx.int32)
    n_b = mx.broadcast_to(_asarray(n).astype(mx.float32), shape)
    p_b = mx.broadcast_to(_asarray(p).astype(mx.float32), shape)
    _validate_positive("n", n_b)
    _validate_probability("p", p_b)
    kg = kp = None
    if key is not None:
        key_mid, kg = _next_key_from(key)
        _, kp = _next_key_from(key_mid)
    lam = _standard_gamma_sample(n_b, key=kg) * ((1.0 - p_b) / p_b)
    return _poisson_impl(lam=lam, size=None, key=kp)


def _permuted_impl(x: Any, axis: int = 0, key: mx.array | None = None) -> mx.array:
    arr = _asarray(x)
    ax = _normalize_axis_index(axis, arr.ndim)
    shape = tuple(int(s) for s in arr.shape)
    noise = _rand_uniform(shape, key=key)
    idx = core.argsort(noise, axis=ax).astype(mx.int32)
    return core.take_along_axis(arr, idx, axis=ax)


def _asarray(a: Any, dtype: Any | None = None) -> mx.array:
    return core._asarray(a, dtype_=dtype)


def _coerce_public_float_output(arr: mx.array) -> mx.array:
    return core._coerce_public_default_float_output(arr)


def _coerce_public_int_output(arr: mx.array) -> mx.array:
    return core._coerce_public_default_int_output(arr)


def _coerce_public_float_output_for_dtype(arr: mx.array, dtype: Any | None) -> mx.array:
    resolved = core.dtype(dtype)
    if resolved is None:
        return _coerce_public_float_output(arr)
    return arr if arr.dtype is resolved else core._coerce_public_output_dtype(arr, resolved)


def _coerce_public_int_output_for_dtype(arr: mx.array, dtype: Any | None) -> mx.array:
    resolved = core.dtype(dtype)
    if resolved is None:
        return _coerce_public_int_output(arr)
    return arr if arr.dtype is resolved else core._coerce_public_output_dtype(arr, resolved)


def _mx_random_float_dtype_arg(dtype: Any | None) -> Any:
    return core.dtype(dtype) or mx.float32


def _mx_random_int_dtype_arg(dtype: Any | None) -> Any:
    return core.dtype(dtype) or mx.int32


def _next_key_from(key: mx.array) -> tuple[mx.array, mx.array]:
    keys = mx.random.split(key, 2)
    return keys[0], keys[1]


def _seed_from_key_array(key_arr: mx.array) -> int:
    onp = bridge.numpy_module()
    vals = onp.asarray(key_arr, dtype=onp.uint64).reshape(-1)
    seed = 0x9E3779B97F4A7C15
    for val in vals:
        seed ^= int(val)
        seed = (seed * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    return seed


def _global_numpy_rng():
    hi = int(mx.random.randint(0, 2**31 - 1, shape=[]).item())
    lo = int(mx.random.randint(0, 2**31 - 1, shape=[]).item())
    seed = ((hi & 0xFFFFFFFF) << 32) ^ (lo & 0xFFFFFFFF)
    return bridge.numpy_module().random.default_rng(seed)


def _numpy_rng_from_key(key_arr: mx.array):
    return bridge.numpy_module().random.default_rng(_seed_from_key_array(key_arr))


def seed(seed: int) -> None:
    mx.random.seed(seed)
    with suppress(bridge.FallbackDisabledError):
        bridge.numpy_module().random.seed(seed)


def key(seed: int) -> mx.array:
    return mx.random.key(seed)


def split(prng_key: Any, num: int = 2) -> mx.array:
    return mx.random.split(_asarray(prng_key), num=num)


state = mx.random.state


def random(size: Any | None = None) -> mx.array:
    return mx.random.uniform(shape=_shape_for_mlx(size))


def random_sample(size: Any | None = None) -> mx.array:
    return random(size=size)


def sample(size: Any | None = None) -> mx.array:
    return random(size=size)


def ranf(size: Any | None = None) -> mx.array:
    return random(size=size)


def rand(*dims: int) -> mx.array:
    return mx.random.uniform(shape=dims or [])


def randn(*dims: int) -> mx.array:
    return mx.random.normal(shape=dims or [])


def uniform(low: Any = 0.0, high: Any = 1.0, size: Any | None = None) -> mx.array:
    return mx.random.uniform(low=low, high=high, shape=_shape_for_mlx(size))


def normal(loc: Any = 0.0, scale: Any = 1.0, size: Any | None = None) -> mx.array:
    return mx.random.normal(shape=_shape_for_mlx(size), loc=loc, scale=scale)


def standard_normal(size: Any | None = None) -> mx.array:
    return mx.random.normal(shape=_shape_for_mlx(size))


def randint(low: Any, high: Any | None = None, size: Any | None = None, dtype: Any | None = None) -> mx.array:
    if high is None:
        high_ = low
        low_ = 0
    else:
        low_ = low
        high_ = high
    out = mx.random.randint(low_, high_, shape=_shape_for_mlx(size), dtype=_mx_random_int_dtype_arg(dtype))
    return _coerce_public_int_output_for_dtype(out, dtype)


def integers(
    low: Any,
    high: Any | None = None,
    size: Any | None = None,
    dtype: Any | None = None,
    endpoint: bool = False,
) -> mx.array:
    if high is None:
        high_ = low
        low_ = 0
    else:
        low_ = low
        high_ = high
    if endpoint:
        high_ = high_ + 1
    return randint(low=low_, high=high_, size=size, dtype=dtype)


def bernoulli(p: Any = 0.5, size: Any | None = None) -> mx.array:
    return mx.random.bernoulli(p=p, shape=_normalize_size(size))


def laplace(loc: float = 0.0, scale: float = 1.0, size: Any | None = None) -> mx.array:
    return mx.random.laplace(shape=_shape_for_mlx(size), loc=loc, scale=scale)


def gumbel(loc: float = 0.0, scale: float = 1.0, size: Any | None = None) -> mx.array:
    _validate_positive("scale", scale)
    return mx.random.gumbel(shape=_sample_shape(size, loc, scale)) * scale + loc


def exponential(scale: float = 1.0, size: Any | None = None) -> mx.array:
    _validate_positive("scale", scale)
    u = _uniform_open01(size, scale)
    return -mx.log1p(-u) * scale


def standard_exponential(size: Any | None = None) -> mx.array:
    return exponential(scale=1.0, size=size)


def rayleigh(scale: float = 1.0, size: Any | None = None) -> mx.array:
    _validate_positive("scale", scale)
    u = _uniform_open01(size, scale)
    return scale * mx.sqrt(-2.0 * mx.log1p(-u))


def logistic(loc: float = 0.0, scale: float = 1.0, size: Any | None = None) -> mx.array:
    _validate_positive("scale", scale)
    u = _uniform_open01(size, loc, scale)
    return loc + scale * mx.log(u / (1.0 - u))


def lognormal(mean: float = 0.0, sigma: float = 1.0, size: Any | None = None) -> mx.array:
    _validate_positive("sigma", sigma)
    base = mx.random.normal(shape=_shape_for_mlx(size), loc=mean, scale=sigma)
    return _coerce_public_float_output(mx.exp(base))


def standard_cauchy(size: Any | None = None) -> mx.array:
    u = _uniform_open01(size)
    return mx.tan(mx.pi * (u - 0.5))


def triangular(left: float = 0.0, mode: float = 0.5, right: float = 1.0, size: Any | None = None) -> mx.array:
    left_arr = _asarray(left).astype(mx.float32)
    mode_arr = _asarray(mode).astype(mx.float32)
    right_arr = _asarray(right).astype(mx.float32)
    if _scalar_bool(core.any(core.greater(left_arr, mode_arr))) or _scalar_bool(
        core.any(core.greater(mode_arr, right_arr)),
    ):
        msg = "left <= mode <= right must hold"
        raise ValueError(msg)
    width = right_arr - left_arr
    u = _uniform_open01(size, left_arr, mode_arr, right_arr)
    c = core.where(core.equal(width, 0), 0.0, (mode_arr - left_arr) / width)
    lower = left_arr + mx.sqrt(u * width * (mode_arr - left_arr))
    upper = right_arr - mx.sqrt((1.0 - u) * width * (right_arr - mode_arr))
    out = core.where(core.less(u, c), lower, upper)
    return core.where(core.equal(width, 0), left_arr, out)


def weibull(a: Any, size: Any | None = None) -> mx.array:
    _validate_positive("a", a)
    u = _uniform_open01(size, a)
    return mx.power(-mx.log1p(-u), 1.0 / _asarray(a))


def power(a: Any, size: Any | None = None) -> mx.array:
    _validate_positive("a", a)
    u = _uniform_open01(size, a)
    return mx.power(u, 1.0 / _asarray(a))


def pareto(a: Any, size: Any | None = None) -> mx.array:
    _validate_positive("a", a)
    u = _uniform_open01(size, a)
    return mx.power(1.0 - u, -1.0 / _asarray(a)) - 1.0


def geometric(p: Any, size: Any | None = None) -> mx.array:
    _validate_probability("p", p)
    p_arr = _asarray(p).astype(mx.float32)
    u = _uniform_open01(size, p_arr)
    out = (mx.floor(mx.log1p(-u) / mx.log1p(-p_arr)) + 1).astype(mx.int32)
    return _coerce_public_int_output(out)


def bytes(length: int) -> builtins.bytes:
    n = int(length)
    if n < 0:
        msg = "length must be non-negative"
        raise ValueError(msg)
    vals = randint(0, 256, size=n, dtype=mx.uint8)
    return builtins.bytes(_tolist_list(vals))


def standard_gamma(shape: Any, size: Any | None = None) -> mx.array:
    return _standard_gamma_impl(shape, size=size)


def gamma(shape: Any, scale: Any = 1.0, size: Any | None = None) -> mx.array:
    return _gamma_impl(shape, scale=scale, size=size)


def chisquare(df: Any, size: Any | None = None) -> mx.array:
    return _chisquare_impl(df, size=size)


def f(dfnum: Any, dfden: Any, size: Any | None = None) -> mx.array:
    return _f_impl(dfnum, dfden, size=size)


def beta(a: Any, b: Any, size: Any | None = None) -> mx.array:
    return _beta_impl(a, b, size=size)


def dirichlet(alpha: Any, size: Any | None = None) -> mx.array:
    return _dirichlet_impl(alpha, size=size)


def standard_t(df: Any, size: Any | None = None) -> mx.array:
    return _standard_t_impl(df, size=size)


def multinomial(n: int, pvals: Any, size: Any | None = None) -> mx.array:
    return _multinomial_impl(n, pvals, size=size)


def negative_binomial(n: Any, p: Any, size: Any | None = None) -> mx.array:
    return _negative_binomial_impl(n, p, size=size)


def poisson(lam: Any = 1.0, size: Any | None = None) -> mx.array:
    return _poisson_impl(lam=lam, size=size)


def binomial(n: Any, p: Any, size: Any | None = None) -> mx.array:
    return _binomial_impl(n=n, p=p, size=size)


def truncated_normal(
    lower: Any,
    upper: Any,
    size: Any | None = None,
    dtype: Any | None = None,
) -> mx.array:
    out = mx.random.truncated_normal(
        lower,
        upper,
        shape=_normalize_size(size),
        dtype=_mx_random_float_dtype_arg(dtype),
    )
    return _coerce_public_float_output_for_dtype(out, dtype)


def multivariate_normal(mean: Any, cov: Any, size: Any | None = None, dtype: Any | None = None) -> mx.array:
    out = mx.random.multivariate_normal(
        _asarray(mean),
        _asarray(cov),
        shape=_shape_for_mlx(size),
        dtype=_mx_random_float_dtype_arg(dtype),
        stream=_CPU_STREAM,
    )
    return _coerce_public_float_output_for_dtype(out, dtype)


def permutation(x: Any, axis: int = 0) -> mx.array:
    if isinstance(x, int):
        return _coerce_public_int_output(mx.random.permutation(x))
    return mx.random.permutation(_asarray(x), axis=axis)


def shuffle(x: Any, axis: int = 0) -> mx.array:
    return permutation(x, axis=axis)


def permuted(x: Any, axis: int = 0) -> mx.array:
    return _permuted_impl(x, axis=axis)


def _choice_impl(
    a: Any,
    size: Any | None = None,
    replace: bool = True,
    p: Any | None = None,
    axis: int = 0,
    key_: mx.array | None = None,
) -> mx.array:
    size_shape = _normalize_size(size)
    out_shape = () if size_shape is None else tuple(int(s) for s in size_shape)
    k = 1 if size_shape is None else _prod_int(out_shape)

    def _draw_indices(pop_size: int, probs: mx.array | None) -> mx.array:
        if not replace and k > pop_size:
            msg = "Cannot take a larger sample than population when replace=False"
            raise ValueError(msg)
        if replace:
            if probs is None:
                return _rand_int(0, pop_size, out_shape, key=key_)
            return _categorical_from_probs(probs, out_shape, key=key_).astype(mx.int32)
        if k == 0:
            return core.zeros(out_shape, dtype=mx.int32)
        if probs is None:
            perm = mx.random.permutation(pop_size) if key_ is None else mx.random.permutation(pop_size, key=key_)
            flat_idx = perm[:k].astype(mx.int32)
        else:
            nz = core.sum(core.greater(probs, 0).astype(mx.int32))
            if _scalar_bool(core.less(nz, k)):
                msg = "Fewer non-zero entries in p than size"
                raise ValueError(msg)
            # Exact weighted-without-replacement sampling via Gumbel-top-k (Plackett-Luce).
            g = mx.random.gumbel(shape=(pop_size,)) if key_ is None else mx.random.gumbel(shape=(pop_size,), key=key_)
            scores = mx.log(probs) + g
            order = core.argsort(scores, axis=0).astype(mx.int32)
            flat_idx = order[-k:][::-1]
        if size_shape is None:
            return flat_idx[0]
        return core.reshape(flat_idx, out_shape)

    if isinstance(a, int):
        if a <= 0:
            msg = "a must be a positive integer"
            raise ValueError(msg)
        probs = None if p is None else _normalize_prob_vector(p, int(a))
        return _draw_indices(int(a), probs)

    pop = _asarray(a)
    if pop.ndim == 0:
        msg = "a must be 1-dimensional or an integer"
        raise ValueError(msg)
    ax = _normalize_axis_index(axis, pop.ndim)
    pop_size = int(pop.shape[ax])
    probs = None if p is None else _normalize_prob_vector(p, pop_size)
    idx = _draw_indices(pop_size, probs)
    return core.take(pop, idx, axis=ax)


def choice(
    a: Any,
    size: Any | None = None,
    replace: bool = True,
    p: Any | None = None,
    axis: int = 0,
) -> mx.array:
    out = _choice_impl(a, size=size, replace=replace, p=p, axis=axis)
    if isinstance(a, int):
        return _coerce_public_int_output(out)
    return out


class Generator:
    def __init__(self, seed_or_key: int | mx.array | None = None) -> None:
        if seed_or_key is None:
            seed_or_key = secrets.randbits(32)
        if isinstance(seed_or_key, int):
            self._key = mx.random.key(seed_or_key)
        else:
            self._key = _asarray(seed_or_key)

    @property
    def key(self) -> mx.array:
        return self._key

    @property
    def bit_generator(self) -> Any:
        return _numpy_rng_from_key(self._key).bit_generator

    def _next_key(self) -> mx.array:
        self._key, subkey = _next_key_from(self._key)
        return subkey

    def random(self, size: Any | None = None) -> mx.array:
        return mx.random.uniform(shape=_shape_for_mlx(size), key=self._next_key())

    def random_sample(self, size: Any | None = None) -> mx.array:
        return self.random(size=size)

    def uniform(self, low: Any = 0.0, high: Any = 1.0, size: Any | None = None) -> mx.array:
        return mx.random.uniform(low=low, high=high, shape=_shape_for_mlx(size), key=self._next_key())

    def normal(self, loc: Any = 0.0, scale: Any = 1.0, size: Any | None = None) -> mx.array:
        return mx.random.normal(shape=_shape_for_mlx(size), loc=loc, scale=scale, key=self._next_key())

    def standard_normal(self, size: Any | None = None) -> mx.array:
        return mx.random.normal(shape=_shape_for_mlx(size), key=self._next_key())

    def integers(
        self,
        low: Any,
        high: Any | None = None,
        size: Any | None = None,
        dtype: Any | None = None,
        endpoint: bool = False,
    ) -> mx.array:
        if high is None:
            high_ = low
            low_ = 0
        else:
            low_ = low
            high_ = high
        if endpoint:
            high_ = high_ + 1
        out = mx.random.randint(
            low_,
            high_,
            shape=_shape_for_mlx(size),
            dtype=_mx_random_int_dtype_arg(dtype),
            key=self._next_key(),
        )
        return _coerce_public_int_output_for_dtype(out, dtype)

    def choice(
        self,
        a: Any,
        size: Any | None = None,
        replace: bool = True,
        p: Any | None = None,
        axis: int = 0,
    ) -> mx.array:
        out = _choice_impl(a, size=size, replace=replace, p=p, axis=axis, key_=self._next_key())
        if isinstance(a, int):
            return _coerce_public_int_output(out)
        return out

    def permutation(self, x: Any, axis: int = 0) -> mx.array:
        if isinstance(x, int):
            return _coerce_public_int_output(mx.random.permutation(x, key=self._next_key()))
        return mx.random.permutation(_asarray(x), axis=axis, key=self._next_key())

    def shuffle(self, x: Any, axis: int = 0) -> mx.array:
        return self.permutation(x, axis=axis)

    def permuted(self, x: Any, axis: int = 0) -> mx.array:
        return _permuted_impl(x, axis=axis, key=self._next_key())

    def bernoulli(self, p: Any = 0.5, size: Any | None = None) -> mx.array:
        return mx.random.bernoulli(p=p, shape=_normalize_size(size), key=self._next_key())

    def laplace(self, loc: float = 0.0, scale: float = 1.0, size: Any | None = None) -> mx.array:
        return mx.random.laplace(shape=_shape_for_mlx(size), loc=loc, scale=scale, key=self._next_key())

    def gumbel(self, loc: float = 0.0, scale: float = 1.0, size: Any | None = None) -> mx.array:
        _validate_positive("scale", scale)
        return mx.random.gumbel(shape=_sample_shape(size, loc, scale), key=self._next_key()) * scale + loc

    def exponential(self, scale: float = 1.0, size: Any | None = None) -> mx.array:
        _validate_positive("scale", scale)
        u = _uniform_open01(size, scale, key=self._next_key())
        return -mx.log1p(-u) * scale

    def standard_exponential(self, size: Any | None = None) -> mx.array:
        return self.exponential(scale=1.0, size=size)

    def rayleigh(self, scale: float = 1.0, size: Any | None = None) -> mx.array:
        _validate_positive("scale", scale)
        u = _uniform_open01(size, scale, key=self._next_key())
        return scale * mx.sqrt(-2.0 * mx.log1p(-u))

    def logistic(self, loc: float = 0.0, scale: float = 1.0, size: Any | None = None) -> mx.array:
        _validate_positive("scale", scale)
        u = _uniform_open01(size, loc, scale, key=self._next_key())
        return loc + scale * mx.log(u / (1.0 - u))

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, size: Any | None = None) -> mx.array:
        _validate_positive("sigma", sigma)
        base = mx.random.normal(shape=_shape_for_mlx(size), loc=mean, scale=sigma, key=self._next_key())
        return _coerce_public_float_output(mx.exp(base))

    def standard_cauchy(self, size: Any | None = None) -> mx.array:
        u = _uniform_open01(size, key=self._next_key())
        return mx.tan(mx.pi * (u - 0.5))

    def triangular(self, left: float = 0.0, mode: float = 0.5, right: float = 1.0, size: Any | None = None) -> mx.array:
        left_arr = _asarray(left).astype(mx.float32)
        mode_arr = _asarray(mode).astype(mx.float32)
        right_arr = _asarray(right).astype(mx.float32)
        if _scalar_bool(core.any(core.greater(left_arr, mode_arr))) or _scalar_bool(
            core.any(core.greater(mode_arr, right_arr)),
        ):
            msg = "left <= mode <= right must hold"
            raise ValueError(msg)
        width = right_arr - left_arr
        u = _uniform_open01(size, left_arr, mode_arr, right_arr, key=self._next_key())
        c = core.where(core.equal(width, 0), 0.0, (mode_arr - left_arr) / width)
        lower = left_arr + mx.sqrt(u * width * (mode_arr - left_arr))
        upper = right_arr - mx.sqrt((1.0 - u) * width * (right_arr - mode_arr))
        out = core.where(core.less(u, c), lower, upper)
        return core.where(core.equal(width, 0), left_arr, out)

    def weibull(self, a: Any, size: Any | None = None) -> mx.array:
        _validate_positive("a", a)
        u = _uniform_open01(size, a, key=self._next_key())
        return mx.power(-mx.log1p(-u), 1.0 / _asarray(a))

    def power(self, a: Any, size: Any | None = None) -> mx.array:
        _validate_positive("a", a)
        u = _uniform_open01(size, a, key=self._next_key())
        return mx.power(u, 1.0 / _asarray(a))

    def pareto(self, a: Any, size: Any | None = None) -> mx.array:
        _validate_positive("a", a)
        u = _uniform_open01(size, a, key=self._next_key())
        return mx.power(1.0 - u, -1.0 / _asarray(a)) - 1.0

    def geometric(self, p: Any, size: Any | None = None) -> mx.array:
        _validate_probability("p", p)
        p_arr = _asarray(p).astype(mx.float32)
        u = _uniform_open01(size, p_arr, key=self._next_key())
        out = (mx.floor(mx.log1p(-u) / mx.log1p(-p_arr)) + 1).astype(mx.int32)
        return _coerce_public_int_output(out)

    def standard_gamma(self, shape: Any, size: Any | None = None) -> mx.array:
        return _standard_gamma_impl(shape, size=size, key=self._next_key())

    def gamma(self, shape: Any, scale: Any = 1.0, size: Any | None = None) -> mx.array:
        return _gamma_impl(shape, scale=scale, size=size, key=self._next_key())

    def chisquare(self, df: Any, size: Any | None = None) -> mx.array:
        return _chisquare_impl(df, size=size, key=self._next_key())

    def f(self, dfnum: Any, dfden: Any, size: Any | None = None) -> mx.array:
        return _f_impl(dfnum, dfden, size=size, key=self._next_key())

    def beta(self, a: Any, b: Any, size: Any | None = None) -> mx.array:
        return _beta_impl(a, b, size=size, key=self._next_key())

    def dirichlet(self, alpha: Any, size: Any | None = None) -> mx.array:
        return _dirichlet_impl(alpha, size=size, key=self._next_key())

    def standard_t(self, df: Any, size: Any | None = None) -> mx.array:
        return _standard_t_impl(df, size=size, key=self._next_key())

    def multinomial(self, n: int, pvals: Any, size: Any | None = None) -> mx.array:
        return _multinomial_impl(n, pvals, size=size, key=self._next_key())

    def negative_binomial(self, n: Any, p: Any, size: Any | None = None) -> mx.array:
        return _negative_binomial_impl(n, p, size=size, key=self._next_key())

    def bytes(self, length: int) -> builtins.bytes:
        n = int(length)
        if n < 0:
            msg = "length must be non-negative"
            raise ValueError(msg)
        vals = self.integers(0, 256, size=n, dtype=mx.uint8)
        return builtins.bytes(_tolist_list(vals))

    def poisson(self, lam: Any = 1.0, size: Any | None = None) -> mx.array:
        return _poisson_impl(lam=lam, size=size, key=self._next_key())

    def binomial(self, n: Any, p: Any, size: Any | None = None) -> mx.array:
        return _binomial_impl(n=n, p=p, size=size, key=self._next_key())

    def truncated_normal(
        self,
        lower: Any,
        upper: Any,
        size: Any | None = None,
        dtype: Any | None = None,
    ) -> mx.array:
        out = mx.random.truncated_normal(
            lower,
            upper,
            shape=_normalize_size(size),
            dtype=_mx_random_float_dtype_arg(dtype),
            key=self._next_key(),
        )
        return _coerce_public_float_output_for_dtype(out, dtype)

    def multivariate_normal(self, mean: Any, cov: Any, size: Any | None = None, dtype: Any | None = None) -> mx.array:
        out = mx.random.multivariate_normal(
            _asarray(mean),
            _asarray(cov),
            shape=_shape_for_mlx(size),
            dtype=_mx_random_float_dtype_arg(dtype),
            key=self._next_key(),
            stream=_CPU_STREAM,
        )
        return _coerce_public_float_output_for_dtype(out, dtype)

    def spawn(self, n_children: int = 1) -> list["Generator"]:
        n = int(n_children)
        if n < 0:
            msg = "n_children must be non-negative"
            raise ValueError(msg)
        if n == 0:
            return []
        keys = mx.random.split(self._next_key(), n)
        return [Generator(keys[i]) for i in range(n)]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        onp = bridge.numpy_module()
        if not hasattr(onp.random.Generator, name):
            msg = f"{type(self).__name__!r} object has no attribute {name!r}"
            raise AttributeError(msg)
        class_attr = getattr(onp.random.Generator, name)
        if callable(class_attr):

            def method(*args: Any, **kwargs: Any) -> Any:
                bridge.record_fallback(f"random.generator.dynamic:{name}")
                rng = _numpy_rng_from_key(self._next_key())
                fn = getattr(rng, name)
                result = fn(
                    *(bridge.coerce_to_numpy(arg) for arg in args),
                    **{k: bridge.coerce_to_numpy(v) for k, v in kwargs.items()},
                )
                return bridge.wrap_from_numpy(result)

            method.__name__ = name
            method.__doc__ = getattr(class_attr, "__doc__", None)
            method.__mumpy_bridge_kind__ = "numpy"
            method.__mumpy_fallback_site__ = f"random.generator.dynamic:{name}"
            method.__mumpy_bridge_dynamic__ = True
            return method
        rng = _numpy_rng_from_key(self._key)
        return getattr(rng, name)

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        with suppress(NotImplementedError, bridge.FallbackDisabledError):
            names |= {n for n in dir(bridge.numpy_module().random.Generator) if not n.startswith("_")}
        return sorted(names)


def _wrap_module_random_float_output(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _coerce_public_float_output(fn(*args, **kwargs))

    return wrapper


def _wrap_module_random_int_output(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _coerce_public_int_output(fn(*args, **kwargs))

    return wrapper


def _wrap_generator_random_float_output(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(self: Generator, *args: Any, **kwargs: Any) -> Any:
        return _coerce_public_float_output(fn(self, *args, **kwargs))

    return wrapper


def _wrap_generator_random_int_output(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(self: Generator, *args: Any, **kwargs: Any) -> Any:
        return _coerce_public_int_output(fn(self, *args, **kwargs))

    return wrapper


for _name in (
    "random",
    "random_sample",
    "sample",
    "ranf",
    "rand",
    "randn",
    "uniform",
    "normal",
    "standard_normal",
    "laplace",
    "gumbel",
    "exponential",
    "standard_exponential",
    "rayleigh",
    "logistic",
    "lognormal",
    "standard_cauchy",
    "triangular",
    "weibull",
    "power",
    "pareto",
    "standard_gamma",
    "gamma",
    "chisquare",
    "f",
    "beta",
    "dirichlet",
    "standard_t",
):
    globals()[_name] = _wrap_module_random_float_output(globals()[_name])

for _name in (
    "geometric",
    "multinomial",
    "negative_binomial",
    "poisson",
    "binomial",
):
    globals()[_name] = _wrap_module_random_int_output(globals()[_name])

for _name in (
    "random",
    "random_sample",
    "uniform",
    "normal",
    "standard_normal",
    "laplace",
    "gumbel",
    "exponential",
    "standard_exponential",
    "rayleigh",
    "logistic",
    "lognormal",
    "standard_cauchy",
    "triangular",
    "weibull",
    "power",
    "pareto",
    "standard_gamma",
    "gamma",
    "chisquare",
    "f",
    "beta",
    "dirichlet",
    "standard_t",
):
    setattr(Generator, _name, _wrap_generator_random_float_output(getattr(Generator, _name)))

for _name in (
    "geometric",
    "multinomial",
    "negative_binomial",
    "poisson",
    "binomial",
):
    setattr(Generator, _name, _wrap_generator_random_int_output(getattr(Generator, _name)))


def default_rng(seed: int | mx.array | None = None) -> Generator:
    return Generator(seed)


def __getattr__(name: str) -> Any:
    return dynamic_exports.resolve_module_fallback_attr(
        name,
        cache=_FALLBACK_ATTR_CACHE,
        native_namespace=mx.random,
        numpy_namespace_getter=lambda: bridge.numpy_module().random,
        bridge_module=bridge,
        bridge_namespace="random",
        public_module_name="mumpy.random",
    )


def __dir__() -> list[str]:
    return dynamic_exports.dynamic_dir(
        __all__,
        mx.random,
        numpy_namespace_getter=lambda: bridge.numpy_module().random,
        numpy_exceptions=_NUMPY_EXPORT_EXCEPTIONS,
    )


__all__ = [
    "Generator",
    "bernoulli",
    "beta",
    "binomial",
    "chisquare",
    "choice",
    "default_rng",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "gumbel",
    "integers",
    "key",
    "laplace",
    "multinomial",
    "multivariate_normal",
    "negative_binomial",
    "normal",
    "permutation",
    "permuted",
    "poisson",
    "rand",
    "randint",
    "randn",
    "random",
    "random_sample",
    "ranf",
    "sample",
    "seed",
    "shuffle",
    "split",
    "standard_gamma",
    "standard_normal",
    "standard_t",
    "state",
    "truncated_normal",
    "uniform",
]


__all__ = dynamic_exports.extend_all_with_fallback_names(  # pyright: ignore[reportUnsupportedDunderAll]
    __all__,
    mx.random,
    numpy_namespace_getter=lambda: bridge.numpy_module().random,
    numpy_exceptions=_NUMPY_EXPORT_EXCEPTIONS,
)
