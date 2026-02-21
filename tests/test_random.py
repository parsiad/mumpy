"""Tests for module-level and generator-based random APIs."""

import numpy as np
import pytest

import mumpy as mp

from .conftest import assert_array_equal, to_numpy


def _dtype_name(x):
    dt = getattr(x, "dtype", x)
    return str(dt).removeprefix("mlx.core.")


def test_module_seed_reproducibility():
    mp.random.seed(12345)
    a1 = mp.random.rand(2, 3)
    b1 = mp.random.randint(0, 10, size=(2, 3))
    e1 = mp.random.exponential(scale=2.5, size=5)
    p1 = mp.random.poisson(lam=3.0, size=5)
    bi1 = mp.random.binomial(n=10, p=0.25, size=5)

    mp.random.seed(12345)
    a2 = mp.random.rand(2, 3)
    b2 = mp.random.randint(0, 10, size=(2, 3))
    e2 = mp.random.exponential(scale=2.5, size=5)
    p2 = mp.random.poisson(lam=3.0, size=5)
    bi2 = mp.random.binomial(n=10, p=0.25, size=5)

    assert_array_equal(a1, a2)
    assert_array_equal(b1, b2)
    assert_array_equal(e1, e2)
    assert_array_equal(p1, p2)
    assert_array_equal(bi1, bi2)


def test_module_random_shapes_and_bounds():
    mp.random.seed(0)
    r = mp.random.random((2, 4))
    rs = mp.random.random_sample(3)
    smp = mp.random.sample((1, 2))
    ranf = mp.random.ranf(5)
    rn = mp.random.randn(2, 2)
    uni = mp.random.uniform(low=-2.0, high=3.0, size=1000)
    norm = mp.random.normal(loc=5.0, scale=2.0, size=10)
    bern = mp.random.bernoulli(p=0.2, size=200)
    lap = mp.random.laplace(loc=1.0, scale=0.5, size=(3, 2))
    expo = mp.random.exponential(scale=2.0, size=100)
    pois = mp.random.poisson(lam=4.0, size=100)
    bino = mp.random.binomial(n=10, p=0.3, size=100)
    trunc = mp.random.truncated_normal(-1.0, 1.0, size=500)
    mvn = mp.random.multivariate_normal([0.0, 1.0], [[1.0, 0.1], [0.1, 2.0]], size=4)

    assert r.shape == (2, 4)
    assert rs.shape == (3,)
    assert smp.shape == (1, 2)
    assert ranf.shape == (5,)
    assert rn.shape == (2, 2)
    assert uni.shape == (1000,)
    assert norm.shape == (10,)
    assert bern.shape == (200,)
    assert lap.shape == (3, 2)
    assert expo.shape == (100,)
    assert pois.shape == (100,)
    assert bino.shape == (100,)
    assert trunc.shape == (500,)
    assert mvn.shape == (4, 2)

    uni_np = to_numpy(uni)
    assert np.all(uni_np >= -2.0)
    assert np.all(uni_np < 3.0)
    bern_np = to_numpy(bern)
    assert set(np.unique(bern_np)).issubset({False, True})
    assert np.all(to_numpy(expo) >= 0.0)
    assert np.all(to_numpy(pois) >= 0)
    assert np.all((to_numpy(bino) >= 0) & (to_numpy(bino) <= 10))
    trunc_np = to_numpy(trunc)
    assert np.all(trunc_np > -1.0)
    assert np.all(trunc_np < 1.0)


def test_randint_and_integers_endpoint_behavior():
    mp.random.seed(1)
    x = mp.random.randint(5, size=200)
    y = mp.random.integers(2, 4, size=200, endpoint=True)
    z = mp.random.integers(3, size=50)

    x_np = to_numpy(x)
    y_np = to_numpy(y)
    z_np = to_numpy(z)
    assert np.all((x_np >= 0) & (x_np < 5))
    assert np.all((y_np >= 2) & (y_np <= 4))
    assert np.all((z_np >= 0) & (z_np < 3))


def test_permutation_shuffle_and_choice_basic_properties():
    mp.random.seed(7)
    perm = mp.random.permutation(10)
    assert_array_equal(np.sort(to_numpy(perm)), np.arange(10))

    arr = mp.array([10, 20, 30, 40, 50])
    shuf = mp.random.shuffle(arr)
    assert_array_equal(np.sort(to_numpy(shuf)), np.array([10, 20, 30, 40, 50]))

    c1 = mp.random.choice(5, size=20, replace=True)
    c1_np = to_numpy(c1)
    assert c1.shape == (20,)
    assert np.all((c1_np >= 0) & (c1_np < 5))

    c2 = mp.random.choice([10, 20, 30, 40], size=3, replace=False)
    c2_np = to_numpy(c2)
    assert c2.shape == (3,)
    assert len(np.unique(c2_np)) == 3
    assert set(c2_np.tolist()).issubset({10, 20, 30, 40})


def test_choice_with_probabilities_and_errors():
    mp.random.seed(99)
    draws = mp.random.choice([0, 1, 2], size=500, p=[0.05, 0.1, 0.85])
    counts = np.bincount(to_numpy(draws), minlength=3)
    assert counts[2] > counts[1] > counts[0]

    sample = mp.random.choice([1, 2, 3], size=2, replace=False, p=[0.2, 0.3, 0.5])
    assert sample.shape == (2,)
    assert len(np.unique(to_numpy(sample))) == 2

    grid = mp.random.choice([1, 2, 3, 4], size=(2, 2), replace=True)
    assert grid.shape == (2, 2)

    mat = np.arange(12).reshape(3, 4)
    row_pick = mp.random.choice(mat, size=2, replace=False, axis=0)
    col_pick = mp.random.choice(mat, size=2, replace=False, axis=1)
    assert row_pick.shape == (2, 4)
    assert col_pick.shape == (3, 2)

    with pytest.raises(ValueError, match="a must be a positive integer"):
        mp.random.choice(0)


def test_generator_reproducibility_and_state_progression():
    g1 = mp.random.default_rng(123)
    g2 = mp.random.default_rng(123)

    assert_array_equal(g1.random((2, 3)), g2.random((2, 3)))
    assert_array_equal(g1.normal(size=4), g2.normal(size=4))
    assert_array_equal(g1.integers(0, 10, size=5), g2.integers(0, 10, size=5))
    assert_array_equal(g1.choice(6, size=4, replace=False), g2.choice(6, size=4, replace=False))

    g = mp.random.default_rng(321)
    a = g.random(3)
    b = g.random(3)
    assert not np.array_equal(to_numpy(a), to_numpy(b))
    assert g.key.shape == (2,)


def test_generator_helpers_and_permutation():
    g = mp.random.default_rng(42)
    perm = g.permutation(8)
    assert_array_equal(np.sort(to_numpy(perm)), np.arange(8))

    arr = mp.array([1, 2, 3, 4])
    shuf = g.shuffle(arr)
    assert_array_equal(np.sort(to_numpy(shuf)), np.array([1, 2, 3, 4]))

    bern = g.bernoulli(p=0.7, size=20)
    assert set(np.unique(to_numpy(bern))).issubset({False, True})

    lap = g.laplace(size=5)
    expo = g.exponential(size=10)
    pois = g.poisson(lam=2.5, size=10)
    bino = g.binomial(n=8, p=0.4, size=10)
    trunc = g.truncated_normal(-2.0, 2.0, size=20)
    mvn = g.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], size=3)
    assert lap.shape == (5,)
    assert expo.shape == (10,)
    assert pois.shape == (10,)
    assert bino.shape == (10,)
    assert trunc.shape == (20,)
    assert mvn.shape == (3, 2)
    assert np.all(to_numpy(expo) >= 0)
    assert np.all((to_numpy(bino) >= 0) & (to_numpy(bino) <= 8))


def test_generator_new_distributions_are_reproducible():
    g1 = mp.random.default_rng(2024)
    g2 = mp.random.default_rng(2024)
    assert_array_equal(g1.exponential(scale=1.5, size=7), g2.exponential(scale=1.5, size=7))
    assert_array_equal(g1.poisson(lam=3.2, size=7), g2.poisson(lam=3.2, size=7))
    assert_array_equal(g1.binomial(n=12, p=0.6, size=7), g2.binomial(n=12, p=0.6, size=7))


def test_gamma_family_and_multinomial_dirichlet_properties():
    mp.random.seed(1234)

    sg = mp.random.standard_gamma(shape=2.0, size=100)
    ga = mp.random.gamma(shape=2.0, scale=1.5, size=100)
    cs = mp.random.chisquare(df=4.0, size=100)
    ff = mp.random.f(dfnum=5.0, dfden=7.0, size=100)
    be = mp.random.beta(a=2.0, b=5.0, size=100)
    st = mp.random.standard_t(df=6.0, size=100)
    nb = mp.random.negative_binomial(n=4.0, p=0.35, size=100)
    di = mp.random.dirichlet([1.0, 2.0, 3.0], size=20)
    mu = mp.random.multinomial(7, [0.2, 0.3, 0.5], size=20)
    pm = mp.random.permuted(np.arange(20).reshape(5, 4), axis=1)

    assert sg.shape == (100,)
    assert ga.shape == (100,)
    assert cs.shape == (100,)
    assert ff.shape == (100,)
    assert be.shape == (100,)
    assert st.shape == (100,)
    assert nb.shape == (100,)
    assert di.shape == (20, 3)
    assert mu.shape == (20, 3)
    assert pm.shape == (5, 4)

    assert np.all(to_numpy(sg) >= 0)
    assert np.all(to_numpy(ga) >= 0)
    assert np.all(to_numpy(cs) >= 0)
    assert np.all(to_numpy(ff) >= 0)
    be_np = to_numpy(be)
    assert np.all((be_np >= 0) & (be_np <= 1))
    assert np.all(to_numpy(nb) >= 0)
    di_np = to_numpy(di)
    np.testing.assert_allclose(di_np.sum(axis=1), np.ones(20, dtype=np.float32), rtol=1e-5, atol=1e-5)
    assert np.all(di_np > 0)
    assert_array_equal(to_numpy(mu).sum(axis=1), np.full(20, 7, dtype=np.int32))
    assert_array_equal(np.sort(to_numpy(pm), axis=1), np.sort(np.arange(20).reshape(5, 4), axis=1))

    g1 = mp.random.default_rng(55)
    g2 = mp.random.default_rng(55)
    assert_array_equal(g1.gamma(shape=2.0, scale=1.5, size=8), g2.gamma(shape=2.0, scale=1.5, size=8))
    assert_array_equal(g1.beta(a=2.0, b=5.0, size=8), g2.beta(a=2.0, b=5.0, size=8))
    assert_array_equal(g1.dirichlet([1.0, 2.0, 3.0], size=4), g2.dirichlet([1.0, 2.0, 3.0], size=4))
    assert_array_equal(g1.multinomial(6, [0.2, 0.3, 0.5], size=4), g2.multinomial(6, [0.2, 0.3, 0.5], size=4))
    assert_array_equal(g1.negative_binomial(n=4.0, p=0.35, size=8), g2.negative_binomial(n=4.0, p=0.35, size=8))


def test_large_parameter_poisson_binomial_fast_paths_basic_properties():
    mp.random.seed(987)
    pois = mp.random.poisson(lam=200.0, size=2000)
    bino = mp.random.binomial(n=1000, p=0.3, size=2000)

    pois_np = to_numpy(pois)
    bino_np = to_numpy(bino)
    assert np.all(pois_np >= 0)
    assert np.all((bino_np >= 0) & (bino_np <= 1000))
    # Broad bounds to avoid flakiness while still catching pathological outputs.
    assert abs(float(pois_np.mean()) - 200.0) < 30.0
    assert abs(float(bino_np.mean()) - 300.0) < 50.0

    g1 = mp.random.default_rng(777)
    g2 = mp.random.default_rng(777)
    assert_array_equal(g1.poisson(lam=200.0, size=20), g2.poisson(lam=200.0, size=20))
    assert_array_equal(g1.binomial(n=1000, p=0.3, size=20), g2.binomial(n=1000, p=0.3, size=20))


def test_random_default_dtype_parity_for_selected_module_and_generator_paths():
    mp.random.seed(101)
    assert _dtype_name(mp.random.random(4)) == np.random.default_rng(101).random(4).dtype.name
    mp.random.seed(102)
    assert _dtype_name(mp.random.randn(4)) == np.random.default_rng(102).standard_normal(4).dtype.name
    mp.random.seed(103)
    assert _dtype_name(mp.random.uniform(size=4)) == np.random.default_rng(103).uniform(size=4).dtype.name
    mp.random.seed(104)
    assert _dtype_name(mp.random.normal(size=4)) == np.random.default_rng(104).normal(size=4).dtype.name
    mp.random.seed(105)
    assert _dtype_name(mp.random.randint(10, size=4)) == np.random.default_rng(105).integers(10, size=4).dtype.name
    mp.random.seed(106)
    assert _dtype_name(mp.random.poisson(3.0, size=4)) == np.random.default_rng(106).poisson(3.0, size=4).dtype.name
    mp.random.seed(107)
    assert _dtype_name(mp.random.choice(5, size=4)) == np.random.default_rng(107).choice(5, size=4).dtype.name
    mp.random.seed(108)
    assert _dtype_name(mp.random.permutation(5)) == np.random.default_rng(108).permutation(5).dtype.name
    mp.random.seed(109)
    assert _dtype_name(mp.random.truncated_normal(-1.0, 1.0, size=4)) == "float64"
    mp.random.seed(110)
    assert _dtype_name(mp.random.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], size=2)) == "float64"

    g = mp.random.default_rng(201)
    ng = np.random.default_rng(201)
    assert _dtype_name(g.random(4)) == ng.random(4).dtype.name
    g = mp.random.default_rng(202)
    ng = np.random.default_rng(202)
    assert _dtype_name(g.standard_normal(4)) == ng.standard_normal(4).dtype.name
    g = mp.random.default_rng(203)
    ng = np.random.default_rng(203)
    assert _dtype_name(g.integers(10, size=4)) == ng.integers(10, size=4).dtype.name
    g = mp.random.default_rng(204)
    ng = np.random.default_rng(204)
    assert _dtype_name(g.poisson(3.0, size=4)) == ng.poisson(3.0, size=4).dtype.name
    g = mp.random.default_rng(205)
    ng = np.random.default_rng(205)
    assert _dtype_name(g.choice(5, size=4)) == ng.choice(5, size=4).dtype.name
    g = mp.random.default_rng(206)
    ng = np.random.default_rng(206)
    assert _dtype_name(g.permutation(5)) == ng.permutation(5).dtype.name
    g = mp.random.default_rng(207)
    assert _dtype_name(g.truncated_normal(-1.0, 1.0, size=4)) == "float64"
    g = mp.random.default_rng(208)
    assert _dtype_name(g.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], size=2)) == "float64"


def test_random_explicit_dtype_overrides_and_preserve_input_dtype_semantics():
    assert _dtype_name(mp.random.randint(10, size=4, dtype=mp.int32)) == "int32"
    assert _dtype_name(mp.random.integers(10, size=4, dtype=mp.int16)) == "int16"
    assert _dtype_name(mp.random.truncated_normal(-1.0, 1.0, size=4, dtype=mp.float32)) == "float32"
    assert (
        _dtype_name(
            mp.random.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], size=2, dtype=mp.float32),
        )
        == "float32"
    )

    g = mp.random.default_rng(303)
    assert _dtype_name(g.integers(10, size=4, dtype=mp.int32)) == "int32"
    assert _dtype_name(g.truncated_normal(-1.0, 1.0, size=4, dtype=mp.float32)) == "float32"
    assert (
        _dtype_name(
            g.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], size=2, dtype=mp.float32),
        )
        == "float32"
    )

    arr = mp.array([10, 20, 30], dtype=mp.int16)
    assert _dtype_name(mp.random.choice(arr, size=2)) == "int16"
    assert _dtype_name(mp.random.permutation(arr)) == "int16"
    g2 = mp.random.default_rng(404)
    assert _dtype_name(g2.choice(arr, size=2)) == "int16"
    assert _dtype_name(g2.permutation(arr)) == "int16"
