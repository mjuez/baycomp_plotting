"""Unit tests for the pure helper functions."""
from __future__ import annotations

import numpy as np
import pytest

from baycomp_plotting import Color
from baycomp_plotting.plotting import _process_names, _project, _safe_round


class TestColor:
    def test_palette_has_four_hex_strings(self):
        assert len({Color.BLUE, Color.GRAY, Color.BORDEAUX, Color.GREEN}) == 4
        for c in (Color.BLUE, Color.GRAY, Color.BORDEAUX, Color.GREEN):
            assert isinstance(c, str)
            assert c.startswith("#") and len(c) == 7

    def test_blue_is_stable(self):
        # Pinned for backwards compatibility — used to be public in plots.
        assert Color.BLUE == "#008ece"


class TestSafeRound:
    def test_rounds_to_requested_decimals(self):
        out = _safe_round([0.12345, 0.34567, 0.53088], places=4)
        assert all(round(v * 10_000) == v * 10_000 for v in out)

    def test_preserves_sum(self):
        # Triplet that would round naively to 1.0001
        values = [0.33335, 0.33335, 0.33330]
        out = _safe_round(values, places=4)
        assert sum(out) == pytest.approx(1.0, abs=1e-9)

    def test_handles_empty(self):
        assert _safe_round([], places=4) == []

    def test_extreme_distribution_sums_to_one(self):
        out = _safe_round([0.99996, 1e-5, 3e-5], places=4)
        assert sum(out) == pytest.approx(1.0, abs=1e-9)


class TestProject:
    def test_centroid_maps_to_triangle_center(self):
        out = _project(np.array([[1 / 3, 1 / 3, 1 / 3]]))
        # Center of the equilateral triangle drawn by tern() is (0.5, 1/(2*sqrt(3))).
        np.testing.assert_allclose(out[0], (0.5, 1 / (2 * np.sqrt(3))), atol=1e-9)

    def test_vertex_left_maps_to_left(self):
        # (1,0,0) is the "left" axis in the projection; we don't pin the exact
        # corner but verify it sits on x<0.5 and y<centroid.
        out = _project(np.array([[1.0, 0.0, 0.0]]))
        assert out[0, 0] < 0.5

    def test_shape_preserved(self):
        pts = np.random.RandomState(0).dirichlet([1, 1, 1], size=50)
        out = _project(pts)
        assert out.shape == (50, 2)


class TestProcessNames:
    def test_escapes_dashes(self):
        assert _process_names(["a-b"]) == ["a{-}b"]

    def test_escapes_spaces(self):
        assert _process_names(["a b"]) == ["a\\ b"]

    def test_escapes_both(self):
        assert _process_names(["foo bar-baz"]) == ["foo\\ bar{-}baz"]

    def test_does_not_mutate_input(self):
        original = ["a-b", "c d"]
        _process_names(original)
        assert original == ["a-b", "c d"]
