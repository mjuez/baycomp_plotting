"""Image-regression tests for the plotting functions.

Baselines live in ``tests/baseline_images``. To regenerate them run::

    pytest --mpl-generate-path=tests/baseline_images

To check existing plots against the baselines (the default in CI)::

    pytest --mpl
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import baycomp_plotting as bplt


@pytest.mark.mpl_image_compare(baseline_dir="baseline_images", tolerance=10)
def test_dens_single_posterior(correlated_ttest_posterior):
    fig = bplt.dens(correlated_ttest_posterior, label="B", color=bplt.Color.BLUE)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline_images", tolerance=10)
def test_dens_two_posteriors(correlated_ttest_posterior):
    fig = bplt.dens(correlated_ttest_posterior, label="A", color=bplt.Color.BLUE)
    fig.add_posterior(
        correlated_ttest_posterior, label="B", ls=(0, (5, 1)), color=bplt.Color.BORDEAUX,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline_images", tolerance=10)
def test_tern_basic(signed_rank_posterior):
    fig = bplt.tern(signed_rank_posterior, names=["A", "B"])
    return fig


class TestStructural:
    """Cheap structural checks that run even without the --mpl baselines."""

    def test_dens_returns_figure_with_legend_data(self, correlated_ttest_posterior):
        fig = bplt.dens(correlated_ttest_posterior, label="X", color=bplt.Color.GREEN)
        ax = fig.axes[0]
        assert len(ax.lines) >= 3  # 1 density + 2 ROPE verticals
        assert hasattr(fig, "add_posterior")
        plt.close(fig)

    def test_add_posterior_appends_a_line(self, correlated_ttest_posterior):
        fig = bplt.dens(correlated_ttest_posterior, label="A", color=bplt.Color.BLUE)
        n_before = len(fig.axes[0].lines)
        fig.add_posterior(correlated_ttest_posterior, label="B", color=bplt.Color.BORDEAUX)
        assert len(fig.axes[0].lines) == n_before + 1
        plt.close(fig)

    def test_tern_renders_three_vertex_labels(self, signed_rank_posterior):
        fig = bplt.tern(signed_rank_posterior, names=["L", "R"])
        ax = fig.axes[0]
        text_contents = [t.get_text() for t in ax.texts]
        assert any("L" in t for t in text_contents)
        assert any("R" in t for t in text_contents)
        assert any("ROPE" in t for t in text_contents)
        plt.close(fig)


class TestRopeAxvline:
    """The two orange axvlines must reflect the posterior's ``rope`` parameter,
    not a hardcoded value.
    """

    @pytest.mark.parametrize("rope", [0.001, 0.01, 0.05])
    def test_axvlines_match_rope(self, rng, rope):
        bc = pytest.importorskip("baycomp")
        base = rng.normal(0.85, 0.015, 10)
        x = base + rng.normal(0.0, 0.005, 10)
        y = base + rng.normal(0.015, 0.005, 10)
        posterior = bc.CorrelatedTTest(x, y, rope=rope, runs=1)

        fig = bplt.dens(posterior, label="X")
        ax = fig.axes[0]
        vlines = sorted(
            line.get_xdata()[0] for line in ax.lines
            if line.get_xdata()[0] == line.get_xdata()[1]  # vertical line
        )
        # Two of those vertical lines should be at +/- rope
        assert pytest.approx(rope, abs=1e-12) in vlines
        assert pytest.approx(-rope, abs=1e-12) in vlines
        plt.close(fig)


class TestDegeneratePosterior:
    """``dens`` should fail loudly when the posterior is degenerate (var=0),
    instead of crashing on NaN axis limits inside matplotlib.
    """

    def test_var_zero_raises_value_error(self, rng):
        bc = pytest.importorskip("baycomp")
        identical = (rng.uniform(size=20) < 0.85).astype(float)
        posterior = bc.CorrelatedTTest(identical, identical.copy(), rope=0.01, runs=1)

        with pytest.raises(ValueError, match="degenerate"):
            bplt.dens(posterior, label="X")
