"""Smoke tests: package imports and version is exposed."""

import gpubid


def test_version_exposed():
    # Bump this assertion in lockstep with `src/gpubid/__init__.py` when we cut a new version.
    assert gpubid.__version__ == "0.5.1"


def test_submodules_import():
    from gpubid import schema, market, llm  # noqa: F401
    from gpubid.agents import buyer, seller, deterministic, prompts  # noqa: F401
    from gpubid.engine import board, round_runner, clearing  # noqa: F401
    from gpubid.benchmark import vcg, posted_price  # noqa: F401
    from gpubid.eval import metrics, judge  # noqa: F401
    from gpubid.viz import market_view, trading_floor, trace_view, charts, figures  # noqa: F401
    from gpubid.experiments import bake_presets, run_sweep  # noqa: F401
