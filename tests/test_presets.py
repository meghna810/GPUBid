"""Preset save/load roundtrip — no LLM calls needed."""

import tempfile
from pathlib import Path

from gpubid.experiments.bake_presets import (
    PRESET_SPECS,
    deserialize_snapshot,
    load_preset,
    save_preset,
    serialize_snapshot,
    PresetSpec,
)
from gpubid.market import generate_market
from gpubid.viz.trading_floor import collect_snapshots


def test_preset_specs_have_unique_ids():
    ids = [s.scenario_id for s in PRESET_SPECS]
    assert len(ids) == len(set(ids))


def test_save_and_load_preset_roundtrip():
    market = generate_market(4, 2, "tight", seed=42)
    snapshots = collect_snapshots(market, max_rounds=3)
    spec = PresetSpec(
        scenario_id="test_roundtrip",
        description="roundtrip test",
        n_buyers=4, n_sellers=2, regime="tight", seed=42, max_rounds=3,
    )
    metadata = {"buyer_provider": "deterministic", "seller_provider": "deterministic"}
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        path = save_preset(spec=spec, market=market, snapshots=list(snapshots), metadata=metadata, output_dir=out_dir)
        assert path.exists()
        loaded = load_preset(path)
    assert loaded["scenario_id"] == "test_roundtrip"
    assert loaded["market"].id == market.id
    assert len(loaded["snapshots"]) == len(snapshots)
    # Final snapshot should match
    last_a = snapshots[-1]
    last_b = loaded["snapshots"][-1]
    assert last_a.round_n == last_b.round_n
    assert len(last_a.all_deals) == len(last_b.all_deals)
    if last_a.all_deals:
        assert last_a.all_deals[0].id == last_b.all_deals[0].id


def test_serialize_snapshot_idempotent():
    market = generate_market(3, 2, "slack", seed=1)
    snapshots = collect_snapshots(market, max_rounds=2)
    snap = snapshots[-1]
    payload = serialize_snapshot(snap)
    rebuilt = deserialize_snapshot(payload)
    assert rebuilt.round_n == snap.round_n
    assert rebuilt.is_final == snap.is_final
    assert len(rebuilt.all_deals) == len(snap.all_deals)
