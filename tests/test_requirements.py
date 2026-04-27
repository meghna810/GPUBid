"""Phase 3 tests: requirement library + sampler."""

import numpy as np

from gpubid.domain.requirements import REQUIREMENT_LIBRARY, sample_requirements


def test_library_has_at_least_12_entries():
    assert len(REQUIREMENT_LIBRARY) >= 12


def test_library_covers_diversity_rubric():
    """Per spec §4.2: cover the workload × urgency matrix."""
    pairs = {(r.workload_category, r.expected_urgency_band) for r in REQUIREMENT_LIBRARY}
    required = {
        ("training", "urgent"), ("training", "soon"), ("training", "routine"),
        ("fine_tuning", "urgent"), ("fine_tuning", "routine"),
        ("inference_realtime", "urgent"),
        ("inference_batch", "soon"), ("inference_batch", "routine"),
        ("evaluation_sweep", "soon"),
    }
    missing = required - pairs
    assert not missing, f"missing diversity coverage: {missing}"


def test_sample_without_replacement_when_n_le_library():
    rng = np.random.default_rng(42)
    samples = sample_requirements(8, rng)
    assert len(samples) == 8
    # All distinct
    assert len({r.requirement_id for r in samples}) == 8


def test_sample_falls_back_to_with_replacement_when_n_gt_library():
    rng = np.random.default_rng(42)
    samples = sample_requirements(20, rng)
    assert len(samples) == 20  # exactly 20 even though library is 12


def test_sample_is_deterministic_with_seed():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    a = sample_requirements(5, rng_a)
    b = sample_requirements(5, rng_b)
    assert [r.requirement_id for r in a] == [r.requirement_id for r in b]


def test_each_requirement_has_persona_and_raw_text():
    for r in REQUIREMENT_LIBRARY:
        assert r.persona, f"empty persona on {r.requirement_id}"
        assert len(r.raw_text) > 50, f"raw_text too short on {r.requirement_id}"
