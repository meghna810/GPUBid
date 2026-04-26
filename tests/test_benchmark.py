"""VCG and posted-price baselines against hand-computable markets."""

from gpubid.benchmark.posted_price import compute_posted_prices, solve_posted_price
from gpubid.benchmark.vcg import compute_welfare, solve_vcg
from gpubid.eval.metrics import compute_metrics, gini
from gpubid.market import generate_market
from gpubid.schema import (
    Buyer,
    CapacitySlot,
    GPUType,
    InterruptionTolerance,
    Job,
    Market,
    Seller,
)


def _two_by_two_market_simple() -> Market:
    """Hand-checkable 2-buyer × 2-seller market.

    Both buyers want 1 H100 for 2 hours, time window 8-14.
    Seller A: 1 H100 at reserve $4, time 8-14.
    Seller B: 1 H100 at reserve $5, time 8-14.
    Buyer X: value $10/GPU-hr (high)
    Buyer Y: value $7/GPU-hr  (low)

    Welfare-optimal allocation: each buyer paired with a separate seller. Specifically,
    X→A and Y→B both work (or X→B and Y→A). Both pair-up combinations have welfare:
    (10-4)*1*2 + (7-5)*1*2 = 12 + 4 = 16
    OR
    (10-5)*1*2 + (7-4)*1*2 = 10 + 6 = 16
    Same total welfare, just different price transfers. VCG should return 16.
    """
    bx = Buyer(
        id="BX", label="X",
        urgency=0.5,
        job=Job(
            qty=1, acceptable_gpus=(GPUType.H100,),
            earliest_start=8, latest_finish=14, duration=2,
            interruption_tolerance=InterruptionTolerance.NONE,
            max_value_per_gpu_hr=10.0,
        ),
    )
    by = Buyer(
        id="BY", label="Y",
        urgency=0.5,
        job=Job(
            qty=1, acceptable_gpus=(GPUType.H100,),
            earliest_start=8, latest_finish=14, duration=2,
            interruption_tolerance=InterruptionTolerance.NONE,
            max_value_per_gpu_hr=7.0,
        ),
    )
    sa = Seller(
        id="SA", label="Cloud A",
        capacity_slots=(
            CapacitySlot(id="SA-slot0", gpu_type=GPUType.H100, start=8, duration=6, qty=1, reserve_per_gpu_hr=4.0),
        ),
    )
    sb = Seller(
        id="SB", label="Cloud B",
        capacity_slots=(
            CapacitySlot(id="SB-slot0", gpu_type=GPUType.H100, start=8, duration=6, qty=1, reserve_per_gpu_hr=5.0),
        ),
    )
    return Market(id="2x2-simple", regime="tight", seed=0, buyers=(bx, by), sellers=(sa, sb))


def _two_by_two_one_seller_only() -> Market:
    """2 buyers, 1 H100 slot.

    Seller A has qty=1 H100. Two buyers compete: X (value $10), Y (value $6).
    Welfare-optimal: assign to X (higher surplus). welfare = (10-4)*1*2 = 12.
    """
    bx = Buyer(
        id="BX", label="X", urgency=0.5,
        job=Job(qty=1, acceptable_gpus=(GPUType.H100,), earliest_start=8, latest_finish=14,
                duration=2, interruption_tolerance=InterruptionTolerance.NONE, max_value_per_gpu_hr=10.0),
    )
    by = Buyer(
        id="BY", label="Y", urgency=0.5,
        job=Job(qty=1, acceptable_gpus=(GPUType.H100,), earliest_start=8, latest_finish=14,
                duration=2, interruption_tolerance=InterruptionTolerance.NONE, max_value_per_gpu_hr=6.0),
    )
    sa = Seller(
        id="SA", label="Cloud A",
        capacity_slots=(
            CapacitySlot(id="SA-slot0", gpu_type=GPUType.H100, start=8, duration=6, qty=1, reserve_per_gpu_hr=4.0),
        ),
    )
    return Market(id="2x1", regime="tight", seed=0, buyers=(bx, by), sellers=(sa,))


def _incompatible_market() -> Market:
    """Buyer wants H100 but seller only has A100 — VCG should give welfare=0."""
    bx = Buyer(
        id="BX", label="X", urgency=0.5,
        job=Job(qty=1, acceptable_gpus=(GPUType.H100,), earliest_start=8, latest_finish=14,
                duration=2, interruption_tolerance=InterruptionTolerance.NONE, max_value_per_gpu_hr=10.0),
    )
    sa = Seller(
        id="SA", label="A",
        capacity_slots=(
            CapacitySlot(id="SA-slot0", gpu_type=GPUType.A100, start=8, duration=6, qty=1, reserve_per_gpu_hr=2.0),
        ),
    )
    return Market(id="incompat", regime="tight", seed=0, buyers=(bx,), sellers=(sa,))


# ---------------------------------------------------------------------------
# VCG
# ---------------------------------------------------------------------------


def test_vcg_simple_2x2_matches_hand_calculation():
    m = _two_by_two_market_simple()
    result = solve_vcg(m)
    assert result.solver_status == "Optimal"
    assert result.welfare == 16.0
    assert len(result.assignments) == 2
    assigned_buyers = {a[0] for a in result.assignments}
    assert assigned_buyers == {"BX", "BY"}


def test_vcg_one_seller_picks_higher_value_buyer():
    m = _two_by_two_one_seller_only()
    result = solve_vcg(m)
    assert result.solver_status == "Optimal"
    assert result.welfare == 12.0
    assert result.assignments == [("BX", "SA-slot0")]


def test_vcg_returns_zero_when_no_compatible_pairs():
    m = _incompatible_market()
    result = solve_vcg(m)
    assert result.welfare == 0.0
    assert result.assignments == []


def test_vcg_runs_on_synthetic_markets():
    """VCG should solve the synthetic markets in well under the time limit."""
    for seed in range(5):
        m = generate_market(8, 4, "tight", seed=seed)
        result = solve_vcg(m)
        assert result.solver_status == "Optimal"
        assert result.welfare >= 0


# ---------------------------------------------------------------------------
# Posted-price
# ---------------------------------------------------------------------------


def test_posted_prices_use_median_reserve_with_markup():
    m = _two_by_two_market_simple()
    posted = compute_posted_prices(m, markup=1.0)
    # Median of {4, 5} = 4.5
    assert posted[GPUType.H100] == 4.5
    posted = compute_posted_prices(m, markup=1.25)
    # 4.5 × 1.25 = 5.625, rounded to 5.62 or 5.63 depending on banker's rounding
    assert abs(posted[GPUType.H100] - 5.625) < 0.01


def test_posted_price_buyers_below_post_walk_away():
    """If a buyer's value is below the posted price, they don't transact."""
    bx = Buyer(
        id="BX", label="X", urgency=0.5,
        job=Job(qty=1, acceptable_gpus=(GPUType.H100,), earliest_start=8, latest_finish=14,
                duration=2, interruption_tolerance=InterruptionTolerance.NONE,
                max_value_per_gpu_hr=4.0),  # below median × markup
    )
    sa = Seller(
        id="SA", label="A",
        capacity_slots=(
            CapacitySlot(id="SA-slot0", gpu_type=GPUType.H100, start=8, duration=6, qty=1, reserve_per_gpu_hr=4.0),
        ),
    )
    m = Market(id="below", regime="tight", seed=0, buyers=(bx,), sellers=(sa,))
    result = solve_posted_price(m)
    assert result.deals == []


def test_posted_price_picks_lowest_priced_compatible():
    m = _two_by_two_market_simple()
    result = solve_posted_price(m)
    # Posted price for H100 = median reserve × 1.25 = 4.5 × 1.25 = 5.625
    # Buyer X (value 10) and Y (value 7) both can afford at 5.625.
    # Both sellers' reserves are below 5.625 too.
    assert len(result.deals) == 2
    for d in result.deals:
        assert d.price_per_gpu_hr == 5.62 or d.price_per_gpu_hr == 5.63


# ---------------------------------------------------------------------------
# VCG vs Posted welfare ordering
# ---------------------------------------------------------------------------


def test_vcg_welfare_at_least_posted_welfare():
    """Welfare-optimal should always weakly dominate posted-price welfare."""
    for seed in range(8):
        m = generate_market(8, 4, "tight", seed=seed)
        vcg = solve_vcg(m)
        posted = solve_posted_price(m)
        assert vcg.welfare >= posted.welfare - 1e-6, f"seed {seed}: VCG ${vcg.welfare} < posted ${posted.welfare}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_gini_zero_on_uniform_distribution():
    assert gini([5.0, 5.0, 5.0, 5.0]) == 0.0


def test_gini_one_on_concentrated_distribution():
    """Almost all welfare on a single agent → Gini close to 1."""
    g = gini([0.0, 0.0, 0.0, 100.0])
    # For n=4 with one nonzero, gini = (n-1)/n = 0.75 — that's the max for finite n.
    assert 0.7 < g <= 0.8


def test_compute_metrics_runs():
    m = _two_by_two_market_simple()
    vcg = solve_vcg(m)
    metrics = compute_metrics(m, vcg.deals)
    assert metrics.welfare == 16.0
    assert metrics.n_deals == 2
    assert metrics.n_buyers_filled == 2
