"""VC Commitment Pacing Model – Monte-Carlo Simulation
=====================================================

Simulates cash flows and NAV dynamics for a hypothetical fund-of-funds
vehicle using historical Preqin transaction data.

Key features
------------
* Two-layer model: underlying fund NAVs projected to current age via
  median Preqin patterns, then tracked forward.
* **Recallable distributions**: excess cash (dists > calls) is paid to
  the LP as a recallable distribution.  Shortfalls (calls > dists) are
  funded first by recalling previously distributed cash, then by
  drawing on the LP reserve.
* FoF NAV = invested NAV only (no cash held in the fund).
* LP purchase price = projected portfolio NAV; reserve = purchase price.
* **Dynamic aging**: all funds age dynamically.  Burn-in funds start
  at their projected age and advance each year; new fund slots start
  at age 1.  Each simulation year, 10 new fund slots are created to
  deploy the annual commitment.  Calls are capped at each fund's
  remaining unfunded commitment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_BUCKET_SIZE = 5  # Minimum observations per (year, age) bucket before fallback

# ---------------------------------------------------------------------------
# Asset class mapping
# ---------------------------------------------------------------------------

def map_universe_asset_class(asset_class: str) -> str:
    """Map raw asset class strings to coarse categories."""
    if not isinstance(asset_class, str):
        return "Other"
    val = asset_class.lower()
    if "venture" in val:
        return "Venture Capital"
    if "buyout" in val or "equity" in val:
        return "Buyout"
    if "real estate" in val:
        return "Real Estate"
    if "infrastructure" in val:
        return "Infrastructure"
    return "Other"


# ---------------------------------------------------------------------------
# NAV growth & distribution helper – Modified Dietz
# ---------------------------------------------------------------------------

def _compute_nav_growth_by_fund_age(
    df: pd.DataFrame,
    clip_lo: float = -1.0,
    clip_hi: float = 3.0,
    dist_nav_clip_hi: float = 2.0,
) -> pd.DataFrame:
    """Derive NAV growth rates (Modified Dietz) and distribution-to-NAV ratios.

    Returns
    -------
    pd.DataFrame
        Columns: ``FUND ID``, ``fund_age``, ``nav_growth_pct``, ``dist_nav_pct``.
    """
    vals = df[df["TRANSACTION TYPE"] == "Value"].copy()
    vals["TRANSACTION DATE"] = pd.to_datetime(vals["TRANSACTION DATE"], errors="coerce")
    nav_fa = (
        vals.sort_values("TRANSACTION DATE")
        .groupby(["FUND ID", "fund_age"])
        .agg(nav_end=("TRANSACTION AMOUNT", "last"))
        .reset_index()
        .sort_values(["FUND ID", "fund_age"])
    )
    nav_fa["nav_begin"] = nav_fa.groupby("FUND ID")["nav_end"].shift(1).fillna(0.0)

    c = df[df["TRANSACTION TYPE"].str.lower().str.contains("capital call", na=False)]
    d = df[df["TRANSACTION TYPE"].str.lower().str.contains("distribution", na=False)]
    c_agg = (
        c.groupby(["FUND ID", "fund_age"])["TRANSACTION AMOUNT"]
        .apply(lambda x: x.abs().sum())
        .reset_index(name="period_calls")
    )
    d_agg = (
        d.groupby(["FUND ID", "fund_age"])["TRANSACTION AMOUNT"]
        .apply(lambda x: x.abs().sum())
        .reset_index(name="period_dists")
    )
    nav_fa = nav_fa.merge(c_agg, on=["FUND ID", "fund_age"], how="left")
    nav_fa = nav_fa.merge(d_agg, on=["FUND ID", "fund_age"], how="left")
    nav_fa["period_calls"] = nav_fa["period_calls"].fillna(0.0)
    nav_fa["period_dists"] = nav_fa["period_dists"].fillna(0.0)

    def _g(row):
        numerator = (row["nav_end"] - row["nav_begin"]
                     - row["period_calls"] + row["period_dists"])
        denom = (row["nav_begin"]
                 + 0.5 * row["period_calls"]
                 - 0.5 * row["period_dists"])
        if denom <= 0:
            return 0.0
        return numerator / denom

    nav_fa["nav_growth_pct"] = nav_fa.apply(_g, axis=1).clip(clip_lo, clip_hi)
    nav_fa["dist_nav_pct"] = np.where(
        nav_fa["nav_begin"] > 0,
        nav_fa["period_dists"] / nav_fa["nav_begin"],
        0.0,
    )
    nav_fa["dist_nav_pct"] = nav_fa["dist_nav_pct"].clip(0.0, dist_nav_clip_hi)

    return nav_fa[["FUND ID", "fund_age", "nav_growth_pct", "dist_nav_pct"]]


# ---------------------------------------------------------------------------
# Pattern construction
# ---------------------------------------------------------------------------

def compute_pattern_dict(
    universe_df: pd.DataFrame, asset_class: str
) -> Dict[int, List[Tuple[float, float, float]]]:
    """Compute call, distribution, and NAV-growth patterns by fund age.

    Returns ``pattern_dict[age]`` → list of
    ``(call_pct, dist_nav_pct, nav_growth_pct)`` tuples.
    """
    df = universe_df.copy()
    df["asset_class_mapped"] = df["ASSET CLASS"].apply(map_universe_asset_class)
    df = df[df["asset_class_mapped"] == asset_class].copy()

    df["transaction_year"] = pd.to_datetime(df["TRANSACTION DATE"], errors="coerce").dt.year
    df["vintage_year"] = pd.to_numeric(df["VINTAGE / INCEPTION YEAR"], errors="coerce")
    df["fund_age"] = df["transaction_year"] - df["vintage_year"] + 1
    df = df[(df["fund_age"].notna()) & (df["fund_age"] > 0)].copy()

    df["call_amount"] = np.where(
        df["TRANSACTION TYPE"].str.lower().str.contains("capital call", na=False),
        df["TRANSACTION AMOUNT"].abs(), 0.0,
    )
    df["dist_amount"] = np.where(
        df["TRANSACTION TYPE"].str.lower().str.contains("distribution", na=False),
        df["TRANSACTION AMOUNT"].abs(), 0.0,
    )

    total_called = df.groupby("FUND ID")["call_amount"].sum().to_dict()

    pat = (
        df.groupby(["FUND ID", "fund_age"])
        .agg(call_amount=("call_amount", "sum"), dist_amount=("dist_amount", "sum"))
        .reset_index()
    )
    pat["total_called"] = pat["FUND ID"].map(total_called)
    pat["call_pct"] = pat.apply(
        lambda r: r["call_amount"] / r["total_called"] if r["total_called"] > 0 else 0.0, axis=1,
    )

    nav_metrics = _compute_nav_growth_by_fund_age(df)
    pat = pat.merge(nav_metrics, on=["FUND ID", "fund_age"], how="left")
    pat["nav_growth_pct"] = pat["nav_growth_pct"].fillna(0.0)
    pat["dist_nav_pct"] = pat["dist_nav_pct"].fillna(0.0)

    pattern_dict: Dict[int, List[Tuple[float, float, float]]] = {}
    for _, row in pat.iterrows():
        age = int(row["fund_age"])
        pattern_dict.setdefault(age, []).append(
            (row["call_pct"], row["dist_nav_pct"], row["nav_growth_pct"])
        )
    return pattern_dict


def compute_pattern_dict_by_year(
    universe_df: pd.DataFrame, asset_class: str
) -> Dict[int, Dict[int, List[Tuple[float, float, float]]]]:
    """Compute patterns by transaction year and fund age.

    Returns ``patterns_by_year[year][age]`` → list of
    ``(call_pct, dist_nav_pct, nav_growth_pct)`` tuples.
    """
    df = universe_df.copy()
    df["asset_class_mapped"] = df["ASSET CLASS"].apply(map_universe_asset_class)
    df = df[df["asset_class_mapped"] == asset_class].copy()
    if df.empty:
        return {}

    df["transaction_year"] = pd.to_datetime(df["TRANSACTION DATE"], errors="coerce").dt.year
    df["vintage_year"] = pd.to_numeric(df["VINTAGE / INCEPTION YEAR"], errors="coerce")
    df["fund_age"] = df["transaction_year"] - df["vintage_year"] + 1
    df = df[(df["fund_age"].notna()) & (df["fund_age"] > 0)].copy()

    df["call_amount"] = np.where(
        df["TRANSACTION TYPE"].str.lower().str.contains("capital call", na=False),
        df["TRANSACTION AMOUNT"].abs(), 0.0,
    )
    df["dist_amount"] = np.where(
        df["TRANSACTION TYPE"].str.lower().str.contains("distribution", na=False),
        df["TRANSACTION AMOUNT"].abs(), 0.0,
    )

    total_called = df.groupby("FUND ID")["call_amount"].sum().to_dict()

    grouped = (
        df.groupby(["FUND ID", "transaction_year", "fund_age"])
        .agg(call_amount=("call_amount", "sum"), dist_amount=("dist_amount", "sum"))
        .reset_index()
    )
    grouped["total_called"] = grouped["FUND ID"].map(total_called)
    grouped["call_pct"] = grouped.apply(
        lambda r: r["call_amount"] / r["total_called"] if r["total_called"] > 0 else 0.0, axis=1,
    )

    nav_metrics = _compute_nav_growth_by_fund_age(df)
    grouped = grouped.merge(nav_metrics, on=["FUND ID", "fund_age"], how="left")
    grouped["nav_growth_pct"] = grouped["nav_growth_pct"].fillna(0.0)
    grouped["dist_nav_pct"] = grouped["dist_nav_pct"].fillna(0.0)

    patterns_by_year: Dict[int, Dict[int, List[Tuple[float, float, float]]]] = {}
    for _, row in grouped.iterrows():
        year = int(row["transaction_year"])
        age = int(row["fund_age"])
        patterns_by_year.setdefault(year, {}).setdefault(age, []).append(
            (row["call_pct"], row["dist_nav_pct"], row["nav_growth_pct"])
        )
    return patterns_by_year


# ---------------------------------------------------------------------------
# Initial NAV projection (burn-in)
# ---------------------------------------------------------------------------

def compute_initial_fund_navs(
    pattern_dict: Dict[int, List[Tuple[float, float, float]]],
    ages: np.ndarray,
    commitments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project each fund's expected NAV and cumulative calls at its current age.

    Uses **median** patterns per age to deterministically simulate each
    fund from age 1 through its current age.

    Returns
    -------
    initial_navs : np.ndarray
        Projected NAV per fund at its current age.
    initial_cum_calls : np.ndarray
        Projected cumulative capital called per fund through its current age.
    """
    max_age = max(pattern_dict.keys())

    median_patterns: Dict[int, Tuple[float, float, float]] = {}
    for age_key, plist in pattern_dict.items():
        median_patterns[age_key] = (
            float(np.median([p[0] for p in plist])),
            float(np.median([p[1] for p in plist])),
            float(np.median([p[2] for p in plist])),
        )

    n_funds = len(ages)
    initial_navs = np.zeros(n_funds, dtype=float)
    initial_cum_calls = np.zeros(n_funds, dtype=float)

    for i in range(n_funds):
        fund_age = int(ages[i])
        nav = 0.0
        cum_calls = 0.0
        commitment = float(commitments[i])

        for a in range(1, fund_age + 1):
            a_key = min(a, max_age)
            call_pct, dist_nav_pct, growth = median_patterns.get(a_key, (0.0, 0.0, 0.0))

            call_amt = call_pct * commitment
            call_amt = min(call_amt, max(commitment - cum_calls, 0.0))
            cum_calls += call_amt

            nav_begin = nav
            dist_amt = dist_nav_pct * nav_begin
            dist_amt = min(dist_amt, max(nav_begin, 0.0))

            nav = nav * (1.0 + growth) + call_amt - dist_amt
            nav = max(nav, 0.0)

        initial_navs[i] = nav
        initial_cum_calls[i] = cum_calls

    return initial_navs, initial_cum_calls


# ---------------------------------------------------------------------------
# Simulation engine – constant age, year-specific patterns
# ---------------------------------------------------------------------------

def run_simulation_constant_age_by_year(
    portfolio_df: pd.DataFrame,
    patterns_by_year: Dict[int, Dict[int, List[Tuple[float, float, float]]]],
    pattern_dict_fallback: Dict[int, List[Tuple[float, float, float]]],
    scenario_years: Iterable[int],
    base_year_end: int,
    num_simulations: int = 500,
    return_fund_level: bool = False,
    min_bucket_size: int = MIN_BUCKET_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate cash flows with recallable distribution mechanics.

    **Initialisation (burn-in):**
    Each fund's underlying NAV is projected to its current age using
    median Preqin patterns.  LP purchases the portfolio at NAV and
    commits an equal amount as reserve (50% buffer).

    **Dynamic aging:**
    All funds age dynamically.  Burn-in funds start at their projected
    age and advance by 1 each simulation year; new fund slots start at
    age 1.  Each simulation year, 10 new fund slots are created to
    deploy the annual commitment ($1 x funds_per_vintage = $10).

    **Recallable distributions:**
    * Excess cash (dists > calls) is paid to the LP as a recallable
      distribution.  The fund holds **no cash**.
    * When calls > dists, the shortfall is funded by first recalling
      previously distributed cash, then drawing on the LP reserve.
    * Recallable balance does **not** affect LP Unfunded %.
    """
    import numpy as _np

    if not pattern_dict_fallback:
        raise ValueError("pattern_dict_fallback is empty")
    max_age_global = max(pattern_dict_fallback.keys())

    scenario_years_list = list(scenario_years)
    n_scenario_years = len(scenario_years_list)

    # ---- Portfolio setup ----
    n_initial = len(portfolio_df)
    commitments_initial = portfolio_df["commitment"].to_numpy(dtype=float)
    vintage_years_initial = portfolio_df["vintage_year"].to_numpy(dtype=int)
    burnin_ages = base_year_end - vintage_years_initial + 1

    # Infer pacing parameters from portfolio structure
    funds_per_vintage = int(
        portfolio_df.groupby("vintage_year").size().mode().iloc[0]
    )
    new_fund_commitment = float(commitments_initial[0])

    # Pre-allocate for max portfolio size: initial + new funds over all years
    n_new_total = funds_per_vintage * n_scenario_years
    n_max = n_initial + n_new_total

    commitments_all = _np.zeros(n_max, dtype=float)
    commitments_all[:n_initial] = commitments_initial

    creation_years = _np.zeros(n_max, dtype=int)
    creation_years[:n_initial] = -1  # sentinel: burn-in fund

    # ---- Burn-in ----
    initial_navs, initial_cum_calls = compute_initial_fund_navs(
        pattern_dict_fallback, burnin_ages, commitments_initial,
    )
    lp_nav_purchase = float(initial_navs.sum())
    print(f"Projected portfolio NAV (burn-in): ${lp_nav_purchase:.2f}")
    print(f"  LP purchase price = ${lp_nav_purchase:.2f}")
    print(f"  LP reserve (50%) = ${lp_nav_purchase:.2f}")
    print(f"  LP total commitment = ${2 * lp_nav_purchase:.2f}")
    print(f"  Initial fund commitments = ${commitments_initial.sum():.2f}")
    print(f"  New commitment per year = ${funds_per_vintage * new_fund_commitment:.2f}")

    # Pre-convert fallback patterns
    call_fb: Dict[int, _np.ndarray] = {}
    dist_fb: Dict[int, _np.ndarray] = {}
    navg_fb: Dict[int, _np.ndarray] = {}
    for age_key, plist in pattern_dict_fallback.items():
        if plist:
            call_fb[age_key] = _np.array([p[0] for p in plist], dtype=float)
            dist_fb[age_key] = _np.array([p[1] for p in plist], dtype=float)
            navg_fb[age_key] = _np.array([p[2] for p in plist], dtype=float)

    fund_level_records: List[pd.DataFrame] = []
    agg_records: List[Dict[str, object]] = []

    # ---- Initialise simulation state (pre-allocated for max size) ----
    cum_calls_pf = _np.zeros((num_simulations, n_max), dtype=float)
    cum_calls_pf[:, :n_initial] = initial_cum_calls[None, :]

    nav_pf = _np.zeros((num_simulations, n_max), dtype=float)
    nav_pf[:, :n_initial] = initial_navs[None, :]

    cum_calls_total = cum_calls_pf[:, :n_initial].sum(axis=1).copy()
    cum_dists_total = _np.zeros(num_simulations, dtype=float)

    # LP economics
    lp_reserve = _np.full(num_simulations, lp_nav_purchase, dtype=float)
    recallable_balance = _np.zeros(num_simulations, dtype=float)

    n_active = n_initial
    running_commitment = float(commitments_initial.sum())

    # Fund names for reporting
    fund_names_list = list(portfolio_df["fund_name"].to_numpy())

    for year_idx, year in enumerate(scenario_years_list):
        # ---- Deploy new fund slots ----
        ns = n_active
        ne = n_active + funds_per_vintage
        commitments_all[ns:ne] = new_fund_commitment
        creation_years[ns:ne] = year
        for j in range(funds_per_vintage):
            fund_names_list.append(f"New_{year}{chr(ord('A') + j)}")
        n_active = ne
        running_commitment += funds_per_vintage * new_fund_commitment

        # ---- Current ages for all active funds ----
        ages = _np.empty(n_active, dtype=int)
        ages[:n_initial] = burnin_ages + year_idx  # burn-in: dynamic aging
        if n_active > n_initial:
            ages[n_initial:n_active] = (
                year - creation_years[n_initial:n_active] + 1
            )
        trunc_ages = _np.clip(ages, 1, max_age_global)

        # ---- Sample patterns for all active funds ----
        call_mat = _np.zeros((num_simulations, n_active), dtype=float)
        dnav_mat = _np.zeros((num_simulations, n_active), dtype=float)
        navg_mat = _np.zeros((num_simulations, n_active), dtype=float)

        for age_key in set(trunc_ages.tolist()):
            idx = _np.where(trunc_ages == age_key)[0]
            if idx.size == 0:
                continue
            py = patterns_by_year.get(year, {}).get(age_key)
            if py and len(py) >= min_bucket_size:
                pc = _np.array([p[0] for p in py], dtype=float)
                pd_ = _np.array([p[1] for p in py], dtype=float)
                pn = _np.array([p[2] for p in py], dtype=float)
            else:
                pc = call_fb.get(age_key)
                pd_ = dist_fb.get(age_key)
                pn = navg_fb.get(age_key)
                if pc is None or pc.size == 0:
                    continue
            L = len(pc)
            ri = _np.random.randint(0, L, size=(num_simulations, idx.size))
            call_mat[:, idx] = pc[ri]
            dnav_mat[:, idx] = pd_[ri]
            navg_mat[:, idx] = pn[ri]

        # ---- Calls: capped at remaining commitment ----
        c_all = commitments_all[:n_active]
        call_amounts = call_mat * c_all
        remaining = c_all - cum_calls_pf[:, :n_active]
        remaining = _np.maximum(remaining, 0.0)
        call_amounts = _np.minimum(call_amounts, remaining)
        cum_calls_pf[:, :n_active] += call_amounts

        # ---- NAV snapshot ----
        nav_begin = nav_pf[:, :n_active].copy()
        nav_total_begin = nav_begin.sum(axis=1)

        # NAV-weighted growth rate
        w = _np.where(
            nav_total_begin[:, None] > 0,
            nav_begin / nav_total_begin[:, None],
            1.0 / max(n_active, 1),
        )
        avg_nav_growth = (navg_mat * w).sum(axis=1)

        # Distributions: NAV-based, capped
        dist_amounts = dnav_mat * nav_begin
        dist_amounts = _np.minimum(
            dist_amounts, _np.maximum(nav_begin, 0.0)
        )

        # Update underlying fund NAVs
        nav_pf[:, :n_active] = (
            nav_pf[:, :n_active] * (1.0 + navg_mat)
            + call_amounts - dist_amounts
        )
        nav_pf[:, :n_active] = _np.maximum(nav_pf[:, :n_active], 0.0)

        # ---- Aggregate cash flows ----
        total_call = call_amounts.sum(axis=1)
        total_dist = dist_amounts.sum(axis=1)
        net_cf = total_dist - total_call

        cum_calls_total += total_call
        cum_dists_total += total_dist

        mgr_unfunded_pct = _np.where(
            running_commitment > 0,
            (running_commitment - cum_calls_total) / running_commitment,
            _np.ones(num_simulations),
        )

        # ---- Recallable distribution mechanics ----
        rec_dist = _np.maximum(net_cf, 0.0)
        recallable_balance += rec_dist

        shortfall = _np.maximum(-net_cf, 0.0)
        recall = _np.minimum(shortfall, recallable_balance)
        recallable_balance -= recall
        lp_draw = shortfall - recall
        lp_reserve -= lp_draw

        # ---- FoF NAV = invested NAV only (no cash) ----
        invested_nav = nav_pf[:, :n_active].sum(axis=1)
        fof_nav = invested_nav

        # ---- LP Unfunded % ----
        lp_denom = fof_nav + lp_reserve
        lp_unfunded_pct = _np.where(
            lp_denom > 0,
            lp_reserve / lp_denom,
            _np.ones(num_simulations),
        )
        lp_breach = _np.where(lp_unfunded_pct < 0.20, 1, 0)

        for sim in range(num_simulations):
            agg_records.append({
                "scenario_year": year,
                "simulation_number": sim,
                "total_call": float(total_call[sim]),
                "total_distribution": float(total_dist[sim]),
                "net_cashflow": float(net_cf[sim]),
                "manager_unfunded_pct": float(mgr_unfunded_pct[sim]),
                "nav_growth_pct": float(avg_nav_growth[sim]),
                "invested_nav": float(invested_nav[sim]),
                "fof_nav": float(fof_nav[sim]),
                "recallable_balance": float(recallable_balance[sim]),
                "recallable_dist": float(rec_dist[sim]),
                "recall": float(recall[sim]),
                "lp_reserve_draw": float(lp_draw[sim]),
                "lp_reserve": float(lp_reserve[sim]),
                "lp_unfunded_pct": float(lp_unfunded_pct[sim]),
                "lp_unfunded_breach": int(lp_breach[sim]),
                "cumulative_commitment": running_commitment,
                "cumulative_contributions": float(cum_calls_total[sim]),
                "cumulative_distributions_total": float(cum_dists_total[sim]),
                "underlying_unfunded_pct": float(
                    (running_commitment - cum_calls_total[sim])
                    / running_commitment
                    if running_commitment > 0 else 1.0
                ),
            })

        if return_fund_level:
            nr = num_simulations * n_active
            si = _np.repeat(_np.arange(num_simulations), n_active)
            fn = _np.tile(
                _np.array(fund_names_list[:n_active]), num_simulations
            )
            fa = _np.tile(ages, num_simulations)
            fl_df = pd.DataFrame({
                "fund_name": fn,
                "scenario_year": year,
                "simulation_number": si,
                "fund_age": fa,
                "call_pct": call_mat.reshape(nr),
                "dist_nav_pct": dnav_mat.reshape(nr),
                "nav_growth_pct": navg_mat.reshape(nr),
                "call_amount": call_amounts.reshape(nr),
                "dist_amount": dist_amounts.reshape(nr),
                "underlying_nav": nav_pf[:, :n_active].reshape(nr),
            })
            fund_level_records.append(fl_df)

        print(f"simulation year {year} completed (active funds: {n_active})")

    fund_df = (
        pd.concat(fund_level_records, ignore_index=True)
        if return_fund_level and fund_level_records
        else pd.DataFrame()
    )
    agg_df = pd.DataFrame(agg_records)
    return fund_df, agg_df

def run_simulation(
    portfolio_df: pd.DataFrame,
    pattern_dict: Dict[int, List[Tuple[float, float, float]]],
    scenario_years: Iterable[int],
    num_simulations: int = 500,
    return_fund_level: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte-Carlo simulation with dynamic fund ages.

    Same recallable distribution architecture.
    """
    import numpy as _np

    if not pattern_dict:
        raise ValueError("pattern_dict is empty")
    max_age = max(pattern_dict.keys())

    n_funds = len(portfolio_df)
    commitments = portfolio_df["commitment"].to_numpy(dtype=float)
    vintage_years = portfolio_df["vintage_year"].to_numpy(dtype=int)

    scenario_years_list = list(scenario_years)
    first_year = scenario_years_list[0]
    initial_ages = np.maximum(first_year - vintage_years + 1, 1)

    initial_navs, initial_cum_calls = compute_initial_fund_navs(
        pattern_dict, initial_ages, commitments,
    )
    lp_nav_purchase = float(initial_navs.sum())
    print(f"Projected portfolio NAV (burn-in): ${lp_nav_purchase:.2f}")
    print(f"  LP purchase = ${lp_nav_purchase:.2f}, reserve = ${lp_nav_purchase:.2f}")

    call_arrays: Dict[int, _np.ndarray] = {}
    dist_arrays: Dict[int, _np.ndarray] = {}
    navg_arrays: Dict[int, _np.ndarray] = {}
    for age_key, plist in pattern_dict.items():
        if plist:
            call_arrays[age_key] = _np.array([p[0] for p in plist], dtype=float)
            dist_arrays[age_key] = _np.array([p[1] for p in plist], dtype=float)
            navg_arrays[age_key] = _np.array([p[2] for p in plist], dtype=float)

    fund_level_records: List[pd.DataFrame] = []
    agg_records: List[Dict[str, object]] = []

    total_commitment = float(commitments.sum())
    cumulative_calls_per_fund = _np.broadcast_to(
        initial_cum_calls, (num_simulations, n_funds),
    ).copy()
    cumulative_calls = cumulative_calls_per_fund.sum(axis=1).copy()
    cumulative_distributions = _np.zeros(num_simulations, dtype=float)

    lp_reserve = _np.full(num_simulations, lp_nav_purchase, dtype=float)
    recallable_balance = _np.zeros(num_simulations, dtype=float)
    underlying_nav = _np.broadcast_to(
        initial_navs, (num_simulations, n_funds),
    ).copy()

    annual_new_commitment = float(
        portfolio_df.groupby("vintage_year")["commitment"].sum().mean()
    )
    running_cum_commitment = total_commitment

    _median_call_curve_dyn: Dict[int, float] = {}
    for _ak, _pl in pattern_dict.items():
        _median_call_curve_dyn[_ak] = float(_np.median([p[0] for p in _pl]))
    _max_call_age_dyn = max(_median_call_curve_dyn.keys())
    commitment_vintages_dyn: List[Tuple[float, int]] = [(total_commitment, 0)]

    for year in scenario_years_list:
        running_cum_commitment += annual_new_commitment
        commitment_vintages_dyn.append((annual_new_commitment, year))

        vintage_unfunded = 0.0
        for v_amt, v_year in commitment_vintages_dyn:
            if v_year == 0:
                vintage_unfunded += 0.0
            else:
                ages_since = year - v_year + 1
                cum_call_pct = 0.0
                for a in range(1, ages_since + 1):
                    a_key = min(a, _max_call_age_dyn)
                    cum_call_pct += _median_call_curve_dyn.get(a_key, 0.0)
                cum_call_pct = min(cum_call_pct, 1.0)
                vintage_unfunded += v_amt * (1.0 - cum_call_pct)
        ages = year - vintage_years + 1
        call_mat = _np.zeros((num_simulations, n_funds), dtype=float)
        dist_nav_mat = _np.zeros((num_simulations, n_funds), dtype=float)
        navg_mat = _np.zeros((num_simulations, n_funds), dtype=float)

        for age_key in set(_np.minimum(_np.maximum(ages, 1), max_age)):
            idx = _np.where(_np.minimum(_np.maximum(ages, 1), max_age) == age_key)[0]
            if idx.size == 0:
                continue
            p_call = call_arrays.get(age_key)
            p_dist = dist_arrays.get(age_key)
            p_navg = navg_arrays.get(age_key)
            if p_call is None or len(p_call) == 0:
                continue
            L = len(p_call)
            rand_idx = _np.random.randint(0, L, size=(num_simulations, idx.size))
            call_mat[:, idx] = p_call[rand_idx]
            dist_nav_mat[:, idx] = p_dist[rand_idx]
            navg_mat[:, idx] = p_navg[rand_idx]

        # --- Calls: commitment-based, capped at remaining commitment ---
        call_amounts = call_mat * commitments
        remaining = commitments - cumulative_calls_per_fund
        remaining = _np.maximum(remaining, 0.0)
        call_amounts = _np.minimum(call_amounts, remaining)
        cumulative_calls_per_fund += call_amounts

        underlying_nav_begin = underlying_nav.copy()
        underlying_nav_total_begin = underlying_nav_begin.sum(axis=1)

        weights = _np.where(
            underlying_nav_total_begin[:, None] > 0,
            underlying_nav_begin / underlying_nav_total_begin[:, None],
            1.0 / n_funds,
        )
        avg_nav_growth = (navg_mat * weights).sum(axis=1)

        dist_amounts = dist_nav_mat * underlying_nav_begin
        dist_amounts = _np.minimum(dist_amounts, _np.maximum(underlying_nav_begin, 0.0))

        underlying_nav = underlying_nav * (1.0 + navg_mat) + call_amounts - dist_amounts
        underlying_nav = _np.maximum(underlying_nav, 0.0)

        total_call = call_amounts.sum(axis=1)
        total_dist = dist_amounts.sum(axis=1)
        net_cashflow = total_dist - total_call

        cumulative_calls += total_call
        cumulative_distributions += total_dist

        if total_commitment > 0:
            manager_unfunded_pct = (total_commitment - cumulative_calls) / total_commitment
        else:
            manager_unfunded_pct = _np.ones(num_simulations)

        # --- Recallable distribution mechanics ---
        recallable_dist = _np.maximum(net_cashflow, 0.0)
        recallable_balance += recallable_dist

        shortfall = _np.maximum(-net_cashflow, 0.0)
        recall = _np.minimum(shortfall, recallable_balance)
        recallable_balance -= recall
        lp_draw = shortfall - recall
        lp_reserve -= lp_draw

        invested_nav = underlying_nav.sum(axis=1)
        fof_nav = invested_nav

        lp_denominator = fof_nav + lp_reserve
        lp_unfunded_pct = _np.where(
            lp_denominator > 0,
            lp_reserve / lp_denominator,
            _np.ones(num_simulations),
        )
        lp_unfunded_breach = _np.where(lp_unfunded_pct < 0.20, 1, 0)

        for sim in range(num_simulations):
            agg_records.append({
                "scenario_year": year,
                "simulation_number": sim,
                "total_call": float(total_call[sim]),
                "total_distribution": float(total_dist[sim]),
                "net_cashflow": float(net_cashflow[sim]),
                "manager_unfunded_pct": float(manager_unfunded_pct[sim]),
                "nav_growth_pct": float(avg_nav_growth[sim]),
                "invested_nav": float(invested_nav[sim]),
                "fof_nav": float(fof_nav[sim]),
                "recallable_balance": float(recallable_balance[sim]),
                "recallable_dist": float(recallable_dist[sim]),
                "recall": float(recall[sim]),
                "lp_reserve_draw": float(lp_draw[sim]),
                "lp_reserve": float(lp_reserve[sim]),
                "lp_unfunded_pct": float(lp_unfunded_pct[sim]),
                "lp_unfunded_breach": int(lp_unfunded_breach[sim]),
                "cumulative_commitment": running_cum_commitment,
                "cumulative_contributions": float(cumulative_calls[sim]),
                "cumulative_distributions_total": float(cumulative_distributions[sim]),
                "underlying_unfunded_pct": float(
                    vintage_unfunded / running_cum_commitment
                    if running_cum_commitment > 0 else 1.0
                ),
            })

        if return_fund_level:
            num_rows = num_simulations * n_funds
            sim_indices = _np.repeat(_np.arange(num_simulations), n_funds)
            fund_names = _np.tile(portfolio_df["fund_name"].to_numpy(), num_simulations)
            fund_ages = _np.tile(ages, num_simulations)
            fl_df = pd.DataFrame({
                "fund_name": fund_names,
                "scenario_year": year,
                "simulation_number": sim_indices,
                "fund_age": fund_ages,
                "call_pct": call_mat.reshape(num_rows),
                "dist_nav_pct": dist_nav_mat.reshape(num_rows),
                "nav_growth_pct": navg_mat.reshape(num_rows),
                "call_amount": call_amounts.reshape(num_rows),
                "dist_amount": dist_amounts.reshape(num_rows),
                "underlying_nav": underlying_nav.reshape(num_rows),
            })
            fund_level_records.append(fl_df)
        print(f"simulation year {year} completed")

    fund_df = pd.concat(fund_level_records, ignore_index=True) if return_fund_level and fund_level_records else pd.DataFrame()
    agg_df = pd.DataFrame(agg_records)
    return fund_df, agg_df


# ---------------------------------------------------------------------------
# Portfolio builder
# ---------------------------------------------------------------------------

def build_hypothetical_portfolio(
    n_vintages: int = 20,
    funds_per_vintage: int = 10,
    base_year_end: int = 2025,
    commitment: float = 1.0,
    prefix: str = "",
) -> pd.DataFrame:
    """Construct a hypothetical portfolio DataFrame."""
    vintage_start_year = base_year_end - n_vintages + 1
    records: List[Dict[str, object]] = []
    for vintage_index in range(n_vintages):
        vintage_year = vintage_start_year + vintage_index
        for j in range(funds_per_vintage):
            label = chr(ord("A") + j)
            name = f"{prefix}{vintage_index + 1}{label}"
            records.append({
                "fund_name": name,
                "vintage_year": vintage_year,
                "commitment": commitment,
            })
    return pd.DataFrame(records)


def random_choice(options: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Select a random element from a list of pattern tuples."""
    import random
    return random.choice(options)


# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------

def load_universe(filepath: str, encoding: Optional[str] = None) -> pd.DataFrame:
    """Load the universe transaction data with robust encoding handling."""
    if filepath.endswith(('.xlsx', '.xls')):
        import io, builtins
        try:
            with builtins.open(filepath, 'rb') as _f:
                return pd.read_excel(io.BytesIO(_f.read()))
        except (PermissionError, OSError):
            import requests as _req
            _token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            _host = spark.conf.get('spark.databricks.workspaceUrl')
            _resp = _req.get(
                f'https://{_host}/api/2.0/workspace/export',
                headers={'Authorization': f'Bearer {_token}'},
                params={'path': filepath, 'format': 'AUTO', 'direct_download': 'true'},
            )
            _resp.raise_for_status()
            print(f'Loaded via REST API ({len(_resp.content)} bytes)')
            return pd.read_excel(io.BytesIO(_resp.content))
    if encoding is not None:
        return pd.read_csv(filepath, low_memory=False, encoding=encoding)
    encodings = ['utf-8', 'ISO-8859-1', 'latin-1', 'cp1252']
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(filepath, low_memory=False, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error
    return pd.read_csv(filepath, low_memory=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Configuration ----
    universe_path = 'C:\Box\MZhang\Ad Hoc\2026 Cash Flow Stress\Preqin_Cashflow_export-18_Feb_264addba06-22fa-4476-a2d5-ea8269cca01d.xlsx'
    asset_class = "Venture Capital"
    n_vintages = 20
    funds_per_vintage = 10
    base_year_end = 2025
    commitment = 1.0
    simulations = 5000
    output_path = 'C:\Box\MZhang\Ad Hoc\2026 simulation'

    # ---- Load data ----
    universe_df = load_universe(universe_path)

    use_year_specific = True

    # ---- Build portfolio ----
    portfolio_df = build_hypothetical_portfolio(
        n_vintages, funds_per_vintage, base_year_end, commitment,
    )
    scenario_years = list(range(base_year_end - n_vintages + 1, base_year_end + 1))

    # ---- Run simulation ----
    if use_year_specific:
        patterns_by_year = compute_pattern_dict_by_year(universe_df, asset_class)
        pattern_dict_fallback = compute_pattern_dict(universe_df, asset_class)
        if not patterns_by_year:
            raise RuntimeError(f"No year-specific patterns for {asset_class!r}")
        fund_df, agg_df = run_simulation_constant_age_by_year(
            portfolio_df=portfolio_df,
            patterns_by_year=patterns_by_year,
            pattern_dict_fallback=pattern_dict_fallback,
            scenario_years=scenario_years,
            base_year_end=base_year_end,
            num_simulations=simulations,
            return_fund_level=False,
        )
    else:
        pattern_dict = compute_pattern_dict(universe_df, asset_class)
        if not pattern_dict:
            raise RuntimeError(f"No cash-flow patterns for {asset_class!r}")
        fund_df, agg_df = run_simulation(
            portfolio_df=portfolio_df,
            pattern_dict=pattern_dict,
            scenario_years=scenario_years,
            num_simulations=simulations,
            return_fund_level=False,
        )

    # ---- Save results ----
    import os, io as _io, base64 as _b64, requests as _req2
    def _save_csv(df, fpath):
        try:
            df.to_csv(fpath, index=False)
        except (PermissionError, OSError):
            _tok = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            _h = spark.conf.get("spark.databricks.workspaceUrl")
            buf = df.to_csv(index=False)
            _ws = fpath.replace("/Workspace", "", 1) if fpath.startswith("/Workspace") else fpath
            _c = _b64.b64encode(buf.encode('utf-8')).decode('utf-8')
            _r = _req2.post(
                f"https://{_h}/api/2.0/workspace/import",
                headers={"Authorization": f"Bearer {_tok}"},
                json={"path": _ws, "format": "RAW", "content": _c, "overwrite": True},
            )
            _r.raise_for_status()
            print(f"Saved via REST API: {fpath}")
    _save_csv(fund_df, os.path.join(output_path, f"{asset_class}_sim_fund_level_table.csv"))
    _save_csv(agg_df, os.path.join(output_path, f"{asset_class}_sim_aggregate_table.csv"))

    # ---- Summary ----
    summary_cols = [
        "total_call", "total_distribution", "net_cashflow",
        "manager_unfunded_pct", "nav_growth_pct", "invested_nav",
        "fof_nav", "recallable_balance", "recallable_dist", "recall",
        "lp_reserve_draw", "lp_reserve", "lp_unfunded_pct", "lp_unfunded_breach",
        "cumulative_commitment", "cumulative_contributions",
        "cumulative_distributions_total", "underlying_unfunded_pct",
    ]
    summary = agg_df.groupby("scenario_year")[summary_cols].mean()
    print("\nAverage metrics by scenario year:")
    print(summary)

    lp_threshold = 0.20
    breach_df = agg_df.groupby("scenario_year").apply(
        lambda g: (g["lp_unfunded_pct"] < lp_threshold).mean() * 100
    ).rename("pct_sims_below_20pct")
    print(f"\nPercentage of simulations where LP Unfunded < {lp_threshold:.0%} by scenario year:")
    print(breach_df)


if __name__ == "__main__":
    main()