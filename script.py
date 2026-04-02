"""VC Commitment Pacing Model – Monte-Carlo Simulation
=====================================================

Simulates cash flows and NAV dynamics for a hypothetical fund-of-funds
vehicle using historical Preqin transaction data.

Key features
------------
* Two-layer model: existing funds initialized from historical
  fund-level states at current age, then tracked forward.
* **Recallable distributions**: excess cash (dists > calls) is paid to
  the LP as a recallable distribution.  Shortfalls (calls > dists) are
  funded first by recalling previously distributed cash, then by
  drawing on the LP reserve.
* FoF NAV = invested NAV only (no cash held in the fund).
* LP purchase price = projected portfolio NAV; reserve = purchase price.
* **Dynamic aging**: all funds age dynamically.  Existing funds start
  at their projected age and advance each year; new fund slots start
  at age 1.  Each simulation year, 10 new fund slots are created to
  deploy the annual commitment.  Calls are capped at each fund's
  remaining unfunded commitment.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Iterable
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PatternRecord = Dict[str, Any]


def _coerce_pattern_record(
    pattern: PatternRecord,
    age: Optional[int] = None,
    transaction_year: Optional[int] = None,
) -> PatternRecord:
    """Normalize pattern payloads into a full record shape."""
    rec = dict(pattern)
    if age is not None and rec.get("fund_age") is None:
        rec["fund_age"] = age
    if transaction_year is not None and rec.get("transaction_year") is None:
        rec["transaction_year"] = transaction_year
    rec.setdefault("fund_id", None)
    rec.setdefault("fund_name", None)
    rec.setdefault("vintage_year", None)
    rec.setdefault("call_pct", 0.0)
    rec.setdefault("dist_nav_pct", 0.0)
    rec.setdefault("nav_growth_pct", 0.0)
    rec.setdefault("nav_begin", 0.0)
    rec.setdefault("nav_end", 0.0)
    rec.setdefault("period_calls", 0.0)
    rec.setdefault("period_dists", 0.0)
    return rec

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
        Columns include ``FUND ID``, ``fund_age``, ``nav_begin``, ``nav_end``,
        ``period_calls``, ``period_dists``, ``nav_growth_pct``, ``dist_nav_pct``.
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

    return nav_fa[
        [
            "FUND ID",
            "fund_age",
            "nav_begin",
            "nav_end",
            "period_calls",
            "period_dists",
            "nav_growth_pct",
            "dist_nav_pct",
        ]
    ]


# ---------------------------------------------------------------------------
# Pattern construction
# ---------------------------------------------------------------------------

def compute_pattern_dict_by_year(
    universe_df: pd.DataFrame, asset_class: str
) -> Dict[int, Dict[int, List[PatternRecord]]]:
    """Compute patterns by transaction year and fund age.

    Returns ``patterns_by_year[year][age]`` → list of rich pattern records.
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

    fund_name_col = "FUND NAME" if "FUND NAME" in df.columns else None
    fund_meta_cols = ["FUND ID", "vintage_year"]
    if fund_name_col is not None:
        fund_meta_cols.append(fund_name_col)
    fund_meta = (
        df.groupby("FUND ID")[fund_meta_cols[1:]]
        .first()
        .reset_index()
        .rename(columns={fund_name_col: "fund_name"} if fund_name_col else {})
    )
    grouped = grouped.merge(fund_meta, on="FUND ID", how="left")

    patterns_by_year: Dict[int, Dict[int, List[PatternRecord]]] = {}
    for _, row in grouped.iterrows():
        year = int(row["transaction_year"])
        age = int(row["fund_age"])
        patterns_by_year.setdefault(year, {}).setdefault(age, []).append({
            "fund_id": row["FUND ID"],
            "fund_name": row.get("fund_name") if "fund_name" in row else None,
            "vintage_year": int(row["vintage_year"]) if pd.notna(row["vintage_year"]) else None,
            "fund_age": age,
            "transaction_year": year,
            "call_pct": float(row["call_pct"]),
            "dist_nav_pct": float(row["dist_nav_pct"]),
            "nav_growth_pct": float(row["nav_growth_pct"]),
            "nav_begin": float(row["nav_begin"]) if pd.notna(row["nav_begin"]) else 0.0,
            "nav_end": float(row["nav_end"]) if pd.notna(row["nav_end"]) else 0.0,
            "period_calls": float(row["period_calls"]) if pd.notna(row["period_calls"]) else 0.0,
            "period_dists": float(row["period_dists"]) if pd.notna(row["period_dists"]) else 0.0,
        })
    return patterns_by_year


# ---------------------------------------------------------------------------
# Simulation engine – constant age, year-specific patterns
# ---------------------------------------------------------------------------

def run_simulation_constant_age_by_year(
    portfolio_df: pd.DataFrame,
    patterns_by_year: Dict[int, Dict[int, List[PatternRecord]]],
    scenario_years: Iterable[int],
    num_simulations: int = 500,
    return_fund_level: bool = False,
    audit_simulation_number: Optional[int] = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate cash flows with recallable distribution mechanics.

    **Initialisation (existing funds):**
    Each existing fund is initialized directly from sampled historical
    fund-level states at the same age (NAV-at-beginning-of-age and
    cumulative called). LP purchases the portfolio at NAV and commits
    an equal amount as reserve.

    **Persistent-fund identity paths:**
    Existing portfolio slots keep their original ``fund_id`` through the
    simulation horizon. Each year, new slots are created and assigned a
    sampled ``fund_id`` from that vintage-year cohort; each assigned
    fund is then followed through its own observed year-level cash-flow
    records. If a slot's selected fund has no row for a scenario year,
    its call/dist/NAV-growth are set to zero for that year.

    **Recallable distributions:**
    * Excess cash (dists > calls) is paid to the LP as a recallable
      distribution.  The fund holds **no cash**.
    * When calls > dists, the shortfall is funded by first recalling
      previously distributed cash, then drawing on the LP reserve.
    * Recallable balance does **not** affect LP Unfunded %.
    """
    import numpy as _np

    if "fund_id" not in portfolio_df.columns:
        raise ValueError("portfolio_df must include 'fund_id' for persistent fund-identity simulation")

    scenario_years_list = list(scenario_years)
    if not scenario_years_list:
        raise ValueError("scenario_years is empty")
    start_year = scenario_years_list[0]
    n_scenario_years = len(scenario_years_list)

    # ---- Portfolio setup ----
    n_initial = len(portfolio_df)
    commitments_initial = portfolio_df["commitment"].to_numpy(dtype=float)
    vintage_years_initial = portfolio_df["vintage_year"].to_numpy(dtype=int)
    burnin_ages = _np.maximum(start_year - vintage_years_initial + 1, 1)

    # Infer pacing parameters from portfolio structure
    funds_per_vintage = int(
        portfolio_df.groupby("vintage_year").size().mode().iloc[0]
    )
    new_fund_commitment = float(commitments_initial[0])
    initial_fund_ids = portfolio_df["fund_id"].to_numpy()

    # Pre-allocate for max portfolio size: initial + new funds over all years
    n_new_total = funds_per_vintage * n_scenario_years
    n_max = n_initial + n_new_total

    commitments_all = _np.zeros(n_max, dtype=float)
    commitments_all[:n_initial] = commitments_initial

    creation_years = _np.zeros(n_max, dtype=int)
    creation_years[:n_initial] = -1  # sentinel: burn-in fund

    fund_level_records: List[pd.DataFrame] = []
    agg_records: List[Dict[str, object]] = []
    audit_records: List[Dict[str, object]] = []

    # Build lookup tables for persistent fund-level paths.
    fund_year_record: Dict[Tuple[object, int], PatternRecord] = {}
    fund_meta: Dict[object, Dict[str, object]] = {}
    fund_ids_by_vintage: Dict[int, List[object]] = {}
    fund_call_curve: Dict[object, Dict[int, float]] = {}
    fund_total_called: Dict[object, float] = {}
    for year, ages_map in patterns_by_year.items():
        for _, records in ages_map.items():
            for raw in records:
                rec = _coerce_pattern_record(raw)
                fund_id = rec.get("fund_id")
                if fund_id is None:
                    continue
                year_key = int(year)
                fund_year_record[(fund_id, year_key)] = rec
                fund_call_curve.setdefault(fund_id, {})[year_key] = float(rec.get("call_pct", 0.0))
                fund_total_called[fund_id] = fund_total_called.get(fund_id, 0.0) + float(
                    rec.get("period_calls", 0.0)
                )
                meta = fund_meta.setdefault(fund_id, {})
                if meta.get("fund_name") is None and rec.get("fund_name") is not None:
                    meta["fund_name"] = rec.get("fund_name")
                if meta.get("vintage_year") is None and rec.get("vintage_year") is not None:
                    meta["vintage_year"] = int(rec.get("vintage_year"))
                vy = rec.get("vintage_year")
                if vy is not None and pd.notna(vy):
                    fund_ids_by_vintage.setdefault(int(vy), []).append(fund_id)

    fund_ids_by_vintage = {
        vy: list(dict.fromkeys(ids)) for vy, ids in fund_ids_by_vintage.items()
    }
    if not fund_ids_by_vintage:
        raise ValueError("patterns_by_year has no fund_id/vintage_year information")

    # ---- Existing-fund initialization from selected fund identities (deterministic) ----
    initial_navs_vec = _np.zeros(n_initial, dtype=float)
    initial_cum_calls_vec = _np.zeros(n_initial, dtype=float)
    for i, fid in enumerate(initial_fund_ids):
        rec_start = fund_year_record.get((fid, int(start_year)))
        if rec_start is not None:
            total_called = float(fund_total_called.get(fid, 0.0))
            nav_begin = float(rec_start.get("nav_begin", 0.0))
            nav_begin_multiple = (nav_begin / total_called) if total_called > 0 else 0.0
            initial_navs_vec[i] = nav_begin_multiple * commitments_initial[i]

        hist_calls = fund_call_curve.get(fid, {})
        called_pct_before_start = float(sum(v for y, v in hist_calls.items() if y < int(start_year)))
        called_pct_before_start = float(_np.clip(called_pct_before_start, 0.0, 1.0))
        initial_cum_calls_vec[i] = commitments_initial[i] * called_pct_before_start

    initial_navs = _np.broadcast_to(initial_navs_vec, (num_simulations, n_initial)).copy()
    initial_cum_calls = _np.broadcast_to(initial_cum_calls_vec, (num_simulations, n_initial)).copy()

    lp_nav_purchase = float(initial_navs_vec.sum())
    nav_per_commitment = _np.divide(
        initial_navs_vec,
        commitments_initial,
        out=_np.zeros_like(initial_navs_vec),
        where=commitments_initial > 0,
    )
    median_initial_nav_per_commitment = float(_np.median(nav_per_commitment))
    print(f"Projected portfolio NAV (initial state): ${lp_nav_purchase:.2f}")
    print(f"  Median initial NAV per $1 commitment = ${median_initial_nav_per_commitment:.4f}")
    print(f"  LP purchase price = ${lp_nav_purchase:.2f}")
    print(f"  LP reserve (50%) = ${lp_nav_purchase:.2f}")
    print(f"  LP total commitment = ${2 * lp_nav_purchase:.2f}")
    print(f"  Initial fund commitments = ${commitments_initial.sum():.2f}")
    print(f"  New commitment per year = ${funds_per_vintage * new_fund_commitment:.2f}")

    # ---- Initialise simulation state (pre-allocated for max size) ----
    cum_calls_pf = _np.zeros((num_simulations, n_max), dtype=float)
    cum_calls_pf[:, :n_initial] = initial_cum_calls

    nav_pf = _np.zeros((num_simulations, n_max), dtype=float)
    nav_pf[:, :n_initial] = initial_navs

    cum_calls_total = cum_calls_pf[:, :n_initial].sum(axis=1).copy()
    cum_dists_total = _np.zeros(num_simulations, dtype=float)

    # LP economics
    lp_reserve = _np.full(num_simulations, lp_nav_purchase, dtype=float)
    recallable_balance = _np.zeros(num_simulations, dtype=float)

    n_active = n_initial
    running_commitment = float(commitments_initial.sum())

    # Per-simulation slot assignments.
    slot_fund_ids = _np.full((num_simulations, n_max), None, dtype=object)
    slot_fund_ids[:, :n_initial] = _np.broadcast_to(initial_fund_ids, (num_simulations, n_initial))

    # Fund names for reporting
    fund_names_list: List[str] = list(portfolio_df["fund_name"].to_numpy())

    # ---- BOY record for initial state (before first scenario-year flows) ----
    initial_manager_unfunded = float(commitments_initial.sum() - initial_cum_calls_vec.sum())
    initial_lp_unfunded = lp_nav_purchase
    initial_lp_reserve = initial_lp_unfunded - initial_manager_unfunded
    for sim in range(num_simulations):
        mgr_unfunded_pct_boy = (
            initial_manager_unfunded / lp_nav_purchase if lp_nav_purchase > 0 else 1.0
        )
        lp_unfunded_pct_boy = initial_lp_unfunded / lp_nav_purchase if lp_nav_purchase > 0 else 1.0
        lp_reserve_pct_boy = initial_lp_reserve / lp_nav_purchase if lp_nav_purchase > 0 else 1.0
        agg_records.append({
            "scenario_year": start_year,
            "period_point": "BOY",
            "simulation_number": sim,
            "total_call": 0.0,
            "total_distribution": 0.0,
            "net_cashflow": 0.0,
            "manager_unfunded_dollar": initial_manager_unfunded,
            "manager_unfunded_pct": float(mgr_unfunded_pct_boy),
            "nav_growth_pct": 0.0,
            "invested_nav": lp_nav_purchase,
            "fof_nav": lp_nav_purchase,
            "recallable_balance": float(recallable_balance[sim]),
            "recallable_dist": 0.0,
            "recall": 0.0,
            "lp_reserve_draw": 0.0,
            "lp_unfunded_dollar": initial_lp_unfunded,
            "lp_reserve_dollar": initial_lp_reserve,
            "lp_reserve": initial_lp_reserve,
            "lp_unfunded_pct": float(lp_unfunded_pct_boy),
            "lp_reserve_pct": float(lp_reserve_pct_boy),
            "lp_unfunded_breach": int(lp_unfunded_pct_boy < 0.20),
            "cumulative_commitment": float(commitments_initial.sum()),
            "cumulative_contributions": float(cum_calls_total[sim]),
            "cumulative_distributions_total": float(cum_dists_total[sim]),
            "underlying_unfunded_pct": float(mgr_unfunded_pct_boy),
        })

    for year_idx, year in enumerate(scenario_years_list):
        # ---- Deploy new fund slots ----
        ns = n_active
        ne = n_active + funds_per_vintage
        commitments_all[ns:ne] = new_fund_commitment
        creation_years[ns:ne] = year
        vintage_pool = fund_ids_by_vintage.get(year, [])
        if vintage_pool:
            sampled_new_ids = _np.random.choice(vintage_pool, size=(num_simulations, funds_per_vintage), replace=True)
            slot_fund_ids[:, ns:ne] = sampled_new_ids
            for j in range(funds_per_vintage):
                fid0 = sampled_new_ids[0, j]
                meta = fund_meta.get(fid0, {})
                default_name = f"New_{year}{chr(ord('A') + j)}"
                fund_names_list.append(str(meta.get("fund_name", default_name)))
        else:
            # No observed funds for this vintage -> new slots remain with zero flows.
            slot_fund_ids[:, ns:ne] = None
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

        # ---- Pull persistent fund-level records for all active slots ----
        call_mat = _np.zeros((num_simulations, n_active), dtype=float)
        dnav_mat = _np.zeros((num_simulations, n_active), dtype=float)
        navg_mat = _np.zeros((num_simulations, n_active), dtype=float)
        sampled_nav_begin_mat = _np.zeros((num_simulations, n_active), dtype=float)
        sampled_fund_id = _np.full((num_simulations, n_active), None, dtype=object)
        sampled_fund_name = _np.full((num_simulations, n_active), None, dtype=object)
        sampled_vintage_year = _np.full((num_simulations, n_active), None, dtype=object)
        sampled_txn_year = _np.full((num_simulations, n_active), None, dtype=object)

        for fund_idx in range(n_active):
            fund_ids_col = slot_fund_ids[:, fund_idx]
            unique_ids = [fid for fid in dict.fromkeys(fund_ids_col.tolist()) if fid is not None]
            for fid in unique_ids:
                rec = fund_year_record.get((fid, int(year)))
                if rec is None:
                    continue
                rec = _coerce_pattern_record(rec, age=int(ages[fund_idx]), transaction_year=int(year))
                mask = (fund_ids_col == fid)
                call_mat[mask, fund_idx] = float(rec["call_pct"])
                dnav_mat[mask, fund_idx] = float(rec["dist_nav_pct"])
                navg_mat[mask, fund_idx] = float(rec["nav_growth_pct"])
                sampled_nav_begin_mat[mask, fund_idx] = float(rec.get("nav_begin", 0.0))
                sampled_fund_id[mask, fund_idx] = rec.get("fund_id")
                sampled_fund_name[mask, fund_idx] = rec.get("fund_name")
                sampled_vintage_year[mask, fund_idx] = rec.get("vintage_year")
                sampled_txn_year[mask, fund_idx] = rec.get("transaction_year")

        # ---- Calls: capped at remaining commitment ----
        c_all = commitments_all[:n_active]
        call_amounts = call_mat * c_all
        remaining = c_all - cum_calls_pf[:, :n_active]
        remaining = _np.maximum(remaining, 0.0)
        call_amounts = _np.minimum(call_amounts, remaining)
        cum_calls_pf[:, :n_active] += call_amounts

        # ---- NAV snapshot ----
        nav_begin = nav_pf[:, :n_active].copy()
        new_fund_mask = _np.broadcast_to(
            (creation_years[:n_active] == year)[None, :],
            nav_begin.shape,
        )
        nav_begin = _np.where(new_fund_mask, 0.0, nav_begin)
        nav_pf[:, :n_active] = nav_begin
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
        nav_end = nav_pf[:, :n_active].copy()

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
                "period_point": "EOY",
                "simulation_number": sim,
                "total_call": float(total_call[sim]),
                "total_distribution": float(total_dist[sim]),
                "net_cashflow": float(net_cf[sim]),
                "manager_unfunded_dollar": float(running_commitment - cum_calls_total[sim]),
                "manager_unfunded_pct": float(mgr_unfunded_pct[sim]),
                "nav_growth_pct": float(avg_nav_growth[sim]),
                "invested_nav": float(invested_nav[sim]),
                "fof_nav": float(fof_nav[sim]),
                "recallable_balance": float(recallable_balance[sim]),
                "recallable_dist": float(rec_dist[sim]),
                "recall": float(recall[sim]),
                "lp_reserve_draw": float(lp_draw[sim]),
                "lp_unfunded_dollar": float(fof_nav[sim]),
                "lp_reserve_dollar": float(lp_reserve[sim]),
                "lp_reserve": float(lp_reserve[sim]),
                "lp_unfunded_pct": float(lp_unfunded_pct[sim]),
                "lp_reserve_pct": float(lp_reserve[sim] / fof_nav[sim]) if fof_nav[sim] > 0 else 1.0,
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
                "sampled_fund_id": sampled_fund_id.reshape(nr),
                "sampled_fund_name": sampled_fund_name.reshape(nr),
                "sampled_vintage_year": sampled_vintage_year.reshape(nr),
                "sampled_transaction_year": sampled_txn_year.reshape(nr),
                "sampled_nav_begin": sampled_nav_begin_mat.reshape(nr),
                "call_amount": call_amounts.reshape(nr),
                "dist_amount": dist_amounts.reshape(nr),
                "underlying_nav_begin": nav_begin.reshape(nr),
                "underlying_nav_end": nav_end.reshape(nr),
            })
            fund_level_records.append(fl_df)

        if audit_simulation_number is not None and 0 <= audit_simulation_number < num_simulations:
            sim_idx = int(audit_simulation_number)
            audit_year_df = pd.DataFrame({
                "scenario_year": year,
                "period_point": "EOY",
                "simulation_number": sim_idx,
                "fund_name": fund_names_list[:n_active],
                "fund_age": ages,
                "call_amount": call_amounts[sim_idx, :],
                "distribution_amount": dist_amounts[sim_idx, :],
                "nav_begin": nav_begin[sim_idx, :],
                "nav_end": nav_end[sim_idx, :],
            })
            audit_records.extend(audit_year_df.to_dict("records"))

        print(f"simulation year {year} completed (active funds: {n_active})")

    fund_df = (
        pd.concat(fund_level_records, ignore_index=True)
        if return_fund_level and fund_level_records
        else pd.DataFrame()
    )
    agg_df = pd.DataFrame(agg_records)
    audit_df = pd.DataFrame(audit_records)
    return fund_df, agg_df, audit_df

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


def build_sampled_portfolio_from_universe(
    universe_df: pd.DataFrame,
    asset_class: str,
    portfolio_end_vintage_year: int = 2025,
    n_vintages: int = 20,
    n_per_vintage: int = 10,
    min_funds_per_vintage: int = 10,
    commitment: float = 1.0,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Build a portfolio by sampling real funds from the raw universe.

    Samples funds for each of the last ``n_vintages`` vintages ending at
    ``portfolio_end_vintage_year``. For each vintage, draws
    ``max(n_per_vintage, min_funds_per_vintage)`` funds. If a vintage has fewer
    unique funds than the target size, sampling is done with replacement.
    """
    rng = np.random.default_rng(random_state)

    df = universe_df.copy()
    df["asset_class_mapped"] = df["ASSET CLASS"].apply(map_universe_asset_class)
    df = df[df["asset_class_mapped"] == asset_class].copy()
    if df.empty:
        raise RuntimeError(f"No universe rows found for asset class {asset_class!r}")

    df["vintage_year"] = pd.to_numeric(df["VINTAGE / INCEPTION YEAR"], errors="coerce")
    df = df[df["vintage_year"].notna()].copy()
    df["vintage_year"] = df["vintage_year"].astype(int)

    fund_name_col = "FUND NAME" if "FUND NAME" in df.columns else "FUND"
    if fund_name_col not in df.columns:
        raise RuntimeError("Expected a fund name column ('FUND NAME' or 'FUND') in universe data")

    fund_pool = (
        df[["FUND ID", fund_name_col, "vintage_year"]]
        .dropna(subset=["FUND ID", fund_name_col])
        .drop_duplicates(subset=["FUND ID", "vintage_year"])
        .rename(columns={fund_name_col: "fund_name", "FUND ID": "fund_id"})
    )
    if fund_pool.empty:
        raise RuntimeError("No distinct funds available after filtering universe data")

    target_size = max(int(n_per_vintage), int(min_funds_per_vintage))
    vintage_start_year = portfolio_end_vintage_year - n_vintages + 1
    target_vintages = list(range(vintage_start_year, portfolio_end_vintage_year + 1))

    sampled_frames: List[pd.DataFrame] = []
    for vintage_year in target_vintages:
        vintage_funds = fund_pool[fund_pool["vintage_year"] == vintage_year]
        if vintage_funds.empty:
            raise RuntimeError(f"No funds found for vintage {vintage_year} and asset class {asset_class!r}")
        replace = len(vintage_funds) < target_size
        sampled_idx = rng.choice(vintage_funds.index.to_numpy(), size=target_size, replace=replace)
        sampled_frames.append(vintage_funds.loc[sampled_idx].copy())

    portfolio_df = pd.concat(sampled_frames, ignore_index=True)
    portfolio_df["commitment"] = float(commitment)
    return portfolio_df[["fund_name", "fund_id", "vintage_year", "commitment"]]


def random_choice(options: List[PatternRecord]) -> PatternRecord:
    """Select a random element from a list of pattern records."""
    import random
    return _coerce_pattern_record(random.choice(options))


# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------

def load_universe(filepath: str, encoding: Optional[str] = None) -> pd.DataFrame:
    """Load the universe transaction data with robust encoding handling."""
    if filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
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
    universe_path = r"C:\Box\MZhang\Ad Hoc\2026 simulation\PF structure\Preqin_Cashflow_export.csv"
    asset_class = "Venture Capital"
    n_vintages = 20
    n_per_vintage = 10
    start_year = 2006
    end_year = 2025
    commitment = 1.0
    simulations = 1
    output_path = r"C:\Box\MZhang\Ad Hoc\2026 simulation\PF structure"
    audit_simulation_number = 0

    # ---- Load data ----
    universe_df = load_universe(universe_path)

    # Intended support status: only year-specific engine is supported.
    use_year_specific = True
    if not use_year_specific:
        raise RuntimeError("Legacy age-only mode is no longer supported.")

    # ---- Build portfolio ----
    portfolio_df = build_sampled_portfolio_from_universe(
        universe_df=universe_df,
        asset_class=asset_class,
        portfolio_end_vintage_year=start_year,
        n_vintages=n_vintages,
        n_per_vintage=n_per_vintage,
        min_funds_per_vintage=10,
        commitment=commitment,
    )
    # Include the start year so both BOY and EOY are recorded for that year.
    scenario_years = list(range(start_year, end_year + 1))

    # ---- Run simulation ----
    patterns_by_year = compute_pattern_dict_by_year(universe_df, asset_class)
    if not patterns_by_year:
        raise RuntimeError(f"No year-specific patterns for {asset_class!r}")
    fund_df, agg_df, audit_df = run_simulation_constant_age_by_year(
        portfolio_df=portfolio_df,
        patterns_by_year=patterns_by_year,
        scenario_years=scenario_years,
        num_simulations=simulations,
        return_fund_level=False,
        audit_simulation_number=audit_simulation_number,
    )

    # ---- Save results ----
    import os
    def _save_csv(df, fpath):
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        df.to_csv(fpath, index=False)
    _save_csv(fund_df, os.path.join(output_path, f"{asset_class}_sim_fund_level_table.csv"))
    _save_csv(agg_df, os.path.join(output_path, f"{asset_class}_sim_aggregate_table.csv"))
    if not audit_df.empty:
        audit_file = os.path.join(output_path, f"{asset_class}_sim_audit_path_sim_{audit_simulation_number}.csv")
        _save_csv(audit_df, audit_file)
        print(f"\nAudit path for simulation {audit_simulation_number}:")
        print(audit_df.to_string(index=False))
        print(f"\nAudit file saved to: {Path(audit_file).resolve()}")

    # ---- Summary ----
    summary_cols = [
        "total_call", "total_distribution", "net_cashflow",
        "manager_unfunded_pct", "nav_growth_pct", "invested_nav",
        "fof_nav", "recallable_balance", "recallable_dist", "recall",
        "lp_reserve_draw", "lp_reserve", "lp_unfunded_pct", "lp_unfunded_breach",
        "cumulative_commitment", "cumulative_contributions",
        "cumulative_distributions_total", "underlying_unfunded_pct",
    ]
    group_cols = ["scenario_year", "period_point"] if "period_point" in agg_df.columns else ["scenario_year"]
    summary = agg_df.groupby(group_cols)[summary_cols].mean()
    print("\nAverage metrics by scenario year:")
    print(summary)

    lp_threshold = 0.20
    breach_group_cols = ["scenario_year", "period_point"] if "period_point" in agg_df.columns else ["scenario_year"]
    breach_df = agg_df.groupby(breach_group_cols).apply(
        lambda g: (g["lp_unfunded_pct"] < lp_threshold).mean() * 100
    ).rename("pct_sims_below_20pct")
    print(f"\nPercentage of simulations where LP Unfunded < {lp_threshold:.0%} by scenario year:")
    print(breach_df)


if __name__ == "__main__":
    main()
