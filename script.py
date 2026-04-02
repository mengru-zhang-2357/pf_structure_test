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
from typing import Any, Dict, List, Tuple, Optional, Iterable, Union
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_BUCKET_SIZE = 5  # Minimum observations per (year, age) bucket before fallback
PatternRecord = Dict[str, Any]
LegacyPatternTuple = Tuple[float, float, float]
PatternLike = Union[PatternRecord, LegacyPatternTuple]


def _coerce_pattern_record(
    pattern: PatternLike,
    age: Optional[int] = None,
    transaction_year: Optional[int] = None,
) -> PatternRecord:
    """Normalize tuple/dict pattern payloads into a record shape."""
    if isinstance(pattern, dict):
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

    call_pct, dist_nav_pct, nav_growth_pct = pattern
    return {
        "fund_id": None,
        "fund_name": None,
        "vintage_year": None,
        "fund_age": age,
        "transaction_year": transaction_year,
        "call_pct": float(call_pct),
        "dist_nav_pct": float(dist_nav_pct),
        "nav_growth_pct": float(nav_growth_pct),
        "nav_begin": 0.0,
        "nav_end": 0.0,
        "period_calls": 0.0,
        "period_dists": 0.0,
    }

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

def compute_pattern_dict(
    universe_df: pd.DataFrame, asset_class: str
) -> Dict[int, List[PatternRecord]]:
    """Compute call, distribution, and NAV-growth patterns by fund age.

    Returns ``pattern_dict[age]`` → list of rich pattern records.
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
    pat = pat.merge(fund_meta, on="FUND ID", how="left")

    pattern_dict: Dict[int, List[PatternRecord]] = {}
    for _, row in pat.iterrows():
        age = int(row["fund_age"])
        pattern_dict.setdefault(age, []).append({
            "fund_id": row["FUND ID"],
            "fund_name": row.get("fund_name") if "fund_name" in row else None,
            "vintage_year": int(row["vintage_year"]) if pd.notna(row["vintage_year"]) else None,
            "fund_age": age,
            "transaction_year": None,
            "call_pct": float(row["call_pct"]),
            "dist_nav_pct": float(row["dist_nav_pct"]),
            "nav_growth_pct": float(row["nav_growth_pct"]),
            "nav_begin": float(row["nav_begin"]) if pd.notna(row["nav_begin"]) else 0.0,
            "nav_end": float(row["nav_end"]) if pd.notna(row["nav_end"]) else 0.0,
            "period_calls": float(row["period_calls"]) if pd.notna(row["period_calls"]) else 0.0,
            "period_dists": float(row["period_dists"]) if pd.notna(row["period_dists"]) else 0.0,
        })
    return pattern_dict


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
# Initial fund-state sampling (fund-level, age-specific)
# ---------------------------------------------------------------------------

def build_initial_state_pool_by_age(
    universe_df: pd.DataFrame,
    asset_class: str,
) -> Dict[int, List[Tuple[float, float]]]:
    """Build age buckets of historical fund states for initialization.

    Returns ``state_pool_by_age[age]`` -> list of tuples:
    ``(nav_begin_multiple, cumulative_called_pct)`` where
    * ``nav_begin_multiple`` = NAV-at-beginning-of-age / total_called
    * ``cumulative_called_pct`` = cumulative called before that age / total_called
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
        df["TRANSACTION AMOUNT"].abs(),
        0.0,
    )

    total_called = df.groupby("FUND ID")["call_amount"].sum().rename("total_called")

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

    calls_fa = (
        df.groupby(["FUND ID", "fund_age"])["call_amount"]
        .sum()
        .reset_index(name="period_calls")
        .sort_values(["FUND ID", "fund_age"])
    )
    calls_fa["cum_calls_before_age"] = (
        calls_fa.groupby("FUND ID")["period_calls"].cumsum() - calls_fa["period_calls"]
    )

    state_df = nav_fa.merge(
        calls_fa[["FUND ID", "fund_age", "cum_calls_before_age"]],
        on=["FUND ID", "fund_age"],
        how="left",
    )
    state_df = state_df.merge(total_called, on="FUND ID", how="left")
    state_df["cum_calls_before_age"] = state_df["cum_calls_before_age"].fillna(0.0)
    state_df = state_df[state_df["total_called"] > 0].copy()
    if state_df.empty:
        return {}

    state_df["nav_begin_multiple"] = state_df["nav_begin"] / state_df["total_called"]
    state_df["cumulative_called_pct"] = (
        state_df["cum_calls_before_age"] / state_df["total_called"]
    )
    state_df["cumulative_called_pct"] = state_df["cumulative_called_pct"].clip(0.0, 1.0)
    state_df["nav_begin_multiple"] = state_df["nav_begin_multiple"].clip(lower=0.0)

    state_pool_by_age: Dict[int, List[Tuple[float, float]]] = {}
    for _, row in state_df.iterrows():
        age = int(row["fund_age"])
        state_pool_by_age.setdefault(age, []).append(
            (float(row["nav_begin_multiple"]), float(row["cumulative_called_pct"]))
        )
    return state_pool_by_age


def sample_initial_fund_state_matrices(
    ages: np.ndarray,
    commitments: np.ndarray,
    state_pool_by_age: Dict[int, List[Tuple[float, float]]],
    num_simulations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample initial NAV and cumulative-called state for each fund/simulation."""
    import numpy as _np

    if not state_pool_by_age:
        raise ValueError("state_pool_by_age is empty")
    max_age = max(state_pool_by_age.keys())

    n_funds = len(ages)
    initial_navs = _np.zeros((num_simulations, n_funds), dtype=float)
    initial_cum_calls = _np.zeros((num_simulations, n_funds), dtype=float)

    trunc_ages = _np.clip(ages.astype(int), 1, max_age)
    for age_key in set(trunc_ages.tolist()):
        idx = _np.where(trunc_ages == age_key)[0]
        if idx.size == 0:
            continue
        pool = state_pool_by_age.get(age_key)
        if not pool:
            continue
        nav_mult = _np.array([p[0] for p in pool], dtype=float)
        cum_call_pct = _np.array([p[1] for p in pool], dtype=float)
        L = len(pool)
        ri = _np.random.randint(0, L, size=(num_simulations, idx.size))
        comm = commitments[idx][None, :]
        initial_navs[:, idx] = nav_mult[ri] * comm
        initial_cum_calls[:, idx] = _np.minimum(cum_call_pct[ri] * comm, comm)

    return initial_navs, initial_cum_calls


# ---------------------------------------------------------------------------
# Simulation engine – constant age, year-specific patterns
# ---------------------------------------------------------------------------

def run_simulation_constant_age_by_year(
    portfolio_df: pd.DataFrame,
    patterns_by_year: Dict[int, Dict[int, List[PatternLike]]],
    pattern_dict_fallback: Dict[int, List[PatternLike]],
    initial_state_pool_by_age: Dict[int, List[Tuple[float, float]]],
    scenario_years: Iterable[int],
    base_year_end: int,
    num_simulations: int = 500,
    return_fund_level: bool = False,
    min_bucket_size: int = MIN_BUCKET_SIZE,
    audit_simulation_number: Optional[int] = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate cash flows with recallable distribution mechanics.

    **Initialisation (existing funds):**
    Each existing fund is initialized directly from sampled historical
    fund-level states at the same age (NAV-at-beginning-of-age and
    cumulative called). LP purchases the portfolio at NAV and commits
    an equal amount as reserve.

    **Dynamic aging:**
    All funds age dynamically. Burn-in funds start at their age as of
    the first scenario year and advance by 1 each simulation year; new
    fund slots start at age 1. New fund slots are added each simulation
    year.

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

    # Pre-allocate for max portfolio size: initial + new funds over all years
    n_new_total = funds_per_vintage * n_scenario_years
    n_max = n_initial + n_new_total

    commitments_all = _np.zeros(n_max, dtype=float)
    commitments_all[:n_initial] = commitments_initial

    creation_years = _np.zeros(n_max, dtype=int)
    creation_years[:n_initial] = -1  # sentinel: burn-in fund

    # ---- Existing-fund initialization from sampled historical states ----
    initial_navs, initial_cum_calls = sample_initial_fund_state_matrices(
        burnin_ages, commitments_initial, initial_state_pool_by_age, num_simulations,
    )
    lp_nav_purchase = float(initial_navs.sum(axis=1).mean())
    print(f"Projected portfolio NAV (initial state): ${lp_nav_purchase:.2f}")
    print(f"  LP purchase price = ${lp_nav_purchase:.2f}")
    print(f"  LP reserve (50%) = ${lp_nav_purchase:.2f}")
    print(f"  LP total commitment = ${2 * lp_nav_purchase:.2f}")
    print(f"  Initial fund commitments = ${commitments_initial.sum():.2f}")
    print(f"  New commitment per year = ${funds_per_vintage * new_fund_commitment:.2f}")

    fund_level_records: List[pd.DataFrame] = []
    agg_records: List[Dict[str, object]] = []
    audit_records: List[Dict[str, object]] = []

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
        sampled_nav_begin_mat = _np.zeros((num_simulations, n_active), dtype=float)
        sampled_fund_id = _np.full((num_simulations, n_active), None, dtype=object)
        sampled_fund_name = _np.full((num_simulations, n_active), None, dtype=object)
        sampled_vintage_year = _np.full((num_simulations, n_active), None, dtype=object)
        sampled_txn_year = _np.full((num_simulations, n_active), None, dtype=object)

        for age_key in set(trunc_ages.tolist()):
            idx = _np.where(trunc_ages == age_key)[0]
            if idx.size == 0:
                continue
            records = patterns_by_year.get(year, {}).get(age_key)
            if not (records and len(records) >= min_bucket_size):
                records = pattern_dict_fallback.get(age_key, [])
            if not records:
                continue
            records = [_coerce_pattern_record(r, age=age_key, transaction_year=year) for r in records]
            L = len(records)
            ri = _np.random.randint(0, L, size=(num_simulations, idx.size))
            for col_pos, fund_idx in enumerate(idx):
                chosen = [records[r] for r in ri[:, col_pos]]
                call_mat[:, fund_idx] = _np.array([r["call_pct"] for r in chosen], dtype=float)
                dnav_mat[:, fund_idx] = _np.array([r["dist_nav_pct"] for r in chosen], dtype=float)
                navg_mat[:, fund_idx] = _np.array([r["nav_growth_pct"] for r in chosen], dtype=float)
                sampled_nav_begin_mat[:, fund_idx] = _np.array([r.get("nav_begin", 0.0) for r in chosen], dtype=float)
                sampled_fund_id[:, fund_idx] = [r.get("fund_id") for r in chosen]
                sampled_fund_name[:, fund_idx] = [r.get("fund_name") for r in chosen]
                sampled_vintage_year[:, fund_idx] = [r.get("vintage_year") for r in chosen]
                sampled_txn_year[:, fund_idx] = [r.get("transaction_year") for r in chosen]

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
        nav_begin = _np.where(new_fund_mask & (nav_begin <= 0), sampled_nav_begin_mat, nav_begin)
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

def run_simulation(
    portfolio_df: pd.DataFrame,
    pattern_dict: Dict[int, List[PatternLike]],
    initial_state_pool_by_age: Dict[int, List[Tuple[float, float]]],
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

    initial_navs, initial_cum_calls = sample_initial_fund_state_matrices(
        initial_ages, commitments, initial_state_pool_by_age, num_simulations,
    )
    lp_nav_purchase = float(initial_navs.sum(axis=1).mean())
    print(f"Projected portfolio NAV (initial state): ${lp_nav_purchase:.2f}")
    print(f"  LP purchase = ${lp_nav_purchase:.2f}, reserve = ${lp_nav_purchase:.2f}")

    fund_level_records: List[pd.DataFrame] = []
    agg_records: List[Dict[str, object]] = []

    total_commitment = float(commitments.sum())
    cumulative_calls_per_fund = initial_cum_calls.copy()
    cumulative_calls = cumulative_calls_per_fund.sum(axis=1).copy()
    cumulative_distributions = _np.zeros(num_simulations, dtype=float)

    lp_reserve = _np.full(num_simulations, lp_nav_purchase, dtype=float)
    recallable_balance = _np.zeros(num_simulations, dtype=float)
    underlying_nav = initial_navs.copy()

    annual_new_commitment = float(
        portfolio_df.groupby("vintage_year")["commitment"].sum().mean()
    )
    running_cum_commitment = total_commitment

    _median_call_curve_dyn: Dict[int, float] = {}
    for _ak, _pl in pattern_dict.items():
        _recs = [_coerce_pattern_record(p, age=_ak) for p in _pl]
        _median_call_curve_dyn[_ak] = float(_np.median([p["call_pct"] for p in _recs]))
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
        sampled_nav_begin_mat = _np.zeros((num_simulations, n_funds), dtype=float)

        for age_key in set(_np.minimum(_np.maximum(ages, 1), max_age)):
            idx = _np.where(_np.minimum(_np.maximum(ages, 1), max_age) == age_key)[0]
            if idx.size == 0:
                continue
            records = pattern_dict.get(age_key, [])
            if not records:
                continue
            records = [_coerce_pattern_record(r, age=age_key, transaction_year=year) for r in records]
            L = len(records)
            rand_idx = _np.random.randint(0, L, size=(num_simulations, idx.size))
            for col_pos, fund_idx in enumerate(idx):
                chosen = [records[r] for r in rand_idx[:, col_pos]]
                call_mat[:, fund_idx] = _np.array([r["call_pct"] for r in chosen], dtype=float)
                dist_nav_mat[:, fund_idx] = _np.array([r["dist_nav_pct"] for r in chosen], dtype=float)
                navg_mat[:, fund_idx] = _np.array([r["nav_growth_pct"] for r in chosen], dtype=float)
                sampled_nav_begin_mat[:, fund_idx] = _np.array([r.get("nav_begin", 0.0) for r in chosen], dtype=float)

        # --- Calls: commitment-based, capped at remaining commitment ---
        call_amounts = call_mat * commitments
        remaining = commitments - cumulative_calls_per_fund
        remaining = _np.maximum(remaining, 0.0)
        call_amounts = _np.minimum(call_amounts, remaining)
        cumulative_calls_per_fund += call_amounts

        underlying_nav_begin = _np.where(underlying_nav <= 0, sampled_nav_begin_mat, underlying_nav)
        underlying_nav = underlying_nav_begin.copy()
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


def random_choice(options: List[PatternLike]) -> PatternRecord:
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
    universe_path = r"./data/Preqin_Cashflow_export.xlsx"
    asset_class = "Venture Capital"
    n_vintages = 20
    n_per_vintage = 10
    start_year = 2006
    end_year = 2025
    commitment = 1.0
    simulations = 5000
    output_path = r"./output"
    audit_simulation_number = 0

    # ---- Load data ----
    universe_df = load_universe(universe_path)

    use_year_specific = True

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
    # Start from the year after the initial snapshot vintage window end.
    scenario_years = list(range(start_year + 1, end_year + 1))

    # ---- Run simulation ----
    if use_year_specific:
        patterns_by_year = compute_pattern_dict_by_year(universe_df, asset_class)
        pattern_dict_fallback = compute_pattern_dict(universe_df, asset_class)
        initial_state_pool_by_age = build_initial_state_pool_by_age(universe_df, asset_class)
        if not patterns_by_year:
            raise RuntimeError(f"No year-specific patterns for {asset_class!r}")
        fund_df, agg_df, audit_df = run_simulation_constant_age_by_year(
            portfolio_df=portfolio_df,
            patterns_by_year=patterns_by_year,
            pattern_dict_fallback=pattern_dict_fallback,
            initial_state_pool_by_age=initial_state_pool_by_age,
            scenario_years=scenario_years,
            base_year_end=end_year,
            num_simulations=simulations,
            return_fund_level=False,
            audit_simulation_number=audit_simulation_number,
        )
    else:
        pattern_dict = compute_pattern_dict(universe_df, asset_class)
        initial_state_pool_by_age = build_initial_state_pool_by_age(universe_df, asset_class)
        if not pattern_dict:
            raise RuntimeError(f"No cash-flow patterns for {asset_class!r}")
        fund_df, agg_df = run_simulation(
            portfolio_df=portfolio_df,
            pattern_dict=pattern_dict,
            initial_state_pool_by_age=initial_state_pool_by_age,
            scenario_years=scenario_years,
            num_simulations=simulations,
            return_fund_level=False,
        )
        audit_df = pd.DataFrame()

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
