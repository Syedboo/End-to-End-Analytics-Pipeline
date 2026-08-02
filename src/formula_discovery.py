"""Candidate formula discovery for derived printing-job fields."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import safe_divide


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _score_formula(name: str, target: pd.Series, estimate: pd.Series, formula: str) -> dict[str, object]:
    diff = (estimate - target).replace([np.inf, -np.inf], np.nan).dropna()
    if diff.empty:
        return {
            "Field": name,
            "Candidate Formula": formula,
            "Rows Compared": 0,
            "Exact Match Rate": np.nan,
            "Mean Absolute Error": np.nan,
            "Max Absolute Error": np.nan,
            "Conclusion": "Not enough data",
        }
    exact = diff.abs().lt(1e-6).mean()
    conclusion = "Exact inferred formula" if exact > 0.999 else "Approximate or conditional relationship"
    return {
        "Field": name,
        "Candidate Formula": formula,
        "Rows Compared": int(len(diff)),
        "Exact Match Rate": float(exact),
        "Mean Absolute Error": float(diff.abs().mean()),
        "Max Absolute Error": float(diff.abs().max()),
        "Conclusion": conclusion,
    }


def discover_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate high-probability business formulas for derived columns."""

    sell_price = _num(df, "Sell Price")
    purchases = _num(df, "Purchases")
    rebate = _num(df, "Rebate")
    va_amount = _num(df, "VA Amount")
    impressions = _num(df, "Impressions")
    labour = _num(df, "Labour")
    labmup = _num(df, "labmup")
    manadj = _num(df, "manadj")
    mupnett = _num(df, "mupnett")

    candidates = [
        ("VA Amount", va_amount, sell_price - purchases - rebate, "Sell Price - Purchases - Rebate"),
        ("Sell Price", sell_price, purchases + rebate + va_amount, "Purchases + Rebate + VA Amount"),
        ("VA%", _num(df, "VA%"), safe_divide(va_amount, sell_price), "VA Amount / Sell Price"),
        ("VA/K", _num(df, "VA/K"), safe_divide(va_amount, impressions / 1000), "VA Amount / (Impressions / 1000)"),
        ("mupnett", mupnett, labmup + manadj, "labmup + manadj"),
        ("manadj", manadj, mupnett - labmup, "mupnett - labmup"),
        ("Mup%", _num(df, "Mup%"), safe_divide(mupnett, labour), "mupnett / Labour"),
        ("AmtInv", _num(df, "AmtInv"), sell_price, "Usually Sell Price, but invoice timing/status can differ"),
        ("labmup", labmup, labour * safe_divide(labmup, labour), "No stable closed-form source found; appears separate from manual adjustment"),
    ]
    return pd.DataFrame(
        [_score_formula(name, target, estimate, formula) for name, target, estimate, formula in candidates]
    ).sort_values(["Exact Match Rate", "Rows Compared"], ascending=[False, False])
