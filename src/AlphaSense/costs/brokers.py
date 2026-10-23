"""
Broker fee presets (indicative only).

These values are for simulation purposes only and may not reflect current
broker pricing. Always verify against the broker's official fee schedule.
"""

from __future__ import annotations

from typing import Any


BROKER_PRESETS: dict[str, dict[str, Any]] = {
    "xtb": {
        "label": "XTB (real stocks/ETFs - indicative)",
        "fee_pct": 0.0,
        "fee_fixed": 0.0,
        "slippage_bps": 1.0,
        "notes": (
            "0% commission up to €100k/month nominal (0.2% after, min fee applies). "
            "0.5% FX fee if currency differs from account base. "
            "Market spread/slippage not published by XTB - modelled conservatively at 1.0 bps. "
            "Indicative only, plan/region dependent."
        ),
    },
}


def get_broker_preset(name: str | None) -> dict[str, Any] | None:
    if name is None:
        return None
    key = str(name).strip().lower()
    if key not in BROKER_PRESETS:
        raise ValueError(
            f"Unknown broker preset '{name}'. Available presets: {sorted(BROKER_PRESETS.keys())}"
        )
    return BROKER_PRESETS[key]


def resolve_broker_preset(
    preset_name: str | None,
    fee_pct: float | None,
    fee_fixed: float | None,
    slippage_bps: float | None,
) -> tuple[float, float, float, str | None]:
    """
    Resolve effective costs with precedence: explicit > preset > defaults.
    Returns (fee_pct_resolved, fee_fixed_resolved, slippage_bps_resolved, preset_label)
    """
    preset = get_broker_preset(preset_name)
    resolved_fee_pct = 0.0 if fee_pct is None else float(fee_pct)
    resolved_fee_fixed = 0.0 if fee_fixed is None else float(fee_fixed)
    resolved_slippage_bps = 0.0 if slippage_bps is None else float(slippage_bps)

    if preset is not None:
        if fee_pct is None and preset.get("fee_pct") is not None:
            resolved_fee_pct = float(preset["fee_pct"])
        if fee_fixed is None and preset.get("fee_fixed") is not None:
            resolved_fee_fixed = float(preset["fee_fixed"])
        if slippage_bps is None and preset.get("slippage_bps") is not None:
            resolved_slippage_bps = float(preset["slippage_bps"])
        return resolved_fee_pct, resolved_fee_fixed, resolved_slippage_bps, preset.get("label")

    return resolved_fee_pct, resolved_fee_fixed, resolved_slippage_bps, None


# IBKR added after manual verification of official sources
BROKER_PRESETS["ibkr"] = {
    "label": "Interactive Brokers (IBKR Lite/Pro - indicative)",
    "fee_pct": 0.0,
    "fee_fixed": 0.0,
    "slippage_bps": 1.0,
    "notes": (
        "IBKR Lite: $0 commission for US exchange-listed stocks/ETFs for eligible US residents (with conditions). "
        "IBKR Pro: tiered/fixed commissions apply. "
        "Regulatory/exchange/clearing fees may apply separately. "
        "Market spread/slippage not published as fixed bps - modelled conservatively at 1.0 bps. "
        "Indicative only, plan/region dependent."
    ),
}


# IBKR Pro - France/EU (added after manual verification)
BROKER_PRESETS["ibkr-us-tiered"] = {
    "label": "Interactive Brokers Pro - US stocks (Tiered, France/EU indicative)",
    "fee_pct": 0.0005,
    "fee_fixed": 0.0,
    "slippage_bps": 1.0,
    "notes": (
        "France/EU residents use IBKR Pro (IBKR Lite not eligible). "
        "US stocks IBKR Pro Tiered: 0.0005-0.0035 USD/share by monthly volume (min 0.35 USD/order, 1% cap may apply). "
        "Regulatory/exchange/clearing fees may apply separately. "
        "Market spread/slippage not published - modelled conservatively at 1.0 bps. "
        "Indicative only, plan/region dependent."
    ),
}

BROKER_PRESETS["ibkr-eu-tiered"] = {
    "label": "Interactive Brokers Pro - European stocks (Tiered, France/EU indicative)",
    "fee_pct": 0.0005,
    "fee_fixed": 0.0,
    "slippage_bps": 1.0,
    "notes": (
        "France/EU residents use IBKR Pro (IBKR Lite not eligible). "
        "European stocks IBKR Pro Tiered (e.g. France/Euronext): ~0.05% down to ~0.015% of trade value by monthly trade value (min ~1.25 EUR/order, max ~29 EUR/order may apply). "
        "Exchange/clearing/regulatory fees may apply separately (often passed through). "
        "Market spread/slippage not published - modelled conservatively at 1.0 bps. "
        "Indicative only, plan/region dependent."
    ),
}

# IBKR Pro - France/EU (added after manual verification)
BROKER_PRESETS["ibkr-us-pro"] = {
    "label": "Interactive Brokers Pro - US stocks (mid-tier Tiered, France/EU indicative)",
    "fee_pct": 0.0002,
    "fee_fixed": 0.0,
    "slippage_bps": 1.0,
    "notes": (
        "France/EU residents use IBKR Pro (IBKR Lite not eligible). "
        "US stocks IBKR Pro Tiered mid-tier: ~0.0020 USD/share (300k-3M shares/month). "
        "Modelled as ~0.02% of notional (varies by share price). "
        "Regulatory/exchange/clearing fees may apply separately. "
        "Market spread/slippage not published - modelled conservatively at 1.0 bps. "
        "Indicative only, plan/region dependent."
    ),
}

BROKER_PRESETS["ibkr-eu-fixed"] = {
    "label": "Interactive Brokers Pro - European stocks (Fixed SmartRouting, France/EU indicative)",
    "fee_pct": 0.0005,
    "fee_fixed": 0.0,
    "slippage_bps": 1.0,
    "notes": (
        "France/EU residents use IBKR Pro (IBKR Lite not eligible). "
        "European stocks IBKR Pro Fixed - IB SmartRoutingSM: 0.05% of trade value (min ~1.25 EUR/order, max ~29 EUR/order may apply). "
        "Exchange/clearing/regulatory fees may apply separately (often passed through). "
        "Market spread/slippage not published - modelled conservatively at 1.0 bps. "
        "Indicative only, plan/region dependent."
    ),
}


# DEGIRO (added after manual verification)
BROKER_PRESETS["degiro"] = {
    "label": "DEGIRO (France/EU - indicative)",
    "fee_pct": 0.0,
    "fee_fixed": 2.0,
    "slippage_bps": 1.5,
    "notes": (
        "France/EU residents. Euronext Paris: ~1€ brokerage + ~1€ handling/management fee per order (total ~2€). "
        "Other European markets higher (~3.90€ + 1€). US markets ~1€ + 1€. "
        "FX fee ~0.25% if currency differs. Connectivity fees may apply for non-domestic markets. "
        "Modelled conservatively as fee_fixed=2.00€ per fill for Euronext France-like orders. "
        "Indicative only, plan/market dependent."
    ),
}


# Alpaca (added after manual verification)
BROKER_PRESETS["alpaca"] = {
    "label": "Alpaca (US equities/ETFs - commission-free, indicative)",
    "fee_pct": 0.0,
    "fee_fixed": 0.0,
    "slippage_bps": 1.0,
    "notes": (
        "Commission-free for most US exchange-listed stocks/ETFs on standard plan. "
        "Regulatory fees (SEC/FINRA TAF/CAT) may apply as pass-through and are typically small. "
        "Market spread/slippage not published - modelled conservatively at 1.0 bps. "
        "Indicative only, plan/region dependent."
    ),
}
