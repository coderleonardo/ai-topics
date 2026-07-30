"""Synthetic reference/current datasets shared by all data-drift notebooks.

Continues the retail demand-forecasting scenario from ``docs/01-what-is-data-drift.md``:
a retailer's historical (``reference``) sales data vs. a more recent (``current``)
production batch collected after an online-sales-driving marketing campaign.

Every notebook under ``notebooks/`` calls :func:`make_reference_and_current` with the
same seed, so all four detection techniques are evaluated against the exact same
drift instance and can be compared directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REFERENCE_SIZE = 5000
CURRENT_SIZE = 2000

# Ground truth for what each column actually does between reference and current.
# Notebooks use this only to self-check / annotate results, never as an input to
# any detection method.
COLUMN_DRIFT_GROUND_TRUTH = {
    "channel": "drift",  # categorical share shift: in_store -> online
    "product_category": "drift",  # a brand-new category appears
    "basket_size": "drift",  # mean and variance shift
    "discount_pct": "borderline",  # small shift, easy to over-call at large N
    "unit_price": "none",  # control feature: distribution unchanged
    "customer_age": "data_quality",  # not drift: a handful of corrupted values
}


def _make_reference(rng: np.random.Generator) -> pd.DataFrame:
    n = REFERENCE_SIZE

    channel = rng.choice(["in_store", "online"], size=n, p=[0.85, 0.15])

    product_category = rng.choice(
        ["electronics", "apparel", "grocery", "home", "beauty"],
        size=n,
        p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )

    basket_size = rng.gamma(shape=9.0, scale=5.0, size=n)  # mean ~45
    discount_pct = rng.beta(2.0, 20.0, size=n)  # mean ~0.09
    unit_price = rng.lognormal(mean=3.0, sigma=0.4, size=n)  # mean ~$22
    customer_age = rng.normal(40.0, 12.0, size=n).clip(18, 90)

    return pd.DataFrame(
        {
            "channel": channel,
            "product_category": product_category,
            "basket_size": basket_size,
            "discount_pct": discount_pct,
            "unit_price": unit_price,
            "customer_age": customer_age,
        }
    )


def _make_current(rng: np.random.Generator) -> pd.DataFrame:
    n = CURRENT_SIZE

    # Channel drift: the marketing campaign pushes a lot more traffic online.
    channel = rng.choice(["in_store", "online"], size=n, p=[0.55, 0.45])

    # A new category shows up that didn't exist in the training window, and the
    # rest of the mix shifts a bit to make room for it.
    product_category = rng.choice(
        ["electronics", "apparel", "grocery", "home", "beauty", "online_exclusive"],
        size=n,
        p=[0.24, 0.22, 0.18, 0.14, 0.08, 0.14],
    )

    # Baskets get bigger and more variable online.
    basket_size = rng.gamma(shape=7.0, scale=8.4, size=n)  # mean ~59

    # Discounts creep up slightly - a small, borderline shift.
    discount_pct = rng.beta(2.5, 20.0, size=n)  # mean ~0.111

    # Unit price is a control feature: same distribution as reference.
    unit_price = rng.lognormal(mean=3.0, sigma=0.4, size=n)

    customer_age = rng.normal(40.0, 12.0, size=n).clip(18, 90)
    # Inject a small batch of corrupted values - a data pipeline bug, not drift.
    n_bad = max(1, int(round(0.0125 * n)))
    bad_idx = rng.choice(n, size=n_bad, replace=False)
    customer_age[bad_idx] = rng.choice([-1, -5, -999], size=n_bad)

    return pd.DataFrame(
        {
            "channel": channel,
            "product_category": product_category,
            "basket_size": basket_size,
            "discount_pct": discount_pct,
            "unit_price": unit_price,
            "customer_age": customer_age,
        }
    )


def make_reference_and_current(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (reference, current) DataFrames for the retail drift scenario.

    ``reference`` mimics the historical data a model was trained on; ``current``
    mimics a later production batch. Both share the same schema. See
    ``COLUMN_DRIFT_GROUND_TRUTH`` for what each column is meant to demonstrate.
    """
    rng = np.random.default_rng(seed)
    reference = _make_reference(rng)
    current = _make_current(rng)
    return reference, current


NUMERICAL_COLUMNS = ["basket_size", "discount_pct", "unit_price", "customer_age"]
CATEGORICAL_COLUMNS = ["channel", "product_category"]
