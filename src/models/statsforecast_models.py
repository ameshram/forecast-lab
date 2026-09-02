"""Classical / intermittent-demand models via Nixtla statsforecast (Tier 0).

One config-selected wrapper over statsforecast's model zoo, so Croston, SBA,
TSB, AutoETS, AutoTheta, AutoARIMA (etc.) are each a pure config diff. The
intermittent-demand methods (Croston/SBA/TSB) are the canonical yardstick this
project was missing.

Contract (as all models): fit_predict(train, future_ds) -> [SERIES_ID, ds, yhat],
one row per series per future week, full coverage. Heavy import lives inside
fit_predict so the registry stays importable without statsforecast installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES_ID


class StatsForecastModel:
    """params:
    model  -- statsforecast model class name (e.g. "CrostonClassic", "TSB").
    params -- kwargs for that model's constructor (e.g. TSB alpha_d/alpha_p,
              AutoETS season_length).
    freq   -- pandas offset alias of the series grid (weekly Mondays here).
    n_jobs -- statsforecast parallelism over series (1 = deterministic/safe).
    clip_negative -- floor forecasts at 0 (ETS/Theta/ARIMA can go negative;
                     demand cannot).
    """
    name = "statsforecast"

    _MODELS = ("CrostonClassic", "CrostonSBA", "CrostonOptimized", "TSB",
               "ADIDA", "IMAPA", "AutoETS", "AutoTheta", "AutoARIMA",
               "AutoCES", "SeasonalNaive", "HistoricAverage")

    def __init__(self, model: str, params: dict | None = None,
                 freq: str = "W-MON", n_jobs: int = 1,
                 clip_negative: bool = True):
        if model not in self._MODELS:
            raise ValueError(f"Unknown statsforecast model '{model}'. "
                             f"Known: {self._MODELS}")
        self.model = model
        self.params = params or {}
        self.freq = freq
        self.n_jobs = n_jobs
        self.clip_negative = clip_negative

    def _make_model(self):
        import statsforecast.models as sfm
        return getattr(sfm, self.model)(**self.params)

    def fit_predict(self, train, future_ds):
        from statsforecast import StatsForecast

        h = len(future_ds)
        long = (train[[SERIES_ID, "ds", "y"]]
                .rename(columns={SERIES_ID: "unique_id"}))

        sf = StatsForecast(models=[self._make_model()], freq=self.freq,
                           n_jobs=self.n_jobs)
        fc = sf.forecast(df=long, h=h)

        # statsforecast names the forecast column after the model alias
        val_col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        fc = fc.rename(columns={"unique_id": SERIES_ID, val_col: "yhat"})
        fc["yhat"] = fc["yhat"].astype(float).fillna(0.0)
        if self.clip_negative:
            fc["yhat"] = np.maximum(fc["yhat"].to_numpy(), 0.0)

        # enforce the harness's full-coverage contract: every series x every
        # future week gets a row. Dedupe guard first (a duplicated (series,ds)
        # would fan out the scoring join in src/backtest.py). Any series the
        # library drops (e.g. degenerate all-zero) falls back to 0.
        fc = fc.drop_duplicates([SERIES_ID, "ds"])
        series = train[SERIES_ID].unique()
        grid = pd.DataFrame({
            SERIES_ID: np.repeat(series, h),
            "ds": np.tile(np.array(future_ds, dtype="datetime64[ns]"), len(series)),
        })
        out = grid.merge(fc[[SERIES_ID, "ds", "yhat"]], on=[SERIES_ID, "ds"],
                         how="left")
        out["yhat"] = out["yhat"].fillna(0.0)
        return out
