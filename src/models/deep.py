"""Global trained deep forecaster with a zero-inflation-aware likelihood (Stage 2).

One transformer (PatchTST) trained over all series at once, direct multi-horizon,
with a negative-binomial DistributionLoss for zero-inflated count demand. Trains
from scratch inside fit_predict on the harness-provided train slice, so every
backtest origin gets its own leak-free fit — the same contract as global_lgbm.

Point forecast = the distribution MEAN. neuralforecast returns the sample mean of
the fitted distribution in the base (model-alias) column; the median is a SEPARATE
column ('<alias>-median'). Verified empirically against neuralforecast 3.2.1 — see
tests/test_stage2_deep.py — precisely because the Chronos-Bolt 'mean'==median trap
(cycle 4, run 524453e7d8) showed third-party point-readout semantics must never be
assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES_ID

# Fixed model knobs (memo 2026-08-29-cycle6). No tuning surface: any change here
# is a new experiment/memo. learning_rate 1e-3 is neuralforecast's tutorial default
# paired with the reduced max_steps=1500 (the library constructor default 1e-4 is
# tuned for its 5000-step default); this pairing is untuned, not shopped.
DEFAULTS = dict(
    input_size=64, patch_len=8, stride=8, max_steps=1500,
    hidden_size=128, n_heads=16, batch_size=256, windows_batch_size=1024,
    learning_rate=1e-3, scaler_type="robust", random_seed=7,
)


class GlobalPatchTST:
    name = "patchtst"

    def __init__(self, freq: str = "W-MON", accelerator: str = "cpu",
                 distribution: str = "NegativeBinomial", num_samples: int = 1000,
                 model_params: dict | None = None):
        self.freq = freq
        self.accelerator = accelerator
        self.distribution = distribution
        self.num_samples = num_samples  # MC draws behind the mean readout; pinned for repro
        self.model_params = {**DEFAULTS, **(model_params or {})}

    def fit_predict(self, train, future_ds):
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import PatchTST

        torch.manual_seed(int(self.model_params["random_seed"]))

        h = len(future_ds)
        df = train[[SERIES_ID, "ds", "y"]].rename(columns={SERIES_ID: "unique_id"})

        model = PatchTST(
            h=h,
            loss=DistributionLoss(self.distribution, num_samples=self.num_samples),
            start_padding_enabled=True,   # every series gets a forecast (contract)
            enable_progress_bar=False,
            logger=False,
            accelerator=self.accelerator,
            **self.model_params,
        )
        nf = NeuralForecast(models=[model], freq=self.freq)
        nf.fit(df)
        fc = nf.predict()

        alias = getattr(model, "alias", None) or "PatchTST"
        if alias not in fc.columns:
            # base point column is the only forecast column with no '-' suffix
            cand = [c for c in fc.columns
                    if c not in ("unique_id", "ds") and "-" not in c]
            if not cand:
                raise RuntimeError(f"no point column in predict output {list(fc.columns)}")
            alias = cand[0]

        out = (fc.rename(columns={"unique_id": SERIES_ID, alias: "yhat"})
                 [[SERIES_ID, "ds", "yhat"]].copy())
        out["ds"] = pd.to_datetime(out["ds"])
        out["yhat"] = np.clip(out["yhat"].to_numpy(dtype="float64"), 0.0, None)
        return out


def _to_contract(fc, model, default_alias: str):
    """Map a neuralforecast predict() frame to [SERIES_ID, ds, yhat].

    Same logic the PatchTST/NHITS wrappers inline (the base point column is the
    distribution MEAN; '<alias>-*' quantile columns are ignored). Shared only by
    the Cycle-10 zoo classes below; the existing wrappers keep their inline copy
    so their behaviour is provably unchanged.
    """
    alias = getattr(model, "alias", None) or default_alias
    if alias not in fc.columns:
        cand = [c for c in fc.columns
                if c not in ("unique_id", "ds") and "-" not in c]
        if not cand:
            raise RuntimeError(f"no point column in {list(fc.columns)}")
        alias = cand[0]
    out = (fc.rename(columns={"unique_id": SERIES_ID, alias: "yhat"})
             [[SERIES_ID, "ds", "yhat"]].copy())
    out["ds"] = pd.to_datetime(out["ds"])
    out["yhat"] = np.clip(out["yhat"].to_numpy(dtype="float64"), 0.0, None)
    return out


# Per-architecture untuned defaults (Cycle 10 zoo). Same generic training budget
# as the existing deep nets (input_size 64, max_steps 1500, lr 1e-3, robust
# scaler, seed 7) so the comparison is apples-to-apples; only the
# architecture-specific knobs differ. No tuning surface.
DEEPAR_DEFAULTS = dict(
    input_size=64, max_steps=1500, batch_size=256,
    learning_rate=1e-3, scaler_type="robust", random_seed=7,
    lstm_n_layers=2, lstm_hidden_size=128, lstm_dropout=0.1,
)
TFT_DEFAULTS = dict(
    input_size=64, max_steps=1500, batch_size=256, windows_batch_size=1024,
    learning_rate=1e-3, scaler_type="robust", random_seed=7,
    hidden_size=128, n_head=4,
)


class GlobalDeepAR:
    """Global DeepAR (Amazon's autoregressive probabilistic RNN) with a count
    DistributionLoss. No covariates. Point forecast = distribution mean, same
    contract as the other deep nets. Tier-2 comparison."""
    name = "deepar"

    def __init__(self, freq: str = "W-MON", accelerator: str = "cpu",
                 distribution: str = "Poisson", num_samples: int = 1000,
                 model_params: dict | None = None):
        self.freq = freq
        self.accelerator = accelerator
        self.distribution = distribution
        self.num_samples = num_samples
        self.model_params = {**DEEPAR_DEFAULTS, **(model_params or {})}

    def fit_predict(self, train, future_ds):
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import DeepAR

        torch.manual_seed(int(self.model_params["random_seed"]))
        h = len(future_ds)
        df = train[[SERIES_ID, "ds", "y"]].rename(columns={SERIES_ID: "unique_id"})

        model = DeepAR(
            h=h,
            loss=DistributionLoss(self.distribution, num_samples=self.num_samples),
            enable_progress_bar=False, logger=False,
            accelerator=self.accelerator, **self.model_params,
        )
        nf = NeuralForecast(models=[model], freq=self.freq)
        nf.fit(df)
        fc = nf.predict()
        return _to_contract(fc, model, "DeepAR")


class GlobalTFT:
    """Global Temporal Fusion Transformer with a count DistributionLoss. No
    covariates (matched to the other deep-net comparisons). Point forecast =
    distribution mean. Tier-2 comparison."""
    name = "tft"

    def __init__(self, freq: str = "W-MON", accelerator: str = "cpu",
                 distribution: str = "Poisson", num_samples: int = 1000,
                 model_params: dict | None = None):
        self.freq = freq
        self.accelerator = accelerator
        self.distribution = distribution
        self.num_samples = num_samples
        self.model_params = {**TFT_DEFAULTS, **(model_params or {})}

    def fit_predict(self, train, future_ds):
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import TFT

        torch.manual_seed(int(self.model_params["random_seed"]))
        h = len(future_ds)
        df = train[[SERIES_ID, "ds", "y"]].rename(columns={SERIES_ID: "unique_id"})

        model = TFT(
            h=h,
            loss=DistributionLoss(self.distribution, num_samples=self.num_samples),
            start_padding_enabled=True,
            enable_progress_bar=False, logger=False,
            accelerator=self.accelerator, **self.model_params,
        )
        nf = NeuralForecast(models=[model], freq=self.freq)
        nf.fit(df)
        fc = nf.predict()
        return _to_contract(fc, model, "TFT")


# ---------------------------------------------------------------------------
# N-HiTS with optional exogenous covariates (Cycle 7). PatchTST is univariate
# only (EXOGENOUS_* all False), so the covariate experiment uses N-HiTS (the
# ROADMAP-registered backup). Covariates are built LEAK-SAFELY from `train`
# only: futr are deterministic for the horizon (woy/month/age), hist are
# past-only (price ffill, price_chg4, wsls) and used from the input window,
# stat are per-series constants (client/warehouse).
# ---------------------------------------------------------------------------

# All covariate columns the wrapper knows how to build, by exog role.
FUTR_COVARS = ("woy", "month", "age")          # deterministic for future weeks
HIST_COVARS = ("price", "price_chg4", "wsls")  # past-only (input window)
STAT_COVARS = ("client", "warehouse")          # static per series
# lowercase stat name -> capitalized source column in the panel
STAT_SOURCE = {"client": "Client", "warehouse": "Warehouse"}


def _first_sale_ds(train: pd.DataFrame) -> pd.Series:
    """First week with a positive sale, per series (NaT if never sold in train)."""
    sold = train[train["y"] > 0]
    return sold.groupby(SERIES_ID)["ds"].min()


def _add_futr_covars(frame: pd.DataFrame, cols, first_sale: pd.Series) -> pd.DataFrame:
    """Add deterministic future-known covariates to any frame with [SERIES_ID, ds]."""
    if "woy" in cols:
        frame["woy"] = frame["ds"].dt.isocalendar().week.astype("float32")
    if "month" in cols:
        frame["month"] = frame["ds"].dt.month.astype("float32")
    if "age" in cols:
        fs = frame[SERIES_ID].map(first_sale)
        weeks = (frame["ds"] - fs).dt.days // 7
        # -1 before first sale / for never-sold series (matches features.py convention)
        frame["age"] = weeks.where(fs.notna() & (weeks >= 0), -1).astype("float32")
    return frame


def _add_hist_covars(long: pd.DataFrame, cols) -> pd.DataFrame:
    """Add past-only covariates to the training long frame (sorted by series, ds)."""
    long = long.sort_values([SERIES_ID, "ds"])
    if "price" in cols or "price_chg4" in cols:
        pff = long.groupby(SERIES_ID)["Price"].ffill().fillna(0.0)
        if "price" in cols:
            long["price"] = pff.astype("float32")
        if "price_chg4" in cols:
            prev4 = pff.groupby(long[SERIES_ID]).shift(4)
            long["price_chg4"] = (pff / prev4).replace([np.inf, -np.inf], np.nan) \
                                              .fillna(1.0).astype("float32")
    if "wsls" in cols:
        # weeks since last positive sale, computed from past y only.
        # Sentinel -1 before a series' first sale (deliberately NOT features.py's
        # 999: a small sentinel scales gracefully for the neural net; 999 would
        # be a huge outlier under the robust scaler). Both nhits configs use this
        # identical builder, so the OFF/ON covariate delta stays apples-to-apples.
        def _wsls(y):
            out = np.empty(len(y), dtype="float32")
            since = -1
            for i, v in enumerate(y.to_numpy()):
                since = 0 if v > 0 else (since + 1 if since >= 0 else -1)
                out[i] = since
            return pd.Series(out, index=y.index)
        long["wsls"] = long.groupby(SERIES_ID)["y"].transform(_wsls)
    return long


class GlobalNHITS:
    """Global N-HiTS with a count DistributionLoss and optional covariates.

    Covariate lists select from FUTR/HIST/STAT_COVARS; empty lists => a pure
    univariate model (the covariates-OFF baseline). Point forecast = the
    distribution mean (base alias column), same contract as GlobalPatchTST.
    """
    name = "nhits"

    def __init__(self, freq: str = "W-MON", accelerator: str = "cpu",
                 distribution: str = "Poisson", num_samples: int = 1000,
                 futr_exog: list | None = None, hist_exog: list | None = None,
                 stat_exog: list | None = None, model_params: dict | None = None):
        self.freq = freq
        self.accelerator = accelerator
        self.distribution = distribution
        self.num_samples = num_samples
        self.futr_exog = list(futr_exog or [])
        self.hist_exog = list(hist_exog or [])
        self.stat_exog = list(stat_exog or [])
        # reuse the shared default knobs except PatchTST-specific architecture ones
        # (NHITS has no patch_len/stride/n_heads/hidden_size)
        self.model_params = {k: v for k, v in DEFAULTS.items()
                             if k not in ("patch_len", "stride", "n_heads",
                                          "hidden_size")}
        self.model_params.update(model_params or {})

    def _build_frames(self, train, future_ds):
        first_sale = _first_sale_ds(train)
        keep = [SERIES_ID, "ds", "y"] + [c for c in ("Price",) if c in train.columns]
        long = train[keep].copy()
        long = _add_hist_covars(long, self.hist_exog)
        long = _add_futr_covars(long, self.futr_exog, first_sale)
        long = long.rename(columns={SERIES_ID: "unique_id"}).drop(columns=["Price"],
                                                                   errors="ignore")

        static_df = None
        if self.stat_exog:
            src = {c: STAT_SOURCE.get(c, c) for c in self.stat_exog}
            avail = {c: s for c, s in src.items() if s in train.columns}
            g = train.groupby(SERIES_ID)[list(avail.values())].first().reset_index()
            g = g.rename(columns={SERIES_ID: "unique_id",
                                  **{s: c for c, s in avail.items()}})
            for c in avail:
                g[c] = g[c].astype("float32")
            static_df = g

        futr_df = None
        if self.futr_exog:
            ids = train[SERIES_ID].unique()
            futr_df = pd.MultiIndex.from_product(
                [ids, pd.to_datetime(future_ds)], names=[SERIES_ID, "ds"]
            ).to_frame(index=False)
            futr_df = _add_futr_covars(futr_df, self.futr_exog, first_sale)
            futr_df = futr_df.rename(columns={SERIES_ID: "unique_id"})
        return long, static_df, futr_df

    def fit_predict(self, train, future_ds):
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import NHITS

        torch.manual_seed(int(self.model_params["random_seed"]))
        h = len(future_ds)
        long, static_df, futr_df = self._build_frames(train, future_ds)

        model = NHITS(
            h=h,
            loss=DistributionLoss(self.distribution, num_samples=self.num_samples),
            futr_exog_list=self.futr_exog or None,
            hist_exog_list=self.hist_exog or None,
            stat_exog_list=self.stat_exog or None,
            start_padding_enabled=True,
            enable_progress_bar=False, logger=False, accelerator=self.accelerator,
            **self.model_params,
        )
        nf = NeuralForecast(models=[model], freq=self.freq)
        nf.fit(long, static_df=static_df)
        fc = nf.predict(futr_df=futr_df) if futr_df is not None else nf.predict()

        alias = getattr(model, "alias", None) or "NHITS"
        if alias not in fc.columns:
            cand = [c for c in fc.columns
                    if c not in ("unique_id", "ds") and "-" not in c]
            if not cand:
                raise RuntimeError(f"no point column in {list(fc.columns)}")
            alias = cand[0]
        out = (fc.rename(columns={"unique_id": SERIES_ID, alias: "yhat"})
                 [[SERIES_ID, "ds", "yhat"]].copy())
        out["ds"] = pd.to_datetime(out["ds"])
        out["yhat"] = np.clip(out["yhat"].to_numpy(dtype="float64"), 0.0, None)
        return out
