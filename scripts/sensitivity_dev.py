"""Dev-only sensitivity analyses for docs/model_validation.md §7 (standalone; reads dev data only)."""
import numpy as np, pandas as pd
from volrisk import config as C, io
from volrisk.risk import var_es
from volrisk.risk.backtests import kupiec, traffic_light_zones, zone
from volrisk.models.ml import LGBM
from volrisk.targets import build_targets
from volrisk.evaluation.leaderboard import losses_frame, common_dates

daily = io.load_daily()                           # dev only
fc = pd.read_parquet(C.RESULTS / "forecasts.parquet")
risk = pd.read_parquet(C.RESULTS / "risk.parquet")
risk = risk[risk["split"] == "dev"]
rows = []
for a in C.ASSETS:
    d = daily[daily["asset"] == a].reset_index(drop=True)
    r = var_es.session_returns(d, a)
    win = risk[(risk["asset"] == a) & (risk["model"] == "COMBO+FHS")]["date"]
    dates = pd.DatetimeIndex(win)
    s2 = var_es.sigma2_by_target(d, fc[(fc["horizon"] == "1d") & (fc["asset"] == a)], "COMBO", a)
    variants = {
        "COMBO+FHS pool 1000 (frozen)": var_es.fhs(r, s2, pool=1000, min_pool=500),
        "COMBO+FHS pool 500": var_es.fhs(r, s2, pool=500, min_pool=500),
        "COMBO+Normal": var_es.normal_from_sigma2(s2),
        "RiskMetrics λ=0.94 (frozen)": var_es.riskmetrics(r, lam=0.94),
        "RiskMetrics λ=0.97": var_es.riskmetrics(r, lam=0.97),
    }
    for name, m in variants.items():
        m = m.reindex(dates).dropna(subset=["var99"])
        rr = r.reindex(m.index)
        hits99 = (-rr > m["var99"]).astype(int)
        k = kupiec(hits99.to_numpy(), 0.01)
        rows.append({"asset": a, "variant": name, "T": len(m), "breaches_99": int(hits99.sum()),
                     "p_binom": k["p_binom"], "zone_99": zone(int(hits99.sum()), len(m), 0.01),
                     "breaches_975": int((-rr > m["var975"]).sum())})
risk_sens = pd.DataFrame(rows)
print(risk_sens.round(3).to_string(index=False))

# LightGBM num_leaves 7 vs 15: QLIKE ratio vs HAR on the headline window (common dates with HAR)
out = []
for a in C.ASSETS:
    d = daily[daily["asset"] == a].reset_index(drop=True)
    for h in C.HORIZONS:
        tg = build_targets(d, a, h, C.dev_end())
        f7 = LGBM(params={"num_leaves": 7}).forecast(d, tg, a, h).assign(model="LGBM-7")
        base = fc[(fc["asset"] == a) & (fc["horizon"] == h) & fc["model"].isin(["HAR", "LGBM"])]
        L = losses_frame(pd.concat([base, f7]), tg.assign(asset=a, horizon=h))
        L = L[(L["split"] == "dev") & (L["origin"] >= pd.Timestamp(C.dev_eval_start()))]
        w = common_dates(L, ["HAR", "LGBM", "LGBM-7"])
        out.append({"asset": a, "horizon": h, "LGBM (15 leaves, frozen)": w["LGBM"].mean() / w["HAR"].mean(),
                    "LGBM (7 leaves)": w["LGBM-7"].mean() / w["HAR"].mean(), "n": len(w)})
lgbm_sens = pd.DataFrame(out)
print(lgbm_sens.round(3).to_string(index=False))
risk_sens.to_csv(C.TABLES / "sensitivity_risk_dev.csv", index=False)
lgbm_sens.to_csv(C.TABLES / "sensitivity_lgbm_dev.csv", index=False)
