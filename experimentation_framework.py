"""
Project 2: Experimentation & Causal-Impact Framework for Digital-Bank Product Growth
------------------------------------------------------------------------------------
Part A - Randomised A/B test of an in-app savings-activation nudge:
         power/sample sizing, SRM check, A/A calibration, CUPED, guardrails,
         Holm-corrected segment effects, unit economics.
Part B - When randomisation is NOT possible (self-selected feature opt-in):
         propensity-score matching + difference-in-differences, event-study
         parallel-trend diagnostics, and a scenario where DiD must NOT be used.

ALL DATA IS SYNTHETIC. True effects are known by construction so that estimators
can be validated against ground truth. Economic parameters are assumptions.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
sig = lambda x: 1 / (1 + np.exp(-x))
OUT = {}

# =========================================================================================
# PART A - RANDOMISED EXPERIMENT
# =========================================================================================
def n_per_arm(p1, p2, alpha=0.05, power=0.80):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pb = (p1 + p2) / 2
    return int(np.ceil((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2))

BASE, MDE = 0.08, 0.01
n_plan = n_per_arm(BASE, BASE + MDE)
OUT["A_design"] = {"primary_metric": "30-day savings-pot activation", "baseline": BASE, "MDE_abs": MDE,
                   "alpha": 0.05, "power": 0.80, "n_per_arm_required": n_plan}

# --- simulate 40K eligible customers, 50/50 randomisation --------------------------------
N = 40_000
age = np.clip(rng.normal(31, 9, N), 18, 65)
salary = rng.random(N) < 0.45
tenure_m = rng.integers(1, 25, N)
pre_bal = rng.lognormal(5.6 + 0.7 * salary, 0.9, N)                      # pre-period avg balance (GBP)
treat = rng.random(N) < 0.5
# heterogeneous true effect on activation (log-odds): larger for salary customers and under-25s
lo_base = np.log(BASE / (1 - BASE)) - 0.12 + 0.25 * salary + 0.15 * (np.log(pre_bal) - 6.0)
lift = 0.06 + 0.12 * salary + 0.14 * (age < 25)
p_act = sig(lo_base + treat * lift)
act = rng.random(N) < p_act
# 90-day balance (GBP): persistent with pre balance (-> CUPED), + incremental deposits for activators in treatment
post_bal = 0.85 * pre_bal + rng.normal(0, 450, N) + act * rng.gamma(2.0, 1000, N)   # activators move savings into the pot
# guardrail: push-notification opt-out (non-inferiority margin 0.5pp)
optout = rng.random(N) < (0.030 + 0.001 * treat)
df = pd.DataFrame(dict(treat=treat.astype(int), act=act.astype(int), pre_bal=pre_bal, post_bal=post_bal, optout=optout.astype(int),
                       age=age, salary=salary.astype(int), tenure_m=tenure_m))
nT, nC = int(df.treat.sum()), int((1 - df.treat).sum())

# --- SRM check ---------------------------------------------------------------------------
chi, p_srm = stats.chisquare([nT, nC], [N / 2, N / 2])
OUT["A_srm"] = {"treatment": nT, "control": nC, "chi2": round(float(chi), 3), "p": round(float(p_srm), 3),
                "verdict": "PASS (no sample-ratio mismatch)" if p_srm > 0.001 else "FAIL - investigate assignment"}

# --- primary metric: two-proportion z-test, CI, relative lift ----------------------------
def two_prop(y1, y0):
    p1, p0, n1, n0 = y1.mean(), y0.mean(), len(y1), len(y0)
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0); d = p1 - p0
    pool = (y1.sum() + y0.sum()) / (n1 + n0); z = d / np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n0))
    return d, (d - 1.96 * se, d + 1.96 * se), 2 * (1 - stats.norm.cdf(abs(z))), p1, p0
d, ci, p, p1, p0 = two_prop(df[df.treat == 1].act, df[df.treat == 0].act)
OUT["A_primary"] = {"control_rate": round(p0, 4), "treatment_rate": round(p1, 4), "abs_lift_pp": round(100 * d, 2),
                    "rel_lift_pct": round(100 * d / p0, 1), "ci95_pp": [round(100 * ci[0], 2), round(100 * ci[1], 2)],
                    "p_value": float(f"{p:.2e}")}

# --- A/A calibration: false-positive rate of the pipeline --------------------------------
fp = 0
for _ in range(2000):
    fake = rng.random(N) < 0.5
    _, _, pa, _, _ = two_prop(df.act[fake], df.act[~fake]); fp += pa < 0.05
OUT["A_aa_test"] = {"simulations": 2000, "false_positive_rate": round(fp / 2000, 4), "expected": 0.05}

# --- empirical power at planned n (validates sample-size formula) ------------------------
hits = 0
for _ in range(1500):
    a = rng.random(n_plan) < BASE; b = rng.random(n_plan) < BASE + MDE
    _, _, pw, _, _ = two_prop(pd.Series(b.astype(int)), pd.Series(a.astype(int))); hits += pw < 0.05
OUT["A_power_check"] = {"empirical_power_at_planned_n": round(hits / 1500, 3), "target": 0.80}

# --- CUPED on 90-day balance -------------------------------------------------------------
y, x = df.post_bal.values, df.pre_bal.values
theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
df["post_cuped"] = y - theta * (x - x.mean())
def welch(col):
    a, b = df.loc[df.treat == 1, col], df.loc[df.treat == 0, col]
    d_ = a.mean() - b.mean(); se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d_, se
d_raw, se_raw = welch("post_bal"); d_cup, se_cup = welch("post_cuped")
OUT["A_cuped"] = {"effect_raw_gbp": round(d_raw, 2), "se_raw": round(se_raw, 2), "effect_cuped_gbp": round(d_cup, 2),
                  "se_cuped": round(se_cup, 2), "variance_reduction_pct": round(100 * (1 - (se_cup / se_raw) ** 2), 1),
                  "ci95_cuped_gbp": [round(d_cup - 1.96 * se_cup, 2), round(d_cup + 1.96 * se_cup, 2)],
                  "implied_sample_size_saving_pct": round(100 * (1 - (se_cup / se_raw) ** 2), 1)}

# --- guardrail non-inferiority (opt-out) -------------------------------------------------
o1, o0 = df[df.treat == 1].optout, df[df.treat == 0].optout
dd = o1.mean() - o0.mean(); se = np.sqrt(o1.var() / len(o1) + o0.var() / len(o0)); ub = dd + 1.645 * se
OUT["A_guardrail_optout"] = {"control": round(o0.mean(), 4), "treatment": round(o1.mean(), 4), "diff_pp": round(100 * dd, 2),
                             "upper_95_one_sided_pp": round(100 * ub, 2), "margin_pp": 0.5,
                             "verdict": "PASS - non-inferior" if ub < 0.005 else "NOT ESTABLISHED - upper bound exceeds margin; stage rollout"}

# --- heterogeneity with Holm correction --------------------------------------------------
df["age_band"] = pd.cut(df.age, [0, 25, 35, 100], labels=["<25", "25-34", "35+"], right=False)
df["ten_band"] = pd.cut(df.tenure_m, [0, 6, 12, 100], labels=["<=6m", "7-12m", "13m+"])
segs = []
for var in ["salary", "age_band", "ten_band"]:
    for lvl, g in df.groupby(var, observed=True):
        dseg, cis, pseg, a1, a0 = two_prop(g[g.treat == 1].act, g[g.treat == 0].act)
        segs.append(dict(segment=f"{var}={lvl}", n=len(g), lift_pp=100 * dseg, p_raw=pseg))
S = pd.DataFrame(segs).sort_values("p_raw").reset_index(drop=True)
m = len(S)
adj = np.maximum.accumulate((m - np.arange(m)) * S.p_raw.values); S["p_holm"] = np.minimum(adj, 1.0)
S["significant_after_holm"] = S.p_holm < 0.05
S["lift_pp"] = S.lift_pp.round(2); S["p_raw"] = S.p_raw.map(lambda v: float(f"{v:.3g}")); S["p_holm"] = S.p_holm.map(lambda v: float(f"{v:.3g}"))
OUT["A_segments_holm"] = S.to_dict("records"); S.to_csv("outputs/segment_effects_holm.csv", index=False)

# --- unit economics (assumptions) --------------------------------------------------------
NIM, COST = 0.020, 0.05                          # annual net-interest-margin on deposits, GBP cost per nudged customer
rng_b = np.random.default_rng(11); boots = []
for _ in range(1000):
    i = rng_b.integers(0, N, N); b = df.iloc[i]
    a_, b_ = b[b.treat == 1], b[b.treat == 0]
    boots.append((a_.post_bal.mean() - b_.post_bal.mean() - theta * (a_.pre_bal.mean() - b_.pre_bal.mean())) * NIM - COST)
OUT["A_economics"] = {"assumptions": {"annual_NIM": NIM, "cost_per_treated_customer_gbp": COST},
                      "incremental_annual_margin_per_treated_gbp": round(float(np.mean(boots)), 2),
                      "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
                      "incremental_activations_per_100k_customers": int(round(d * 100_000)),
                      "incremental_annual_margin_if_rolled_to_500k_gbp": int(round(np.mean(boots) * 500_000))}

# =========================================================================================
# PART B - QUASI-EXPERIMENTAL: self-selected feature opt-in
# =========================================================================================
def simulate_panel(M=30_000, pre=6, post=6, true_eff=22.0, trend_violation=0.0, seed=3):
    r = np.random.default_rng(seed)
    latent = r.normal(0, 1, M)                                     # unobserved engagement
    ag = np.clip(r.normal(32, 9, M), 18, 65); sal = r.random(M) < sig(0.2 + 0.5 * latent)
    ten = r.integers(1, 25, M); ses = np.exp(2.0 + 0.4 * latent + r.normal(0, .3, M))
    base = 260 + 55 * latent + 40 * sal + 1.2 * (ag - 32) + r.normal(0, 30, M)
    adopt = r.random(M) < sig(-1.35 + 0.85 * latent + 0.4 * sal)   # self-selection on engagement
    T = pre + post; rows = []
    for t in range(T):
        spend = base + 2.0 * t + 6 * np.sin(t * np.pi / 3) + trend_violation * adopt * t + r.normal(0, 28, M)
        spend = spend + adopt * (t >= pre) * true_eff
        rows.append(spend)
    Y = np.column_stack(rows)
    X = pd.DataFrame(dict(age=ag, salary=sal.astype(int), tenure=ten, sessions=ses, pre_spend_last=Y[:, pre - 1],
                          pre_spend_avg=Y[:, :pre].mean(1)))
    return X, adopt.astype(int), Y, pre

def smd(x, a, w=None):
    x1, x0 = x[a == 1], x[a == 0]
    return float((x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2))

def analyse(X, a, Y, pre, label, true_eff=22.0):
    post_y = Y[:, pre:].mean(1); pre_y = Y[:, :pre].mean(1); out = {}
    out["naive_post_diff"] = post_y[a == 1].mean() - post_y[a == 0].mean()
    # regression adjustment on observed covariates (post-period)
    Z = np.column_stack([np.ones(len(a)), a, X[["age", "salary", "tenure", "sessions"]].values, pre_y])
    out["ols_adjusted"] = float(np.linalg.lstsq(Z, post_y, rcond=None)[0][1])
    # unmatched DiD
    out["did_unmatched"] = (post_y - pre_y)[a == 1].mean() - (post_y - pre_y)[a == 0].mean()
    # propensity matching on pre-period observables only (no outcome info from post period)
    cov = ["age", "salary", "tenure", "sessions", "pre_spend_avg"]
    Xs = (X[cov] - X[cov].mean()) / X[cov].std()
    ps = LogisticRegression(max_iter=1000).fit(Xs, a).predict_proba(Xs)[:, 1]
    lg = np.log(ps / (1 - ps)); cal = 0.2 * lg.std()
    t_idx, c_idx = np.where(a == 1)[0], np.where(a == 0)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(lg[c_idx].reshape(-1, 1))
    dist, j = nn.kneighbors(lg[t_idx].reshape(-1, 1)); keep = dist[:, 0] <= cal
    ti, ci = t_idx[keep], c_idx[j[keep, 0]]
    out["psm_post_diff"] = post_y[ti].mean() - post_y[ci].mean()
    did_pairs = (post_y[ti] - pre_y[ti]) - (post_y[ci] - pre_y[ci])
    out["psm_did"] = did_pairs.mean()
    bs = [rng.choice(did_pairs, len(did_pairs)).mean() for _ in range(500)]
    out["psm_did_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    # diagnostics
    ab = a.copy(); mask_all = np.zeros(len(a), bool)
    bal_before = {c: round(smd(X[c].values, a), 3) for c in cov}
    sel = np.concatenate([ti, ci]); a_sel = np.concatenate([np.ones(len(ti)), np.zeros(len(ci))])
    bal_after = {c: round(smd(X[c].values[sel], a_sel), 3) for c in cov}
    # event study (pre-trend): treated-minus-control gap by month, relative to last pre month
    gap = [(Y[ti, t].mean() - Y[ci, t].mean()) for t in range(Y.shape[1])]
    gap = np.array(gap) - gap[pre - 1]
    # formal pre-trend test: slope of the gap across pre months, on pair-level data
    pre_t = np.arange(pre); slopes = np.array([np.polyfit(pre_t, Y[ti[k], :pre] - Y[ci[k], :pre], 1)[0] for k in range(0, len(ti), 1)])
    t_stat, p_pt = stats.ttest_1samp(slopes, 0.0)
    out_diag = {"n_treated": int(a.sum()), "n_matched_pairs": int(keep.sum()), "treated_retained_pct": round(100 * keep.mean(), 1),
                "SMD_before": bal_before, "SMD_after": bal_after, "pretrend_slope_gbp_per_month": round(float(slopes.mean()), 3),
                "pretrend_p": float(f"{p_pt:.3g}"), "pretrend_verdict": "parallel trends plausible" if p_pt > 0.05 else "VIOLATED - do not trust DiD"}
    res = {k: (round(v, 2) if not isinstance(v, list) else [round(x, 2) for x in v]) for k, v in out.items()}
    res["true_effect"] = true_eff; res["diagnostics"] = out_diag
    return res, gap

XA, aA, YA, preA = simulate_panel(trend_violation=0.0)
resA, gapA = analyse(XA, aA, YA, preA, "valid")
XB, aB, YB, preB = simulate_panel(trend_violation=3.0, seed=4)       # adopters on steeper pre-trend
resB, gapB = analyse(XB, aB, YB, preB, "violated")
OUT["B_scenario1_parallel_trends_hold"] = resA
OUT["B_scenario2_parallel_trends_violated"] = resB

# figures --------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
# 1: CUPED variance reduction & CI
ax[0].errorbar([0, 1], [d_raw, d_cup], yerr=[1.96 * se_raw, 1.96 * se_cup], fmt="o", capsize=6, color="#1f4e79")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Raw", "CUPED"]); ax[0].set(title="90-day balance effect: 95% CI", ylabel="GBP / customer")
# 2: segment lifts
S2 = S.sort_values("lift_pp"); ax[1].barh(S2.segment, S2.lift_pp, color=["#1f4e79" if s else "#9aa5b1" for s in S2.significant_after_holm])
ax[1].axvline(100 * d, c="k", lw=.8, ls="--"); ax[1].set(title="Activation lift by segment (pp) - blue = Holm-significant")
# 3: event study
months = np.arange(-preA, 6)
ax[2].plot(months, gapA, "o-", label="Parallel trends hold (DiD valid)"); ax[2].plot(months, gapB, "s-", c="#c0392b", label="Violated (DiD biased)")
ax[2].axvline(-0.5, c="k", ls=":", lw=.8); ax[2].axhline(0, c="k", lw=.5)
ax[2].set(title="Event study: matched treated-minus-control spend gap", xlabel="Months relative to launch", ylabel="GBP vs last pre-month"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("outputs/experimentation_diagnostics.png", dpi=130)

json.dump(OUT, open("outputs/results_project2.json", "w"), indent=2, default=str)
print(json.dumps(OUT, indent=2, default=str))
