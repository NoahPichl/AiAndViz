# Phase II – What did TiRex-2 learn? (Concept)

> Concept for opening the black box of TiRex-2 on the TIME benchmark, building directly on the Phase I findings. Implementation lives in `phase2/` and `phase2.ipynb`; results are appended at the end of this file.

## 0. Starting point (Phase I findings we build on)

| ID | Finding (Phase I) | Source |
|----|-------------------|--------|
| F1 | Negative ACF1 → higher error; positive ACF1 → lower error | team_blue |
| F2 | High seasonal strength → lower error and narrower intervals | blue + red |
| F3 | Trend strength has **no consistent within-dataset effect** on error, but error *variance* grows strongly with it (pooled effect is dataset-confounded) | team_blue |
| F4 | Two failure shapes: (a) **high trend** → forecast stays flat at last value while truth drifts (US_Term_Structure); (b) **low trend, level switching** → forecast flat in the middle (Coastal_T_S) | team_blue |
| F5 | **Mean-reversion prior**: forecast drift is opposite to the historic trend | team_red |
| F6 | **Seasonal amplitude damping** (≈75 % of expected amplitude even at strong seasonality) + **flat-lining** (Coastal_T_S, current_velocity) | team_red |
| F7 | UMAP "noise cluster": smooth, trending, non-seasonal economic series → MASE ≈ 1.1, flat forecasts, exploding intervals | team_red |
| F8 | Interval width tracks error (ρ≈0.42) but some configs have ρ<0 (over-confident under structural breaks); SG_Carpark width contracts at full-period horizon | team_red |
| F9 | Horizon degradation short → medium → long | team_red |

## 1. Concept

### 1.1 What is inside the model (relevant for method choice)
* Pipeline: `PostProcessor` (**non-learned trend detection + first-order differencing**, `tta_diff=True` by default) → `Scaler` (instance norm + arcsinh) → `Tokenizer` (non-overlapping patches) → `input_patch_embedding` (residual MLP, input = patch values + NaN mask) → **stack of blocks** (time mixer = forward xLSTM alternating mLSTM/sLSTM; variate mixer = grouped attention) → `stack_out_norm` → `output_patch_embedding` (9 quantiles × patch).
* **Univariate ⇒ the variate-mixer attention is trivial (1 token per group). Attention maps are therefore NOT informative** for our setting. The meaningful internals are:
  * per-block hidden states (`TiRex2._forward_layerwise` already yields them),
  * xLSTM gate pre-activations (`igate_preact`/`fgate_preact` in mLSTM, `fgate`/`igate` in sLSTM) → forget gates = "how long does the model remember",
  * gradients (`_predict(..., preserve_grad=True)` outside `torch.no_grad`).
* NaN is a native "missing" signal (explicit mask channel) → **masking a patch with NaN is an in-distribution perturbation**, an ideal baseline for occlusion/LIME/SHAP (avoids the classic "zero baseline is out of distribution" problem).
* The differencing switch is a hand-crafted rule, not learned. Part of F4/F5/F7 may be caused by *whether the diff-path fires* → must be logged per window and controlled for.

### 1.2 Guiding principle
Not "show some saliency maps", but **hypothesis → contrast → consistency check**:
1. Define behaviour groups from Phase I (failure vs. success).
2. Compare **failing vs. successful windows *within the same dataset/config*** (matched pairs) — Phase I showed pooled effects are dataset-confounded (F3).
3. Use ≥2 complementary techniques per hypothesis: one **attribution** view (which input parts drive the output) and one **representation** view (where does the input land in the model's internal space / what is decodable) — then a **counterfactual** test to confirm causally.

### 1.3 Behaviour groups (unit = forecast window)
| Group | Definition (window_cube) | Phase I link |
|-------|--------------------------|--------------|
| G-trend-fail | trend_strength ≥ q90, seasonal_strength low, MASE > 1.1 | F4a, F5, F7 |
| G-trend-ok | same feature range, MASE < 0.7 (same datasets where possible) | control for G-trend-fail |
| G-switch-fail | trend_strength ≤ q10, MASE > 1.1, flat forecast (Coastal_T_S …) | F4b, F6 |
| G-season-ok | seasonal_strength ≥ 0.8, MASE < 0.7 | F2, F6 (damping) |
| G-negacf | e_acf1 < 0 vs. matched e_acf1 > 0 | F1 |
| G-overconf | narrow 80 % interval (low width decile within config) but high error, vs. matched well-calibrated windows | F8 |

Scope (confirmed): **short term only**. Seasonal amplitude damping and horizon degradation are deprioritised (damping only as side-product of G-switch-fail flat-lining).
Sample size: start with **100 windows per group** (≈600 total, stratified, max ~15 per dataset); after a runtime benchmark of LIME/SHAP on the GPU, scale up (target 150–200/group) if feasible. Same sample set reused by all techniques so results are directly comparable.

### 1.4 Hypotheses and technique pairs

**H1 – Context usage / "short memory" in failures** (F4, F6, F7)
*Failing windows rely almost exclusively on the most recent patches; successful seasonal windows also use lags at multiples of the season period.*
* T1 Integrated Gradients / gradient×input over time steps (scalar targets: mean median-forecast, forecast slope, 80 % interval width, window error vs. truth).
* T2 **LIME / KernelSHAP over context patches** (interpretable features = context segments on/off; off = NaN-masked).
* T3 xLSTM **forget-gate effective memory** per layer (∏ f_t → decay horizon in patches).
* Metrics: *recency index* (share of attribution mass in last k patches), *periodic-lag mass* (attribution at lags ≈ m·season).
* Combination: do gradient-based and perturbation-based attributions agree on recency? Does short gate memory coincide with high recency index?

**H2 – Which signal component drives the forecast (trend vs. season vs. noise)** (F1, F2, F3, F5)
*For failing trend series the model "explains" its forecast through level/noise, not trend; for good seasonal series through the seasonal component.*
* T4 **STL-component Shapley ("decomposition SHAP")**: players = {level, trend, seasonal, residual} from STL of the context; coalition = sum of included components (excluded = replaced by constant/smooth baseline). 4 players → exact Shapley with 16 forward passes/window. Directly maps model behaviour onto the Phase I feature axes.
* T1 IG aggregated per component (projection of IG onto STL components) as a second, gradient-based view.
* Combination: compare component shares for fail vs. ok groups; test whether negative-ACF1 windows get high residual attribution (model trying to forecast noise).

**H3 – Representation space: are failures a region of the model's internal space?** (F3, F7, F8)
*Failing windows occupy a distinct region of the hidden space; behaviour (flat-lining, mean reversion) is more predictable from representations than from TS features.*
* T5 Hidden-state embedding: last-context-token state per block → UMAP (per layer), colored by group/MASE/dataset/features; **layer slider** to see when groups separate.
* T6 **k-NN analysis**: (a) neighbour label purity of failures in representation space vs. TIME feature space (Phase I UMAP space); (b) kNN regression of MASE / flat-share from representations vs. from features; (c) nearest-neighbour galleries (query failure + 5 neighbours with forecasts).
* T7 **Linear probes per layer**: decode trend slope, seasonal strength, season period, ACF1, entropy, diff-flag, and "will fail" from hidden states. Probe curves fail vs. ok.
* Combination: if trend slope is decodable in middle layers but not at the last layer for G-trend-fail → "information present but not used"; cross-check with H1 attributions.

**H4 – Causal confirmation with counterfactual/synthetic inputs** (F5, F6, F9)
* T8 Controlled perturbations of real windows and synthetic series: add linear trend of slope s, add level shift / regime switch near the end of context (structural break → F8 over-confidence), add noise, flip sign of ACF1 structure (F1). Measure response curves: predicted slope vs. input slope (mean-reversion = slope < 0 or < identity), interval width vs. break size (does uncertainty react?), flat-line probability vs. noise level. Run with `tta_diff` on/off to separate learned behaviour from the hand-crafted diff rule.

### 1.5 Key visualizations
1. **Lag-attribution profile**: mean |attribution| vs. lag (lag normalised by season period), fail vs. ok, with CI bands — one panel per group pair. (H1)
2. **Recency-index violin/scatter** vs. MASE, colored by group; within-dataset matched lines. (H1)
3. **Attribution overlay strip**: context + forecast + truth with IG heatmap underneath and LIME patch bars — linked gallery for selected points. (H1)
4. **Forget-gate memory heatmap** (layer × context patch) per group. (H1)
5. **Component Shapley stacked bars / ternary plot** (trend / seasonal / residual share) colored by MASE. (H2)
6. **Interactive representation explorer** (plotly + ipywidgets): UMAP of hidden states, layer slider, color selector, click → attribution strip of that window. (H3)
7. **Probe-accuracy-vs-layer curves**, fail vs. ok. (H3)
8. **kNN comparison**: neighbour-purity bars and kNN-predicted vs. true MASE (representation vs. feature space). (H3)
9. **Counterfactual response curves** (input slope → output slope, amplitude in → out), diff on/off. (H4)

