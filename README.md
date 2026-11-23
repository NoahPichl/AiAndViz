# ReadMe

## Task Details

### Task

We want to discover biases and failure modes for univariate time series forecasting. To this end we use the TIME Benchmark Data as our input data and want to design visualization which help discover modes when the tirex-2 model fails to produce good results. To this end we found it interesting to compare different statistical time series features and compare the output based on the difference between these features. 

### Model

We choose the Tirex-2 model, a zero-shot time series forecasting model based on the xLSTM architecture. At the time of its release it was SOTA in its model weights class and promised quite good results on a wide variety of tasks. The base-version of the model is open-source and its low computational demand makes it able to run on CPU for inference. Although the model support multivariate input, we stayed with univariate input due to complexity reasons. <br>
Github Repo: https://github.com/NX-AI/tirex-2 <br>
Huggingface: https://huggingface.co/NX-AI/TiRex-2 <br>
Paper: https://arxiv.org/abs/2607.01204 <br>
Blog Post: https://www.nx-ai.com/en/tirex-2

### Dataset

We choose the TIME Dataset, since it is a common Benchmark beside GIFT-Eval and fev-bench but was not used for developing the tirex-2 model (according to its paper). It has roughly 50 fresh datasets across different domains with 98 forecasting tasks and a total number of 5460 samples. Along with the dataset, the publisher realeased all computed features for each sample which were used to analyze the difference between differen time series foundation models in there original word. <br>
<br>
Paper: https://arxiv.org/pdf/2602.12147 <br>
Github Repo: https://github.com/zqiao11/TIME <br>
Huggingface: https://huggingface.co/datasets/Real-TSF/TIME


## Getting Started

Check out `getting_startet.ipynb`

## Python Environments

The root project and TIME use separate environments because their NumPy
requirements are incompatible. Run these commands from the repository root:

```bash
uv sync
uv sync --project TIME --python 3.12
```

The environments are created at `.venv` and `TIME/.venv`, respectively.
Run root project commands with `uv run`, and TIME commands with
`uv run --project TIME`. For example:

```bash
uv run --project TIME python -c "import timebench; print(timebench.__file__)"
```

## TiRex-2 Experiments

Run the TiRex-2 experiment in the root environment, which includes its model
and benchmark dependencies. The script automatically loads `TIME_DATASET`
from the root `.env`:

```dotenv
TIME_DATASET=data/TIME
```

Run from the repository root so the relative dataset path resolves correctly:

```bash
uv run python TIME/experiments/tirex2_model.py \
	--dataset "Global_Influenza/W" \
	--terms short \
	--batch-size 16
```

Results are saved to `output/results/TiRex2` by default.


## Phase II – Opening the Black Box

Concept and results: `ai-interactions/phase2.md`. Notebook: `phase2.ipynb`. Code: `phase2/`.

Additional setup (raw TIME data and the TIME submodule are needed to rebuild forecast windows):

```bash
git submodule update --init
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Real-TSF/TIME', repo_type='dataset', local_dir='data/TIME')"
```

Long-running jobs cache their results in `output/phase2/` (the notebook only reads the caches):

```bash
uv run python -m phase2 attributions      # IG, LIME, KernelSHAP, STL-Shapley (~1.5 h on an RTX 3070)
uv run python -m phase2 representations   # hidden states + mLSTM gates (~1 min)
```

| module | content |
|---|---|
| `phase2/model_io.py` | TiRex-2 wrapper: differentiable forecasts, hidden states per block, mLSTM gate capture |
| `phase2/windows.py` | rebuilds every TIME forecast window exactly like the benchmark runner |
| `phase2/groups.py` | per-window behaviour table and the Phase I behaviour groups |
| `phase2/attribution.py` | Integrated Gradients (captum), LIME, KernelSHAP (shap), STL-component Shapley |
| `phase2/representations.py` | layer-wise maps (UMAP), dataset-grouped linear probes, kNN analysis |
| `phase2/counterfactual.py` | trend injection, level shifts, residual autocorrelation experiments |
| `phase2/analysis.py`, `phase2/plots.py` | within-dataset statistics, static figures and the interactive explorer |

Notes: the model is built with the native PyTorch kernels and then moved to the GPU because the Triton mLSTM
kernels do not compile with the installed Triton; `Scaler.scale` is replaced by a numerically identical version
with NaN-free gradients. The forecasts reproduce the stored Phase I predictions (max. relative deviation 5e-4).
