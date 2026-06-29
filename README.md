# Enhanced GAT

The implemented model variants are:

- `GAT`: baseline two-layer graph attention encoder.
- `2-Hop GAT`: GAT encoder with an explicit learnable two-hop residual message-passing path.
- `Filtered 2-Hop GAT`: rule-guided 2-hop GAT using the RDF patterns described in the manuscript: `subClassOf + subClassOf` for subclass transitivity and `rdf:type + subClassOf` for membership reasoning. The three views are concatenated before the DistMult decoder.

## Repository Contents

- [main.py](main.py): command-line entry point.
- [src/nsorn_protocol.py](src/nsorn_protocol.py): NSORN-compatible experiment protocol used for thesis comparison.
- [src/gnn.py](src/gnn.py): GAT model training and evaluation utilities.
- [src/utils.py](src/utils.py): dataset parsing and graph construction utilities.
- `datasets/`: local OWL/NT datasets.
- `results/`: generated result files.
- `models/`: generated checkpoints.

## Environment

The experiments were run with:

- Python 3.12
- PyTorch `2.11.0+cpu`
- PyTorch Geometric `2.6.1`/`2.7.0`
- CPU device

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `torch` or `torch-geometric` installation fails, install a compatible PyTorch build for your platform first, then install the remaining requirements.

## Data

The code expects datasets under `datasets/`. For a self-contained thesis submission, include this directory with the submitted archive. If the repository is shared without data, download the data and keep the same local layout:

- Noisy Pizza, Family, and OWL2Bench benchmark datasets generated with the NSORN framework: [jloe2911/NSORN](https://github.com/jloe2911/NSORN)
- OWL2Bench, ORE, and CaLiGraph benchmark datasets: <https://semrec.github.io/>

Required local directories:

```text
datasets/noise_pizza/
datasets/noise_family/
datasets/noise_OWL2Bench/
datasets/OWL2Bench/
datasets/ORE/
datasets/clg/
```

## Reproducing the Results

Use `--protocol nsorn` for direct comparison against the NSORN experiments.

The NSORN-compatible protocol uses five seeds by default, trains for 300 epochs, selects the best checkpoint by validation loss, and reports `MRR`, `Hits@1`, `Hits@5`, and `Hits@10`.

It trains on the clean train graph, validates on the clean validation graph, and evaluates on the selected clean or noisy test graph. For noisy groups, the same checkpoint is reused across the selected test files from that split.

### Main Runs

Run the following commands from the repository root:

```powershell
python main.py --protocol nsorn --groups noise_pizza_100
python main.py --protocol nsorn --groups noise_pizza_250
python main.py --protocol nsorn --groups noise_family
python main.py --protocol nsorn --groups noise_owl2bench
python main.py --protocol nsorn --groups owl2bench
python main.py --protocol nsorn --groups ore
python main.py --protocol nsorn --groups clg
```

Result files are written to:

```text
results/
```

### Quick Smoke Test

Use this to verify the environment and code path without running the full experiment:

```powershell
python main.py --protocol nsorn --groups noise_pizza_100 --datasets noise_pizza_100_pizza_100_test --results-dir tmp_nsorn_smoke --nsorn-runs 1 --nsorn-epochs 2
```

This should train all three model variants for one seed and write:

```text
tmp_nsorn_smoke/noise_pizza_100_pizza_100_test_nsorn_protocol.txt
```

## Command-Line Options

Show all options:

```powershell
python main.py --help
```

Useful arguments:

- `--protocol {current,nsorn}`: choose the original workflow or the NSORN-compatible fixed train/validation/test workflow.
- `--groups ...`: dataset groups to run.
- `--datasets ...`: optional dataset ids to restrict the run.
- `--results-dir`: output directory. Default: `results`.
- `--nsorn-runs`: repeated runs for `--protocol nsorn`. Default: `5`.
- `--nsorn-epochs`: epochs per run for `--protocol nsorn`. Default: `300`.

For `--protocol nsorn`, `--mode` is ignored because this workflow always trains, validates, restores the best checkpoint, evaluates, and writes results in one run.

## Dataset Groups

The CLI exposes dataset groups through `--groups`:

- `noise_pizza_100`: Pizza 100 no-noise and noisy test files.
- `noise_pizza_250`: Pizza 250 no-noise and noisy test files.
- `noise_family`: Family no-noise and noisy test files.
- `noise_owl2bench`: OWL2Bench-derived noisy test files.
- `owl2bench`: `OWL2Bench1`, `OWL2Bench2`.
- `ore`: `ORE1`, `ORE2`, `ORE3`.
- `clg`: `clg_10e4`, `clg_10e5`.
- `all`: every group above.

## Reproducibility Notes

- NSORN-compatible runs use seeds `42 + run_index`.
- The default NSORN-compatible run uses five seeds and 300 epochs per model variant.
- The best model state is selected by validation loss.
- Result files contain individual run metrics and the mean across runs.
- If you change `--nsorn-runs` or `--nsorn-epochs`, report those values with the table.
