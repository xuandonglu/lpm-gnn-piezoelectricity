# LPM-GNN for piezoelectricity

Implementation accompanying **A nonlocal lattice particle formulation for piezoelectricity using physics-guided graph neural networks**.

The code covers the energy-based piezoelectric particle model, graph equilibrium solution, material-aligned lattice rotation, double-window and bimorph examples, an illustrative learned nonlinear update, and the axisymmetric disk comparison with measured admittance.

## Installation

Use Python 3.9 or later. The release was checked with Python 3.9.13, NumPy 2.0.2, SciPy 1.13.1, PyTorch 2.5.1, Matplotlib 3.9.4 and PyAMG 5.3.0. From this directory:

```sh
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS), then install:

```sh
python -m pip install --upgrade pip
python -m pip install -e .
```

The examples run on CPU by default. Nonlinear training also supports CUDA through PyTorch; install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) for the local system before installing this package if required.

## Quick start

Generate and solve the coarsest double-window case, including its independent FEM reference and field plots:

```sh
python -m lpm_gnn.examples.double_window --grids 20 --plot
```

Run the verification tests:

```sh
python -m unittest discover -s tests -v
```

## Examples

| Example | Entry point | Generated outputs |
| --- | --- | --- |
| Aligned double-window frame | `lpm_gnn.examples.double_window` | Particle and independent Q4 fields, response errors, balance diagnostics |
| Fixed-domain orientation verification | `lpm_gnn.examples.orientation` | Rotated particle fields, independent rotated-tensor FEM fields, refinement errors |
| Piezoelectric bimorph | `lpm_gnn.examples.bimorph` | Free/blocked actuation, short/open-circuit sensing, independent FEM comparisons |
| Illustrative nonlinear extension | `lpm_gnn.examples.nonlinear` | Synthetic datasets, trained predictors, independent test comparisons |
| Axisymmetric PZ27 disk | `lpm_gnn.examples.disk` | Nominal Q2 verification, specimen calibration and frequency predictions |

Every entry point supports `--help`. Result directories are created explicitly; choose a new output path for a new run.

### Linear examples

```sh
python -m lpm_gnn.examples.double_window --grids 20 40 80 --output outputs/double_window --plot
python -m lpm_gnn.examples.orientation --grids 20 40 80 160 --angles 30 60 --reference-n 320 --output outputs/orientation
python -m lpm_gnn.examples.bimorph --ny 4 8 16 32 --output outputs/bimorph --plot
```

A smaller orientation check is available with `--grids 20 --reference-n 40`. Add `--iterative-check` to compare the rotated-frame matrix-free graph solver against the exact sparse expansion. The refinement study uses sparse LU for the expanded particle energy; the graph iteration and the independent FEM reference are separate implementations.

### Nonlinear data generation, training and evaluation

All numerical observations and response labels are generated from the prescribed nonlinear particle enthalpy. The four random inputs are total force, electrode voltage, traction centre and traction width. Geometry and material are fixed. The default splits contain 512 training, 128 validation and 256 independent test cases.

```sh
python -m lpm_gnn.examples.nonlinear generate --output outputs/nonlinear
python -m lpm_gnn.examples.nonlinear train --run outputs/nonlinear --method all --device auto
python -m lpm_gnn.examples.nonlinear evaluate --run outputs/nonlinear --output outputs/nonlinear_evaluation --plot
```

The four compared configurations are:

- **Data-GNN:** a graph predictor trained with field supervision.
- **PI-GNN:** the same predictor with an additional equilibrium-residual loss, using fixed training-input RMS scales.
- **LPM-GNN:** a graph controller predicts positive residual weights for three coupled V-cycle updates; each update recomputes the nonlinear physical residual.
- **Newton-FGMRES:** a conventional nonlinear solve with an exact tangent, residual line search and coupled multigrid preconditioning.

Every configuration uses the same cached exact linear initial field. The generator builds this prior from the physical operator. Training selects checkpoints using validation field error; the evaluation command opens the independent test split after loading the selected checkpoints. Matched-accuracy timing replays a recorded numerical iteration budget selected after testing; its stopping event is an ex-post comparison, not an online convergence rule.

For an installation check, reduce only the dataset and training budget:

```sh
python -m lpm_gnn.examples.nonlinear generate --output outputs/nonlinear_smoke --training 8 --validation 4 --test 4
python -m lpm_gnn.examples.nonlinear train --run outputs/nonlinear_smoke --steps 3 --batch-size 4 --checkpoint-every 1 --device cpu
python -m lpm_gnn.examples.nonlinear evaluate --run outputs/nonlinear_smoke --output outputs/nonlinear_smoke_evaluation --repeats 1 --limit 2
```

This reduced run checks the pipeline. Use the default dataset and 4,000 training steps for the paper protocol.

### Experimental disk

The measured radial-disk admittance is an external dataset from the [ICMM-CSIC coefficient-determination examples](https://wp.icmm.csic.es/gffm/determination-of-coefficients/). The optional download command reads only the numeric measurement file and verifies its source hash:

```sh
python -m lpm_gnn.examples.disk download --output data/pz27_radial.csv
python -m lpm_gnn.examples.disk fit --measurements data/pz27_radial.csv --output outputs/disk_fit
python -m lpm_gnn.examples.disk predict --measurements data/pz27_radial.csv --calibration outputs/disk_fit/calibration.json --output outputs/disk_prediction --plot
```

Calibration uses nine frequencies between 110 and 120 kHz. The frozen coefficients then predict the complete measured spectrum. This calibration supplies specimen-specific model inputs for the experimental comparison. The nominal tensor shape is taken from the [manufacturer's modelling data](https://www.ferropermpiezoceramics.com/resources/data-for-modelling/); the corresponding constants and source details are documented in [docs/reproduction.md](docs/reproduction.md).

The nominal disk comparison with an independent Q2 continuum model requires no measured data:

```sh
python -m lpm_gnn.examples.disk verify --output outputs/disk_nominal
```

## Data and repository layout

This source distribution includes data-generation code. Numerical fields, training datasets, learned checkpoints and downloaded measurements are produced when the commands run and are excluded by `.gitignore`.

```text
src/lpm_gnn/
  physics/       Particle enthalpy, graph residuals, constraints and solvers
  nonlinear/     Synthetic data, coupled cycles, networks, training and evaluation
  reference/     Independent planar Q4 and axisymmetric Q2 FEM implementations
  examples/      Runnable study entry points
  plotting.py    Result-field plots
tests/           Physical and implementation checks
docs/            Reproduction settings and source provenance
```

See [docs/reproduction.md](docs/reproduction.md) for units, seeds, model settings and the mapping to the paper. Re-training and timings depend on the numerical libraries and hardware; generated reports record the run settings and input hashes.
