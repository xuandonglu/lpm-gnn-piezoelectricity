"""PZ27 disk: external data acquisition, specimen calibration and prediction."""

import argparse
import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

from lpm_gnn.io import file_hash, new_output, save_json
from lpm_gnn.physics.calibration import residual, step_vector_cg
from lpm_gnn.physics.piezo_axisym_complex_loss_messages import AxisymmetricComplexLossMessageGraph
from lpm_gnn.physics.piezo_axisym_complex_parameter_graph import (
    AxisymmetricComplexPortParameterGraph,
)
from lpm_gnn.physics.piezo_axisym_complex_solve import AxisymmetricComplexMessageProblem
from lpm_gnn.physics.piezo_axisym_message_multigrid import make_axisym_vcycle_factory

DATA_URL = "https://wp.icmm.csic.es/gffm/wp-content/uploads/sites/34/2023/04/hand_rad.zip"
RAW_SHA256 = "6ec9321da439f5ba5c2bb32bf6e479f17ea8b8dac0af4cd0350a8ae305c59c97"
CALIBRATION_KHZ = np.array([110.0, 111.2, 112.5, 113.8, 115.0, 116.2, 117.5, 118.8, 120.0])
NOMINAL = [
    147391094711.82275,
    104936542310.16872,
    93661441729.15106,
    112634000000.0,
    23013000000.0,
    11.6439,
    -3.0874407739346665,
    16.0264,
    1129.69 * 8.8541878128e-12,
    913.73 * 8.8541878128e-12,
]


def download(destination):
    import hashlib

    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "lpm-gnn-reproducibility/0.1"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read()
    # Read only the verified numeric file; never extract executables or archive paths.
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [n for n in archive.namelist() if Path(n).name.lower() == "raw_data.txt"]
        if len(names) != 1:
            raise RuntimeError(
                "The upstream archive layout changed; inspect the official download manually."
            )
        raw = archive.read(names[0])
    if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
        raise RuntimeError("The upstream measurement file differs from the paper source.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    load_measurements(destination)
    print(f"Saved verified radial-disk measurements: {destination}")


def load_measurements(path):
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError("Expected three numeric columns: frequency_kHz, magnitude_S, phase_rad.")
    if np.any(np.diff(raw[:, 0]) <= 0) or np.any(raw[:, 1] <= 0):
        raise ValueError("Frequencies must increase and admittance magnitudes must be positive.")
    return raw


def problem_for(nr=40, nz=4, device="cpu"):
    graph = AxisymmetricComplexLossMessageGraph(
        nr, nz, radius=0.01, height=0.001, rho=7700.0, device=device
    )
    problem = AxisymmetricComplexMessageProblem(
        graph,
        rtol=1e-9,
        restart=384,
        maxiter=1600,
        preconditioner_factory=make_axisym_vcycle_factory(),
    )
    return graph, problem


def metrics(predicted, observed):
    phase = np.angle(predicted / observed)
    return dict(
        samples=len(predicted),
        admittance_magnitude_MAPE_percent=float(
            100 * np.mean(abs(abs(predicted) / abs(observed) - 1))
        ),
        phase_RMS_degrees=float(np.rad2deg(np.sqrt(np.mean(phase**2)))),
        complex_relative_L2=float(np.linalg.norm(predicted - observed) / np.linalg.norm(observed)),
    )


def fit(args):
    out = new_output(args.output)
    raw = load_measurements(args.measurements)
    indices = np.array([np.argmin(abs(raw[:, 0] - f)) for f in CALIBRATION_KHZ])
    if not np.allclose(raw[indices, 0], CALIBRATION_KHZ, rtol=0, atol=1e-7):
        raise ValueError("The nine specified 110-120 kHz calibration frequencies are missing.")
    _, problem = problem_for(args.nr, args.nz, "cpu")
    model = AxisymmetricComplexPortParameterGraph(
        problem, torch.tensor(NOMINAL, dtype=torch.float64)
    )
    frequency = torch.tensor(raw[indices, 0] * 1000, dtype=torch.float64)
    target = torch.tensor(
        np.column_stack((np.log(raw[indices, 1]), raw[indices, 2])), dtype=torch.float64
    )
    theta = torch.tensor([0.0, 0.0, 0.0, math.log(0.01), math.log(0.015), 0.0], dtype=torch.float64)
    lower = torch.tensor(
        [math.log(0.5), math.log(0.3), math.log(0.3), math.log(0.0002), math.log(0.0002), -3.0],
        dtype=torch.float64,
    )
    upper = torch.tensor(
        [math.log(1.5), math.log(2.0), math.log(2.0), math.log(0.08), math.log(0.08), 3.0],
        dtype=torch.float64,
    )
    history, best = [], None
    damping = 0.001

    def evaluate(point):
        nonlocal best
        prediction, jac, forward = model.evaluate(point, frequency, True)
        if not all(r["info"]["converged"] for r in forward["records"]):
            raise RuntimeError("Calibration forward solve did not converge.")
        res = residual(prediction, target)
        loss = float(0.5 * res.square().mean())
        row = dict(evaluation=len(history), theta=point.tolist(), loss=loss)
        history.append(row)
        if best is None or loss < best["loss"]:
            best = row
        save_json(out / "progress.json", dict(best=best, evaluations=history))
        print(f"Calibration evaluation {len(history)}: objective={loss:.6g}", flush=True)
        return jac, res, loss

    jac, res, loss = evaluate(theta)
    stop = "evaluation budget"
    while len(history) < args.max_evaluations:
        diagonal = jac.square().mean(dim=(0, 1)).clamp_min(1e-14)
        gradient = (jac * res[:, :, None]).mean(dim=(0, 1))
        active = ((theta <= lower + 1e-10) & (gradient > 0)) | (
            (theta >= upper - 1e-10) & (gradient < 0)
        )
        delta, gradient = step_vector_cg(jac, res, diagonal, damping, (~active).double())
        cap = torch.tensor([0.10, 0.25, 0.25, 0.8, 0.8, 0.5], dtype=torch.float64)
        delta = delta / max(1.0, float((delta.abs() / cap).max()))
        candidate = torch.maximum(lower, torch.minimum(upper, theta + delta))
        if float((candidate - theta).abs().max()) < 2e-6:
            stop = "parameter step tolerance"
            break
        if float((gradient * (~active)).abs().max()) < 1e-10:
            stop = "projected gradient tolerance"
            break
        trial_jac, trial_res, trial_loss = evaluate(candidate)
        if trial_loss < loss:
            theta, jac, res, loss = candidate, trial_jac, trial_res, trial_loss
            damping = max(damping * 0.3, 1e-8)
            if loss < 1e-15:
                stop = "objective tolerance"
                break
        else:
            damping = min(damping * 10, 1e10)
            if damping >= 1e9:
                stop = "damping limit"
                break
    frozen = torch.tensor(best["theta"], dtype=torch.float64)
    report = dict(
        recovered=model.parameters.decode(frozen),
        theta=best["theta"],
        best=best,
        calibration_frequencies_kHz=CALIBRATION_KHZ.tolist(),
        measurement_sha256=file_hash(args.measurements),
        grid=dict(nr=args.nr, nz=args.nz),
        stopping_reason=stop,
        evaluations=history,
        role="Specimen calibration supplies fixed coefficients for response prediction.",
    )
    save_json(out / "calibration.json", report)
    return report


def predict(args):
    out = new_output(args.output)
    raw = load_measurements(args.measurements)
    fit_record = json.loads(Path(args.calibration).read_text())
    if file_hash(args.measurements) != fit_record["measurement_sha256"]:
        raise ValueError("Prediction measurements differ from the calibrated specimen source.")
    frozen = fit_record["recovered"]
    graph, problem = problem_for(args.nr, args.nz, args.device)
    real = torch.tensor(frozen["real_SI"], dtype=torch.float64, device=args.device)
    imag = torch.tensor(frozen["imag_SI"], dtype=torch.float64, device=args.device)
    indices = (
        np.arange(len(raw))
        if args.frequencies_khz is None
        else np.array([np.argmin(abs(raw[:, 0] - f)) for f in args.frequencies_khz])
    )
    predictions, rows, last = [], [], None
    for first in range(0, len(indices), args.batch):
        ids = indices[first : first + args.batch]
        frequency = torch.tensor(raw[ids, 0] * 1000, dtype=torch.float64, device=args.device)
        warm = None if last is None else last[None].expand(len(ids), graph.n, 3).clone()
        result = problem.solve_batch(
            real, imag, frequency, eta_nonaffine=frozen["eta_mechanical"], initial_state=warm
        )
        if not all(r["info"]["converged"] for r in result["records"]):
            raise RuntimeError("A disk prediction did not converge.")
        predictions.append(result["admittance"].cpu().numpy())
        rows.extend(result["records"])
        np.savez_compressed(
            out / f"fields_{first:04d}.npz",
            frequency_Hz=raw[ids, 0] * 1000,
            state=result["state"].cpu().numpy(),
            coordinates=graph.coordinates.cpu().numpy(),
        )
        last = result["state"][-1].detach()
        print(f"Predicted {first + len(ids)}/{len(indices)} frequencies", flush=True)
    predicted = np.concatenate(predictions)
    measured = raw[indices, 1] * np.exp(1j * raw[indices, 2])
    frequencies = raw[indices, 0]
    masks = dict(
        all=np.ones(len(indices), dtype=bool),
        lower=frequencies < 110,
        upper=frequencies > 120,
        out_of_band=(frequencies < 110) | (frequencies > 120),
    )
    save_json(
        out / "report.json",
        dict(
            calibration_sha256=file_hash(args.calibration),
            metrics={k: metrics(predicted[v], measured[v]) for k, v in masks.items() if np.any(v)},
            records=rows,
            nr=args.nr,
            nz=args.nz,
        ),
    )
    np.savez_compressed(
        out / "spectrum.npz",
        frequency_Hz=frequencies * 1000,
        predicted_admittance=predicted,
        measured_admittance=measured,
    )
    if args.plot:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 10, "mathtext.fontset": "stix"})
        fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
        ax.semilogy(frequencies, abs(measured) * 1000, color=".25", label="Measured")
        ax.semilogy(frequencies, abs(predicted) * 1000, color="#2268ad", label="LPM-GNN")
        ax.set(xlabel="Frequency (kHz)", ylabel=r"$|Y|$ (mS)")
        ax.legend(loc="best", frameon=False)
        fig.savefig(out / "admittance.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def verify(args):
    from lpm_gnn.reference.axisymmetric_fem import AxisymmetricDisk, Material

    out = new_output(args.output)
    graph, problem = problem_for(args.nr, args.nz)
    real = torch.tensor(NOMINAL, dtype=torch.float64)
    imag = torch.cat((0.01 * real[:5], torch.zeros(3, dtype=torch.float64), -0.015 * real[8:]))
    frequency = torch.tensor(args.frequencies_khz, dtype=torch.float64) * 1000
    result = problem.solve_batch(real, imag, frequency, eta_nonaffine=0.01)
    if not all(r["info"]["converged"] for r in result["records"]):
        raise RuntimeError("Nominal disk graph solve did not converge.")
    values = {
        k: v for k, v in zip(("c11", "c12", "c13", "c33", "c44", "e15", "e31", "e33"), NOMINAL[:8])
    }
    material = Material(
        **values, eps11=1129.69, eps33=913.73, rho=7700.0, loss_m=0.01, loss_d=0.015
    )
    reference = AxisymmetricDisk(
        args.reference_nr, args.reference_nz, radius=0.01, height=0.001, order=2
    )
    fem = reference.spectrum(material, frequency.numpy(), diagnostics=True)
    predicted = result["admittance"].numpy()
    save_json(
        out / "report.json",
        dict(
            metrics=metrics(predicted, fem["admittance"]),
            graph_records=result["records"],
            reference="independent axisymmetric Gaussian-Q2 continuum FEM",
        ),
    )
    np.savez_compressed(
        out / "nominal_comparison.npz",
        frequency_Hz=frequency.numpy(),
        graph_admittance=predicted,
        fem_admittance=fem["admittance"],
        graph_state=result["state"].numpy(),
        coordinates=graph.coordinates.numpy(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser(
        "download", help="Acquire only the verified external numeric measurements."
    )
    acquire.add_argument("--output", default="data/pz27_radial.csv")
    for name in ("fit", "predict", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", default=f"outputs/disk_{name}")
        sub.add_argument("--nr", type=int, default=40)
        sub.add_argument("--nz", type=int, default=4)
        if name != "verify":
            sub.add_argument("--measurements", default="data/pz27_radial.csv")
        if name == "fit":
            sub.add_argument("--max-evaluations", type=int, default=40)
        if name == "predict":
            sub.add_argument("--calibration", default="outputs/disk_fit/calibration.json")
            sub.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
            sub.add_argument("--batch", type=int, default=32)
            sub.add_argument("--frequencies-khz", type=float, nargs="+")
            sub.add_argument("--plot", action="store_true")
        if name == "verify":
            sub.add_argument("--reference-nr", type=int, default=40)
            sub.add_argument("--reference-nz", type=int, default=4)
            sub.add_argument(
                "--frequencies-khz",
                type=float,
                nargs="+",
                default=[78.0, 90.0, 100.0, 110.0, 118.0, 130.0, 140.0, 150.0, 158.0],
            )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "download":
        download(args.output)
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
