"""Independent test evaluation and casewise matched-accuracy numerical timing."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from lpm_gnn.io import file_hash, new_output, save_json

from .cpu_backend import CompactCPU, ScheduleCPU
from .data import load_split, normalize_parameters
from .numerical import PrefixNewton
from .physics import STATE_SCALE, field_errors
from .training import METHODS, environment, predictor


def elapsed(function, repeats):
    function()  # Untimed warm-up, identical convention for every method.
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def evaluate(run, output, repeats=5, limit=None, plot=False):
    if repeats < 1 or (limit is not None and limit < 1):
        raise ValueError("Timing repeats and any test limit must be positive.")
    out = new_output(output)
    torch.set_num_threads(1)
    protocol = json.loads((run / "protocol.json").read_text())
    for split, expected in protocol["files"].items():
        if file_hash(run / "data" / (split + ".npz")) != expected:
            raise ValueError(f"Generated {split} file changed after protocol creation.")
    env = environment(run).csr()
    env.cpu_lu = True
    networks, backends, checkpoint_hashes = {}, {}, {}
    for method in METHODS:
        path = run / "training" / method / "checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for split, expected in checkpoint["setup"]["data_sha256"].items():
            if file_hash(run / "data" / (split + ".npz")) != expected:
                raise ValueError(f"Training source mismatch: {method}, {split}")
        if checkpoint["setup"]["normalization_sha256"] != file_hash(run / "normalization.json"):
            raise ValueError("Training normalization changed.")
        if checkpoint["setup"]["basis_sha256"] != file_hash(run / "linear_basis.npz"):
            raise ValueError("The common linear prior changed.")
        net = predictor(method).double().eval()
        net.load_state_dict(checkpoint["state_dict"])
        networks[method] = net
        backends[method] = (ScheduleCPU if method == "lpm_gnn" else CompactCPU)(net, env)
        checkpoint_hashes[method] = file_hash(path)
    # Checkpoints and training choices are frozen before opening test labels.
    save_json(
        out / "evaluation_protocol.json",
        dict(
            checkpoints=checkpoint_hashes,
            test_sha256=protocol["files"]["test"],
            timing_repeats=repeats,
            precision="float64 NumPy/SciPy; one BLAS thread",
            online_includes="Common linear-prior application and each method's response computations.",
            offline_excludes="Training, geometry/operators, prior construction and coarse factorization.",
            matched_timing="Ex-post selection of first actual Newton/Krylov state meeting each predictor's field error; replay excludes reference-error work.",
        ),
    )
    data = load_split(run, "test")
    if limit is not None:
        data = {k: v[:limit] for k, v in data.items()}
    params = normalize_parameters(data["parameters"])
    count = len(params)
    parity = {}
    with torch.no_grad():
        b, lift, p = (
            torch.tensor(v[:1], dtype=torch.float64) for v in (data["b"], data["lift"], params)
        )
        for method in METHODS:
            expected = networks[method](env, b, lift, p)[0].numpy()
            actual = backends[method](data["b"][0], data["lift"][0], params[0])
            parity[method] = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
            if parity[method] > 1e-7:
                raise RuntimeError(f"CPU inference parity failed: {method}")
    solver = PrefixNewton(env.cycle)
    rows = []
    predictions = {name: [] for name in (*METHODS, "newton_fgmres")}
    with threadpool_limits(limits=1):
        for i in range(count):
            b, lift, p, target = data["b"][i], data["lift"][i], params[i], data["y"][i]
            current = {}
            for method in METHODS:
                forward = lambda name=method: backends[name](b, lift, p)
                y = forward()
                predictions[method].append(y)
                current[method] = dict(
                    errors={k: float(v) for k, v in field_errors(env.a, y, target).items()},
                    median_online_seconds=elapsed(forward, repeats),
                )
            strict, info, events, fields = solver.solve(b, lift, p, "coupled", capture=True)
            predictions["newton_fgmres"].append(strict)
            current["newton_fgmres"] = dict(
                errors={k: float(v) for k, v in field_errors(env.a, strict, target).items()},
                solver=info,
                median_online_seconds=elapsed(lambda: solver.solve(b, lift, p, "coupled"), repeats),
            )
            candidates = np.stack(fields).reshape(-1, env.a["mesh"].n, 3)
            trajectory = field_errors(env.a, candidates, target)
            for method in METHODS:
                bound = current[method]["errors"]["combined_relative_L2"]
                qualifying = np.flatnonzero(trajectory["combined_relative_L2"] <= bound)
                if not len(qualifying):
                    current[method]["matched_newton"] = dict(reached=False)
                    continue
                index = int(qualifying[0])
                event = events[index]
                replay = lambda: solver.replay(b, lift, p, "coupled", event)
                replayed = replay()
                difference = float(
                    np.linalg.norm(replayed.ravel() - fields[index])
                    / max(np.linalg.norm(fields[index]), 1e-30)
                )
                if difference > 1e-12:
                    raise RuntimeError("Recorded numerical prefix and replay differ.")
                current[method]["matched_newton"] = dict(
                    reached=True,
                    event=event,
                    field_error=float(trajectory["combined_relative_L2"][index]),
                    median_online_seconds=elapsed(replay, repeats),
                    replay_difference=difference,
                )
            rows.append(dict(index=i, methods=current))
            print(f"Evaluated independent test case {i + 1}/{count}", flush=True)
    summary = {}
    for method in (*METHODS, "newton_fgmres"):
        values = [r["methods"][method] for r in rows]
        summary[method] = dict(
            mean_combined_error=float(
                np.mean([v["errors"]["combined_relative_L2"] for v in values])
            ),
            mean_displacement_error=float(
                np.mean([v["errors"]["displacement_relative_L2"] for v in values])
            ),
            mean_potential_error=float(
                np.mean([v["errors"]["potential_relative_L2"] for v in values])
            ),
            mean_online_seconds=float(np.mean([v["median_online_seconds"] for v in values])),
        )
        if method in METHODS:
            matching = [v["matched_newton"] for v in values if v["matched_newton"]["reached"]]
            summary[method]["matched_newton_cases"] = len(matching)
            if matching:
                summary[method]["mean_matched_newton_seconds"] = float(
                    np.mean([v["median_online_seconds"] for v in matching])
                )
    save_json(
        out / "report.json",
        dict(test_cases=count, inference_backend_parity=parity, summary=summary, cases=rows),
    )
    np.savez_compressed(
        out / "predictions.npz",
        **{k: np.stack(v) * STATE_SCALE for k, v in predictions.items()},
        reference=data["y"] * STATE_SCALE,
        parameters=data["parameters"],
    )
    if plot:
        from lpm_gnn.plotting import planar_fields

        for method in METHODS:
            planar_fields(
                env.a["mesh"].coordinates,
                np.asarray(predictions[method][0]) * STATE_SCALE,
                data["y"][0] * STATE_SCALE,
                out / f"fields_case0000_{method}.png",
                env.a["mesh"].cells,
                labels=(method.replace("_", " ").upper(), "Reference"),
            )
    return summary
