"""Generate all synthetic samples, the shared prior and training-only scales."""

import numpy as np

from lpm_gnn.io import file_hash, new_output, save_json

from .physics import STATE_SCALE, SparsePhysics, load_case, sample_cases


def normalize_parameters(parameters):
    p = np.array(parameters, dtype=np.float64, copy=True)
    p[:, 0] /= 0.25
    p[:, 1] /= 100
    p[:, 2] = (p[:, 2] - 10) / 6
    p[:, 3] -= 2
    return p


def load_split(run, split):
    with np.load(run / "data" / (split + ".npz")) as source:
        return {key: source[key].copy() for key in source.files}


def build_linear_basis(physics, destination):
    a = physics.a
    top, free = a["top"], physics.free
    lookup = {k: i for i, k in enumerate(free)}
    rhs = np.zeros((len(free), len(top)))
    for j, node in enumerate(top):
        rhs[lookup[3 * node + 1], j] = 1.0
    force = np.zeros((len(top), a["mesh"].n * 3))
    force[:, free] = physics.linear_lu.solve(rhs).T
    case = dict(force_N=0.0, voltage_V=100.0, center_mm=10.0, sigma_mm=2.0)
    _, b, lift = load_case(a, case)
    voltage = lift.ravel().copy()
    voltage[free] += physics.linear_lu.solve((b.ravel() - physics.kh @ voltage)[free])
    np.savez_compressed(
        destination, force=force.reshape(len(top), -1, 3), voltage=voltage.reshape(-1, 3), top=top
    )
    return force.reshape(len(top), -1, 3), voltage.reshape(-1, 3), top


def generate(
    output,
    training=512,
    validation=128,
    test=256,
    training_seed=20261021,
    validation_seed=20261022,
    test_seed=20261101,
):
    if min(training, validation, test) < 1:
        raise ValueError("Every split must contain at least one sample.")
    if len({training_seed, validation_seed, test_seed}) != 3:
        raise ValueError("Use a distinct random seed for every split.")
    out = new_output(output)
    (out / "data").mkdir()
    physics = SparsePhysics(40)
    a = physics.a
    force, voltage, top = build_linear_basis(physics, out / "linear_basis.npz")
    protocol = dict(
        mesh_n=40,
        coarse_n=20,
        reference_rtol=1e-10,
        random_variables=dict(
            force_N=[-0.25, 0.25], voltage_V=[-100, 100], center_mm=[4, 16], sigma_mm=[1, 3]
        ),
        sampling="Four independent uniforms; fixed geometry, material and nonlinear law; no rejection.",
        nonlinear_coefficients=[0.45, 0.35, 0.06],
        state_scale_SI=STATE_SCALE.tolist(),
        splits={},
        files={},
    )
    for split, count, seed in (
        ("training", training, training_seed),
        ("validation", validation, validation_seed),
        ("test", test, test_seed),
    ):
        cases = sample_cases(count, seed, split)
        states, loads, lifts, parameters, diagnostics = [], [], [], [], []
        for i, case in enumerate(cases):
            y, info, _ = physics.solve(case)
            _, b, lift = load_case(a, case)
            states.append(y)
            loads.append(b)
            lifts.append(lift)
            parameters.append([case[k] for k in ("force_N", "voltage_V", "center_mm", "sigma_mm")])
            diagnostics.append(info)
            if (i + 1) % 32 == 0 or i + 1 == count:
                print(f"Generated {split}: {i + 1}/{count}", flush=True)
        arrays = dict(
            y=np.stack(states),
            b=np.stack(loads),
            lift=np.stack(lifts),
            parameters=np.array(parameters),
        )
        path = out / "data" / (split + ".npz")
        np.savez_compressed(path, **arrays)
        save_json(
            out / "data" / (split + "_cases.json"), dict(cases=cases, diagnostics=diagnostics)
        )
        protocol["splits"][split] = dict(size=count, seed=seed)
        protocol["files"][split] = file_hash(path)
        if split == "training":
            params = normalize_parameters(arrays["parameters"])
            initial = (
                np.einsum("bt,tnc->bnc", arrays["b"][:, top, 1], force)
                + params[:, 1, None, None] * voltage
            )
            mask = a["mask"]
            total = arrays["y"] - arrays["lift"]
            diff = arrays["y"] - initial
            field_scale = np.sqrt((total**2 * mask).sum((0, 1)) / (count * mask.sum(0)))
            output_scale = np.sqrt((diff**2 * mask).sum((0, 1)) / (count * mask.sum(0)))
            r0 = np.stack(
                [
                    (b.ravel() - physics.action(lift.ravel())).reshape(-1, 3)
                    * mask
                    / np.sqrt(a["diag"])
                    for b, lift in zip(arrays["b"], arrays["lift"])
                ]
            )
            um, pm = mask[:, :2], mask[:, 2]
            refs = [
                np.mean((r0[:, :, :2][:, um] ** 2).sum(-1)),
                np.mean((r0[:, pm, 2] ** 2).sum(-1)),
            ]
            save_json(
                out / "normalization.json",
                dict(
                    field_scale=field_scale.tolist(),
                    output_scale=output_scale.tolist(),
                    reference_squared=refs,
                    source="Training split only; residual scales use inputs only.",
                    training_sha256=protocol["files"]["training"],
                ),
            )
    protocol["basis_sha256"] = file_hash(out / "linear_basis.npz")
    protocol["normalization_sha256"] = file_hash(out / "normalization.json")
    save_json(out / "protocol.json", protocol)
    np.savez_compressed(
        out / "geometry.npz", coordinates=a["mesh"].coordinates, cells=a["mesh"].cells
    )
    return protocol
