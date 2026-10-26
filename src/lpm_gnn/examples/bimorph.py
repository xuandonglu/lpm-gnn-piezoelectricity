"""Generate bimorph actuation, blocked force, short/open-circuit sensing."""

import argparse

import numpy as np
import torch

from lpm_gnn.io import new_output, require_convergence, save_json
from lpm_gnn.physics import bimorph as bm
from lpm_gnn.reference.planar_fem import PiezoMaterial, pzt4, q4_reference, solve_coupled


def run(grids, output, plot=False):
    out = new_output(output)
    torch.set_num_threads(1)
    rows, previous, reciprocity = [], {}, []
    for ny in sorted(set(grids)):
        mesh = bm.geometry(ny)
        graph = bm.make_graph(mesh)
        theta = graph.initial_theta
        material = pzt4()

        def layered(xy):
            return PiezoMaterial(
                material.c,
                np.sign(xy[1] - bm.H / 2) * material.e,
                material.kappa,
                "antiparallel PZT-4",
            )

        matrix = q4_reference(mesh, layered, bm.WIDTH)
        states, case_rows = {}, {}
        for case in ("actuator_free", "actuator_blocked", "sensor_short", "sensor_open"):
            constraints, rhs, boundary, prescribed = bm.boundary(mesh, case)
            tolerance = 1e-8 if ny >= 32 else 1e-10
            problem = bm.MessageEquilibrium(
                graph,
                constraints,
                rtol=tolerance,
                maxiter=1800,
                restart=256,
                preconditioner_factory=lambda p, t, s, a: bm.BimorphVCycle(p, t, s, a, case, ny),
            )
            initial = None
            if case == "sensor_open" and ny >= 32:
                # Exact linear one-port condensation, then residual polishing.
                factor = (
                    -case_rows["sensor_short"]["lpm"]["charge_C"]
                    / case_rows["actuator_free"]["lpm"]["charge_C"]
                )
                guess = states["sensor_short"] + factor * states["actuator_free"]
                initial = constraints.reduce_state(guess - constraints.lift.reshape(-1, 3))
            elif case in previous:
                old_ny, coarse = previous[case]
                full = coarse.reshape(old_ny + 1, 10 * old_ny + 1, 3).permute(2, 0, 1)[None]
                guess = (
                    torch.nn.functional.interpolate(
                        full, size=(ny + 1, 10 * ny + 1), mode="bilinear", align_corners=True
                    )[0]
                    .permute(1, 2, 0)
                    .reshape(-1, 3)
                )
                initial = constraints.reduce_state(guess - constraints.lift.reshape(-1, 3))
            state, info = problem.solve_detached(
                theta, torch.tensor(rhs, dtype=torch.float64), initial=initial
            )
            require_convergence(info)
            action = graph.action(state, theta).detach().numpy()
            ref = solve_coupled(
                matrix, mesh, prescribed, rhs, electrode_groups=(boundary["outer"],)
            )["state"]
            ref_action = (matrix @ ref.ravel()).reshape(mesh.n, 3)
            values = state.numpy()
            row = dict(
                ny=ny,
                nx=10 * ny,
                particles=mesh.n,
                case=case,
                lpm=bm.metrics(values, action, boundary, rhs, graph),
                fem=bm.metrics(ref, ref_action, boundary, rhs),
                solver=info,
                displacement_relative_L2=float(
                    np.linalg.norm(values[:, :2] - ref[:, :2]) / np.linalg.norm(ref[:, :2])
                ),
                potential_relative_L2=float(
                    np.linalg.norm(values[:, 2] - ref[:, 2]) / max(np.linalg.norm(ref[:, 2]), 1e-30)
                ),
            )
            states[case] = state
            previous[case] = (ny, state)
            case_rows[case] = row
            rows.append(row)
            np.savez_compressed(
                out / f"ny{ny}_{case}.npz",
                coordinates=mesh.coordinates,
                cells=mesh.cells,
                state=values,
                fem_state=ref,
                rhs=rhs,
            )
            save_json(out / f"ny{ny}_{case}.json", row)
            if plot:
                from lpm_gnn.plotting import planar_fields

                planar_fields(mesh.coordinates, values, ref, out / f"ny{ny}_{case}.png", mesh.cells)
            print(f"ny={ny}, {case}: tip displacement={row['lpm']['tip_mean_m']:.6g} m", flush=True)
        actuator = case_rows["actuator_free"]["lpm"]["tip_mean_m"] / bm.VOLTAGE
        sensor = case_rows["sensor_short"]["lpm"]["charge_C"] / bm.FORCE
        reciprocity.append(
            dict(
                ny=ny,
                displacement_per_voltage=actuator,
                charge_per_force=sensor,
                relative_difference=abs(actuator - sensor) / max(abs(actuator), abs(sensor)),
            )
        )
    save_json(
        out / "report.json",
        dict(
            cases=rows,
            reciprocity=reciprocity,
            length_m=bm.L,
            height_m=bm.H,
            width_m=bm.WIDTH,
            voltage_V=bm.VOLTAGE,
            force_N=bm.FORCE,
            poling="antiparallel",
        ),
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ny", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--output", default="outputs/bimorph")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run(args.ny, args.output, args.plot)


if __name__ == "__main__":
    main()
