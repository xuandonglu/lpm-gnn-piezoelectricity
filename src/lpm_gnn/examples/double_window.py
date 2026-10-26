"""Generate the aligned double-window response and an independent Q4 reference."""

import argparse

import numpy as np
import torch

from lpm_gnn.io import new_output, require_convergence, save_json
from lpm_gnn.physics.piezo_frame_message_multigrid import make_frame_vcycle_factory
from lpm_gnn.physics.piezo_frame_message_problem import (
    frame_constraints,
    frame_diagnostics,
    frame_load,
    frame_pad_weights,
    make_frame_geometry,
    relative_norm,
)
from lpm_gnn.physics.piezo_message_graph import PiezoMessageGraph
from lpm_gnn.physics.piezo_message_solve import MessageEquilibrium
from lpm_gnn.reference.planar_fem import pzt4, q4_reference, solve_coupled


def run(grids, output, device="cpu", plot=False):
    out = new_output(output)
    torch.set_num_threads(1)
    rows = []
    for n in grids:
        mesh = make_frame_geometry(n)
        graph = PiezoMessageGraph(mesh, device=device)
        theta = graph.initial_theta
        # This matrix belongs only to the independent continuum reference.
        fem_matrix = q4_reference(mesh, pzt4(), thickness=0.001)
        for name, force, voltage in (("force", 0.01, 0.0), ("voltage", 0.0, 1.0)):
            constraints = frame_constraints(mesh, V=voltage, device=device)
            problem = MessageEquilibrium(
                graph,
                constraints,
                rtol=1e-10,
                restart=256,
                maxiter=1600,
                preconditioner_factory=make_frame_vcycle_factory(coarse_iterations=32),
            )
            rhs = frame_load(mesh, force, device=device)
            state, info = problem.solve_detached(theta, rhs)
            require_convergence(info)
            fixed = np.flatnonzero(constraints.full_to_free.cpu().numpy() < 0)
            lift = constraints.lift.cpu().numpy().ravel()
            reference = solve_coupled(
                fem_matrix, mesh, {int(i): float(lift[i]) for i in fixed}, rhs.cpu().numpy()
            )["state"]
            target = torch.tensor(reference, dtype=torch.float64, device=device)
            row = dict(
                n=n,
                particles=mesh.n,
                control=name,
                displacement_relative_L2=relative_norm(state[:, :2], target[:, :2]),
                potential_relative_L2=relative_norm(state[:, 2], target[:, 2]),
                solver=info,
                diagnostics=frame_diagnostics(problem, theta, state, rhs, force, voltage),
            )
            row["pad_displacement_m"] = float(frame_pad_weights(mesh) @ state[:, 1].cpu().numpy())
            values = state.cpu().numpy()
            np.savez_compressed(
                out / f"n{n}_{name}.npz",
                coordinates=mesh.coordinates,
                cells=mesh.cells,
                state=values,
                fem_state=reference,
                rhs=rhs.cpu().numpy(),
            )
            save_json(out / f"n{n}_{name}.json", row)
            if plot:
                from lpm_gnn.plotting import planar_fields

                planar_fields(
                    mesh.coordinates, values, reference, out / f"n{n}_{name}.png", cells=mesh.cells
                )
            rows.append(row)
            print(
                f"n={n}, {name}: displacement error={row['displacement_relative_L2']:.6g}, potential error={row['potential_relative_L2']:.6g}",
                flush=True,
            )
    save_json(
        out / "report.json",
        dict(
            grids=grids,
            controls=["force", "voltage"],
            cases=rows,
            reference="independent continuum Gaussian-Q4 FEM",
        ),
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument("--output", default="outputs/double_window")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run(args.grids, args.output, args.device, args.plot)


if __name__ == "__main__":
    main()
