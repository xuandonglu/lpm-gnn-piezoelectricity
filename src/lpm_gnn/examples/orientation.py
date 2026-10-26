"""Fixed-domain material/lattice rotation and independent rotated-tensor FEM."""

import argparse

import numpy as np
import torch

from lpm_gnn.io import new_output, require_convergence, save_json
from lpm_gnn.physics.fixed_frame import (
    FixedFrameGraph,
    MessageConstraints,
    assemble_graph,
    fixed_boundary_data,
    relative,
    run_case,
    sparse_cases,
)
from lpm_gnn.physics.piezo_message_solve import MessageEquilibrium
from lpm_gnn.physics.rotated_multigrid import FixedFrameVCycle


def iterative_check(n, angle, out):
    graph = FixedFrameGraph(n, angle)
    bottom, top, pad = fixed_boundary_data(graph)
    rows = []
    for name, force, voltage in (("force", 0.01, 0.0), ("voltage", 0.0, 1.0)):
        prescribed = {3 * int(i) + k: 0.0 for i in bottom for k in range(3)}
        prescribed.update({3 * int(i) + 2: voltage for i in top})
        constraints = MessageConstraints(graph.n, prescribed)
        problem = MessageEquilibrium(
            graph,
            constraints,
            rtol=1e-9,
            maxiter=500,
            restart=80,
            preconditioner_factory=lambda p, t, s, a: FixedFrameVCycle(p, t, s, a, n, angle),
        )
        rhs = torch.zeros((graph.n, 3), dtype=torch.float64)
        rhs[:, 1] = force * torch.tensor(pad)
        state, info = problem.solve_detached(graph.initial_theta, rhs)
        require_convergence(info)
        # Reference expansion occurs after the graph iteration, not inside it.
        direct, _ = sparse_cases(assemble_graph(graph), graph.coordinates.numpy(), bottom, top, pad)
        row = dict(
            control=name,
            solver=info,
            displacement_LU_parity=relative(state.numpy()[:, :2], direct[name][:, :2]),
            potential_LU_parity=relative(state.numpy()[:, 2], direct[name][:, 2]),
        )
        rows.append(row)
    save_json(out / f"iterative_n{n}_a{angle:g}.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", nargs="+", type=int, default=[20, 40, 80, 160])
    parser.add_argument("--angles", nargs="+", type=float, default=[30.0, 60.0])
    parser.add_argument("--reference-n", type=int, default=320)
    parser.add_argument("--output", default="outputs/orientation")
    parser.add_argument(
        "--iterative-check",
        action="store_true",
        help="Also compare a matrix-free V-cycle solve with the exact sparse expansion on each grid.",
    )
    args = parser.parse_args()
    out = new_output(args.output)
    torch.set_num_threads(1)
    rows = []
    for n in args.grids:
        for angle in args.angles:
            rows.append(run_case(n, angle, out, args.reference_n))
            if args.iterative_check:
                iterative_check(n, angle, out)
    save_json(
        out / "report.json",
        dict(cases=rows, reference_n=args.reference_n, fixed_geometry_loads_and_electrodes=True),
    )


if __name__ == "__main__":
    main()
