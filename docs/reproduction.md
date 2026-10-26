# Reproduction guide

## Paper map

| Paper content | Implementation |
| --- | --- |
| Planar electric enthalpy and particle interactions | `physics/piezo_message_graph.py` |
| Rotated material/poling axes and boundary completion | `physics/rotated_lattice.py`, `physics/fixed_frame.py` |
| Constrained graph equilibrium and FGMRES | `physics/piezo_message_solve.py`, `physics/piezo_graph_iterations.py` |
| Auxiliary block V-cycle | `physics/piezo_frame_message_multigrid.py`, `physics/piezo_message_multigrid.py` |
| Double-window linear verification | `examples/double_window.py`, `examples/orientation.py` |
| Learned nonlinear updates | `nonlinear/models.py`, `nonlinear/multigrid.py` |
| Bimorph application | `examples/bimorph.py`, `physics/bimorph.py` |
| Harmonic disk and experimental comparison | `examples/disk.py`, `physics/piezo_axisym_*.py` |

Paths in this table are relative to `src/lpm_gnn`. The source lineage and original source hashes appear in [source_manifest.json](source_manifest.json). The release removes machine-specific paths, historical job orchestration and dependencies on previously saved numerical fields. Its generators reconstruct those fields from the physical inputs.

## Conventions

- SI units are used by the physical operators.
- Planar state order is `(ux, uy, phi)`; engineering strain is `(exx, eyy, gamma_xy)`.
- Electric field is `-grad(phi)`. Electrode free charge is minus the internal electrical reaction.
- The coupled electric-enthalpy Hessian is a saddle operator. Flexible GMRES solves the coupled system; the auxiliary positive-block operator is used for preconditioning.
- Harmonic quantities use `exp(+i omega t)` and `Y = i omega Q / V`.
- Independent FEM references use continuum constitutive tensors and Gaussian integration. Their assembled matrices are separate from the message-based physical solver.

## Double-window frame

The outer domain is 20 by 20 mm with thickness 1 mm. Windows occupy `x = 5-9 mm` and `11-15 mm`, both over `y = 5-15 mm`. The bottom edge is clamped and grounded. The top electrode is equipotential. Separate controls apply 0.01 N over the top central 4 mm pad, or 1 V across the electrodes.

The PZT-4 plane-strain constants are:

| Quantity | Value |
| --- | --- |
| `C11, C22, C12, C66` | `139, 115, 74.3, 25.6 GPa` |
| `e15, e31, e33` | `12.7, -5.2, 15.1 C/m^2` |
| `kappa11/epsilon0, kappa22/epsilon0` | `730, 635` |

These constants are stated in the source of `reference/planar_fem.py`, with the original material-table reference. The aligned grid sequence is `n = 20, 40, 80`. Linear equilibrium uses relative tolerance `1e-10`.

The orientation study rotates the material axes, lattice and poling direction by 30 or 60 degrees. Outer geometry, electrodes and applied loads remain fixed. The particle sequence is `n = 20, 40, 80, 160`, and the independent FEM reference uses `n = 320`. Missing directional differences use same-side quadratic reconstruction. Clipped Voronoi areas cover the exact physical domain. Sparse expansion/LU evaluates refinement; a separate matrix-free V-cycle check tests the graph iteration.

## Illustrative nonlinear extension

The geometry is the same double-window frame with `n = 40`. A fixed reversible enthalpy correction uses the normalized local axial strain and electric field. `nonlinear/physics.py::nonlinear_parts` implements its analytic first and second derivatives; the mechanical, dielectric and mixed coefficients are `0.45, 0.35, 0.06`. The smooth correction uses the prescribed tanh functions and the same material constants for every split.

The independent random inputs are:

| Input | Uniform range |
| --- | --- |
| Total force | `[-0.25, 0.25] N` |
| Top electrode voltage | `[-100, 100] V` |
| Gaussian traction centre | `[4, 16] mm` |
| Gaussian traction width | `[1, 3] mm` |

| Split | Cases | NumPy RNG seed |
| --- | --- | --- |
| Training | 512 | 20261021 |
| Validation | 128 | 20261022 |
| Independent test | 256 | 20261101 |

The generator uses `numpy.random.default_rng`, generates every prescribed sample, and solves each reference with sparse direct Newton and residual line search to relative residual `1e-10`. Raw state arrays are dimensionless with channel scales `[2e-8 m, 2e-8 m, 100 V]`. Physical-state prediction exports multiply these scales back in.

All compared methods start from the same exact linear field, formed using cached unit-force and voltage basis responses. Setup constructs the basis without nonlinear response labels. Training-only labels determine field and output scaling. Fixed mechanical/electrical RMS scales for PI training use boundary-lift residuals of training inputs only.

Data-GNN and PI-GNN use width 24, three coarse graph propagation steps and a fine local decoder (11,382 parameters each). The LPM-GNN controller uses width 16 and one coarse graph propagation pass to produce a three-stage, three-channel residual-weight schedule (5,516 parameters). Weights are bounded by `0.8 + 0.35 tanh(z)`. Each physical stage recomputes the nonlinear residual and uses a fully coupled `40 -> 20 -> 40` Galerkin cycle, with one pre/post Jacobi sweep and damping 0.65. The coarse solve is an exact reusable factorization. Training represents the same coarse inverse densely for differentiation on the accelerator; the CPU inference backend uses sparse LU.

Training uses seed `202610211`, 4,000 Adam steps, batch size 16 and cosine learning-rate decay from `0.001` to `0.00003`. PI-GNN's physical-loss weight is `0.02`, ramped over 500 steps. Checkpoints are selected by validation field error. Every 100-step checkpoint and the final step are evaluated. The independent test split is read only by the evaluation command.

The conventional baseline uses an exact nonlinear tangent, Newton line search and FGMRES with a coupled Galerkin V-cycle. Its Jacobi damping is 0.3. Evaluation reports strict numerical convergence and the first recorded numerical state reaching each network's combined field error. Matched budgets are replayed without reference-error calculations inside the timed routine. Reported online times include common prior application and prediction/iteration, with one BLAS thread; offline training and factorization are outside this online timing.

## Bimorph

The two-layer beam is 10 mm long, 1 mm high and 1 mm wide. Identical PZT-4 layers have antiparallel fixed poling. The left edge is mechanically clamped. The lower electrode is grounded, and the upper electrode is driven or floating. Actuation uses 10 V; sensing uses a uniformly distributed 0.001 N tip-edge force. `ny = 4, 8, 16, 32`, with `nx = 10 ny`, gives the refinement sequence.

Fine-grid runs use interpolated particle fields as initial guesses. At `ny = 32`, open-circuit sensing uses exact linear electrode condensation followed by residual polishing. The independently assembled layered Q4 reference uses the same supports and electrode conditions.

## Axisymmetric disk

The radial PZ27 specimen has radius 10 mm, height 1 mm and density 7700 kg/m^3. The graph models a free body with full-face ideal electrodes and 1 V harmonic drive. The nominal tensor shape is the PZ27 column of the September 2023 [manufacturer workbook](https://www.ferropermpiezoceramics.com/wp-content/uploads/2023/09/Ferroperm_materials_data_sep2023.xlsx), sheet `Pzxx`: elastic cells G48-G52, piezoelectric cells G21-G23 and fixed-strain relative permittivities G6-G7. The numeric constants are embedded in `examples/disk.py::NOMINAL`; they specify the nominal prior rather than the historical specimen's measured tensor.

The external [radial example archive](https://wp.icmm.csic.es/gffm/wp-content/uploads/sites/34/2023/04/hand_rad.zip) contains the measured data used by the paper. The numeric `raw_data.txt` file has SHA-256 `6ec9321da439f5ba5c2bb32bf6e479f17ea8b8dac0af4cd0350a8ae305c59c97`. Its three columns are frequency in kHz, admittance magnitude in S, and phase in rad. The complex convention is `Y = magnitude * exp(i phase)`. The 801 frequencies span 78-158 kHz at 0.1 kHz increments. The source explains the [coefficient-determination procedure](https://wp.icmm.csic.es/gffm/determination-of-coefficients/); its associated full-tensor methodology is described in [doi:10.1111/j.1551-2916.2004.00209.x](https://doi.org/10.1111/j.1551-2916.2004.00209.x).

Calibration uses frequencies `110, 111.2, 112.5, 113.8, 115, 116.2, 117.5, 118.8, 120 kHz`, with the complex log-magnitude/phase objective. Six shared coordinates specify elastic, piezoelectric and dielectric scales, mechanical and dielectric losses, and a bounded passive shared piezoelectric loss ratio. The initial coordinates, bounds and step caps are explicit in `examples/disk.py::fit`. Default calibration/prediction grids are `nr = 40, nz = 4`; use `--nr 80 --nz 8` for the frozen-coefficient refinement check.

Measured observations are external and cannot be generated by this numerical model. The model can generate predicted spectra and fields. The nominal Q2 verification is a separate numerical comparison with prescribed coefficients and losses, independent of measured-spectrum calibration.
