"""Matched training and validation selection; no test-split access."""

import json
import time

import numpy as np
import torch

from lpm_gnn.io import file_hash, new_output, save_json

from .data import load_split, normalize_parameters
from .models import CompactEnv, CompactPredictor, SchedulePredictor

METHODS = ("data_gnn", "pi_gnn", "lpm_gnn")


def environment(run, device="cpu", dtype=torch.float64):
    normalization = json.loads((run / "normalization.json").read_text())
    return CompactEnv(run / "linear_basis.npz", normalization, device, dtype)


def predictor(method):
    if method == "lpm_gnn":
        return SchedulePredictor(cycles=3)
    if method not in METHODS:
        raise ValueError(method)
    return CompactPredictor(
        "CompactData" if method == "data_gnn" else "CompactPI", width=24, cycles=1
    )


def tensor_data(run, split, device, dtype):
    if split not in ("training", "validation"):
        raise ValueError("Training only reads training and validation splits.")
    data = load_split(run, split)
    data["params"] = normalize_parameters(data.pop("parameters"))
    return {key: torch.tensor(value, dtype=dtype, device=device) for key, value in data.items()}


def validate(net, env, data):
    net.eval()
    values = []
    with torch.no_grad():
        for first in range(0, len(data["y"]), 16):
            d = {k: v[first : first + 16] for k, v in data.items()}
            y = net(env, d["b"], d["lift"], d["params"])
            values.append(env.field_loss(y, d["y"]))
    squared = torch.cat(values)
    return dict(
        field_squared=float(squared.mean()), mean_combined_error=float(squared.sqrt().mean())
    )


def train(
    run, method, steps=4000, batch_size=16, seed=202610211, device="auto", checkpoint_every=100
):
    if steps < 1 or batch_size < 1 or checkpoint_every < 1:
        raise ValueError("Steps, batch size and checkpoint interval must be positive.")
    out = new_output(run / "training" / method)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    env = environment(run, device, torch.float32)
    data = tensor_data(run, "training", device, torch.float32)
    validation = tensor_data(run, "validation", device, torch.float32)
    net = predictor(method).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.00003)
    generator = torch.Generator().manual_seed(seed + 1)
    physical_weight = 0.02 if method == "pi_gnn" else 0.0
    setup = dict(
        method=method,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        physical_weight=physical_weight,
        parameters=sum(p.numel() for p in net.parameters()),
        device=device,
        data_sha256={s: file_hash(run / "data" / (s + ".npz")) for s in ("training", "validation")},
        normalization_sha256=file_hash(run / "normalization.json"),
        basis_sha256=file_hash(run / "linear_basis.npz"),
    )
    best, best_step, history = float("inf"), None, []
    started = time.perf_counter()

    def checkpoint(step, loss=None):
        nonlocal best, best_step
        score = validate(net, env, validation)
        row = dict(step=step, training_loss=loss, **score)
        if score["field_squared"] < best:
            best, best_step = score["field_squared"], step
            torch.save(
                dict(
                    state_dict=net.state_dict(),
                    method=method,
                    step=step,
                    setup=setup,
                    validation=score,
                ),
                out / "checkpoint.pt",
            )
        history.append(row)
        save_json(out / "history.json", history)
        print(method, step, score, flush=True)

    checkpoint(0)
    for step in range(1, steps + 1):
        net.train()
        ids = torch.randint(len(data["y"]), (batch_size,), generator=generator).to(device)
        d = {k: v[ids] for k, v in data.items()}
        optimizer.zero_grad(set_to_none=True)
        y = net(env, d["b"], d["lift"], d["params"])
        loss = env.field_loss(y, d["y"]).mean()
        if physical_weight:
            loss = (
                loss
                + physical_weight
                * min(step / 500, 1)
                * env.physics_loss(y, d["b"], d["lift"]).mean()
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % checkpoint_every == 0 or step == steps:
            checkpoint(step, float(loss.detach()))
    save_json(
        out / "report.json",
        dict(
            **setup,
            best_step=best_step,
            best_validation_field_squared=best,
            seconds=time.perf_counter() - started,
            checkpoint_sha256=file_hash(out / "checkpoint.pt"),
        ),
    )
