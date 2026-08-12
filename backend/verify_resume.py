"""Read-only audit of the Phase-2 resume state before launching training.

Verifies that retinasense_train_state_phase2.pt is complete and consistent
with training_log_phase2.csv, retinasense_best.pth and training_results.json,
then prints the resume report in the required format. Never modifies any file.

Usage:  python -u verify_resume.py
Exit code 0 = consistent (safe to launch), 1 = problems found.
"""

import csv
import json
import sys
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parent / "app" / "ai" / "models"
CSV_PATH = MODEL_DIR / "training_log_phase2.csv"
STATE_PATH = MODEL_DIR / "retinasense_train_state_phase2.pt"
BEST_PATH = MODEL_DIR / "retinasense_best.pth"
RESULTS_PATH = MODEL_DIR / "training_results.json"

EXPECTED_BATCH = 32


def main() -> int:
    problems = []
    notes = []
    warnings = []

    # ------------------------------------------------------------- CSV
    if not CSV_PATH.exists():
        print("FATAL: training_log_phase2.csv missing")
        return 1
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("FATAL: training_log_phase2.csv is empty")
        return 1
    last_epoch = int(rows[-1]["epoch"])
    csv_best = max(float(r["val_acc"]) for r in rows)
    csv_best_epoch = max(int(r["epoch"]) for r in rows if float(r["val_acc"]) == csv_best)
    last_lr_csv = float(rows[-1]["lr"])
    resume_epoch = last_epoch + 1
    print(f"[CSV ] {len(rows)} rows, last epoch {last_epoch}, "
          f"best val_acc {csv_best:.4f} (epoch {csv_best_epoch}), last lr {last_lr_csv:.3e}")

    # ------------------------------------------------------------- JSON
    json_best = None
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                json_best = float(json.load(f).get("phase2_best", 0.0))
            print(f"[JSON] phase2_best (exact) = {json_best!r}")
        except Exception as e:
            problems.append(f"training_results.json unreadable: {e}")
    else:
        notes.append("training_results.json missing (exact best_acc float unavailable)")

    # ------------------------------------------------------------- State file
    if not STATE_PATH.exists():
        print("\nRESULT: state file MISSING -> optimizer/scheduler/scaler/RNG "
              "cannot be restored exactly. Safe resume only (fresh optim/sched).")
        return 1
    try:
        st = torch.load(str(STATE_PATH), map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"FATAL: state file unreadable: {e}")
        return 1
    if not isinstance(st, dict):
        print("FATAL: state file is not a dict")
        return 1

    state_epoch = st.get("epoch", "?")
    state_best = st.get("best_acc")
    state_batch = st.get("batch_size")
    state_wait = st.get("wait")
    has_rng = all(k in st for k in ("rng_torch", "rng_cuda", "rng_numpy"))
    has_opt = isinstance(st.get("optimizer"), dict)
    has_sched = isinstance(st.get("scheduler"), dict)
    has_scaler = isinstance(st.get("scaler"), dict)
    opt_groups = len(st["optimizer"]["param_groups"]) if has_opt else 0
    opt_tensors = sum(len(v) for v in st["optimizer"].get("state", {}).values()) if has_opt else 0
    sched = st["scheduler"] if has_sched else {}
    sched_steps = sched.get("_step_count", "?")
    sched_epochs = sched.get("last_epoch", "?")
    sched_lr = sched.get("_last_lr")
    sched_lr0 = sched_lr[0] if isinstance(sched_lr, list) and sched_lr else None
    sc = st["scaler"] if has_scaler else {}
    sc_scale = sc.get("scale")
    sc_growth = sc.get("_growth_tracker")

    print(f"[STATE] file exists, state epoch={state_epoch}")
    print(f"[STATE] keys: {sorted(k for k in st if k != 'rng_cuda')} + rng_cuda={st.get('rng_cuda') is not None}")
    print(f"[STATE] optimizer: {opt_groups} param group(s), {opt_tensors} state tensors "
          f"(AdamW moments + step counters)")
    print(f"[STATE] scheduler: _step_count={sched_steps}, last_epoch={sched_epochs}, "
          f"last_lr={sched_lr0:.6e}" if sched_lr0 is not None else
          f"[STATE] scheduler: _step_count={sched_steps}, last_epoch={sched_epochs}, last_lr=??")
    print(f"[STATE] scaler: scale={sc_scale}, growth_tracker={sc_growth}")
    print(f"[STATE] best_acc={state_best!r}, batch_size={state_batch}, wait={state_wait}")

    # ------------------------------------------------------------- Consistency checks
    if has_opt and opt_tensors == 0:
        problems.append("optimizer state is empty (no moments) -> resume would lose AdamW moments")
    if has_sched and not isinstance(sched_steps, int):
        problems.append("scheduler _step_count missing/not an int")
    if not has_rng:
        warnings.append("RNG states missing from state file (legacy save). Fallback: "
                        "deterministic re-seed from CONFIG seed -> sampler order not bit-exact, "
                        "otherwise safe. Will become exact after the next BEST save.")
    if not has_scaler:
        problems.append("scaler state missing from state file")
    if state_batch is None or int(state_batch) != EXPECTED_BATCH:
        problems.append(f"batch_size in state = {state_batch}, expected {EXPECTED_BATCH}")
    else:
        print(f"[OK  ] batch_size == {EXPECTED_BATCH} (exact batch reused, no probe)")
    if state_best is None:
        problems.append("best_acc missing from state file (would fall back to CSV-rounded 4-dp value)")
    else:
        state_best = float(state_best)
        print(f"[OK  ] state best_acc {state_best!r} matches JSON exact "
              f"{'YES' if json_best is not None and state_best == json_best else 'NO'}")
        if json_best is not None and abs(state_best - json_best) > 1e-12:
            problems.append(f"state best_acc {state_best!r} != JSON {json_best!r}")
        if abs(state_best - csv_best) > 5e-5:
            problems.append(f"state best_acc {state_best:.4f} != CSV max {csv_best:.4f}")
    if state_wait is None:
        # Legacy state file without the wait counter. Safe fallback: 0.
        # Provably correct here because state epoch == CSV best epoch, so the
        # early-stop counter had just been reset to 0 when the state was saved.
        warn_wait = "wait counter missing from state file (legacy save). Fallback 0: "
        if state_epoch == csv_best_epoch:
            warn_wait += "correct (state epoch is the CSV best epoch, wait was just reset)"
        else:
            warn_wait += "NOT verifiable -> conservative but not provably exact"
        warnings.append(warn_wait)
    else:
        state_wait = int(state_wait)
        expect_wait = 0 if state_epoch == csv_best_epoch else None
        print(f"[OK  ] wait={state_wait} (expected {expect_wait} "
              f"since state epoch {state_epoch} is {'best' if expect_wait == 0 else 'not best'})")
        if expect_wait is not None and state_wait != expect_wait:
            problems.append(f"wait={state_wait} inconsistent with best epoch {csv_best_epoch}")
    if state_epoch == last_epoch:
        print(f"[OK  ] state epoch == CSV last epoch ({last_epoch}) -> state is current")
    elif isinstance(state_epoch, int) and state_epoch < last_epoch:
        problems.append(f"state file is STALE (epoch {state_epoch} < CSV last {last_epoch}); "
                        "state does not cover the latest completed epoch")
    elif isinstance(state_epoch, int) and state_epoch > last_epoch:
        problems.append(f"state epoch {state_epoch} ahead of CSV last {last_epoch} (weird)")
    if sched_lr0 is not None:
        # CSV stores round(lr, 8) (see train_v2.py row["lr"]); state has the exact float.
        if round(sched_lr0, 8) == last_lr_csv:
            print(f"[OK  ] scheduler last_lr {sched_lr0:.6e} consistent with CSV last-row "
                  f"lr {last_lr_csv} (same optimizer step; CSV is round(lr,8))")
        else:
            problems.append(f"scheduler last_lr {sched_lr0:.6e} != CSV last lr {last_lr_csv}")

    # ------------------------------------------------------------- Best weights
    if BEST_PATH.exists():
        try:
            sd = torch.load(str(BEST_PATH), map_location="cpu", weights_only=True)
            print(f"[OK  ] retinasense_best.pth loads: {len(sd)} tensors, "
                  f"total {sum(t.numel() for t in sd.values()):,} params")
        except Exception as e:
            problems.append(f"retinasense_best.pth unreadable: {e}")
    else:
        problems.append("retinasense_best.pth missing")

    # ------------------------------------------------------------- Report
    opt_state = "restored" if has_opt and opt_tensors > 0 else "unavailable"
    sched_state = "restored" if has_sched and isinstance(sched_steps, int) else "unavailable"
    scaler_state = "restored" if has_scaler and sc_scale is not None else "unavailable"
    wait_state = "restored" if state_wait is not None else "unavailable"
    rng_state = "restored" if has_rng else "unavailable"

    print()
    print("==== Resume verification report ====")
    print(f"Resume epoch: {resume_epoch}")
    print(f"Batch size: {state_batch if state_batch is not None else EXPECTED_BATCH}")
    print(f"Best accuracy: {state_best if state_best is not None else csv_best:.4f}")
    print(f"Optimizer state: {opt_state}")
    print(f"Scheduler state: {sched_state}")
    print(f"GradScaler state: {scaler_state}")
    print(f"Early-stop wait: {wait_state}")
    print(f"RNG state: {rng_state}")
    print()

    if problems:
        print("PROBLEMS FOUND (blocking):")
        for p in problems:
            print(f"  - {p}")
        print("Do NOT launch training until these are resolved.")
        return 1
    for w in warnings:
        print(f"WARNING (non-blocking): {w}")
    for n in notes:
        print(f"NOTE: {n}")
    print("All checks passed -> resume is safe. Verdict: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
