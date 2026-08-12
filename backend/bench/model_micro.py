"""Micro-benchmark: isolate model fwd/bwd/opt cost from the DataLoader.

Loads the exact Phase-2 model + checkpoint, runs fixed random batches of
batch_size 32 at 300x300, and reports GPU kernel times via CUDA events --
same measurement method as train_v2.py --profile.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v2 import CONFIG, RetinaSenseModel, FocalLoss, compute_class_weights

BATCH = 32
IMG = CONFIG["input_size"]
ITERS = 40
WARMUP = 5


def run(model, criterion, images, labels, device, opt, scaler, empty_cache=False):
    fwd, bwd, step = [], [], []
    for i in range(WARMUP + ITERS):
        if empty_cache:
            torch.cuda.empty_cache()
        e0 = torch.cuda.Event(enable_timing=True); e0.record()
        with torch.amp.autocast("cuda"):
            out = model(images)
            loss = criterion(out, labels)
        e1 = torch.cuda.Event(enable_timing=True); e1.record()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        e2 = torch.cuda.Event(enable_timing=True); e2.record()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        e3 = torch.cuda.Event(enable_timing=True); e3.record()
        torch.cuda.synchronize()
        if i >= WARMUP:
            fwd.append(e0.elapsed_time(e1))
            bwd.append(e1.elapsed_time(e2))
            step.append(e2.elapsed_time(e3))
    return fwd, bwd, step


def summ(name, xs):
    xs = sorted(xs)
    n = len(xs)
    print(f"  {name:<6} avg={sum(xs)/n:7.1f}ms p50={xs[n//2]:7.1f}ms "
          f"p95={xs[int(n*0.95)]:7.1f}ms max={xs[-1]:7.1f}ms")


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print(f"VRAM now in use: {torch.cuda.memory_allocated()/1e6:.0f} MiB")

    model = RetinaSenseModel(num_classes=CONFIG["num_classes"], dropout=CONFIG["dropout"]).to(device)
    best = CONFIG["model_dir"] / "retinasense_best.pth"
    model.load_state_dict(torch.load(str(best), map_location=device, weights_only=True))
    model.train()

    train_ds = __import__("train_v2", fromlist=["RetinalDataset"]).RetinalDataset(
        CONFIG["dataset_path"] / "train", CONFIG["class_names"])
    cw = compute_class_weights(train_ds).to(device)
    criterion = FocalLoss(alpha=cw, gamma=CONFIG["focal_gamma"])

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    images = torch.randn(BATCH, 3, IMG, IMG, device=device)
    labels = torch.randint(0, CONFIG["num_classes"], (BATCH,), device=device)

    print(f"\nFixed batch {BATCH}x3x{IMG}x{IMG}, {WARMUP} warmup + {ITERS} timed iters")
    print("Model-only (no DataLoader):")
    fwd, bwd, step = run(model, criterion, images, labels, device, opt, scaler)
    summ("fwd", fwd); summ("bwd", bwd); summ("opt", step)
    total = sum(fwd) + sum(bwd) + sum(step)
    n = len(fwd)
    print(f"  per-batch total: {total/n:.0f}ms -> {BATCH/(total/n/1e3):.1f} img/s "
          f"({BATCH*n/(total/1e3):.0f} img/s sustained)")

    print("\nWith torch.cuda.empty_cache() between iters (allocator stress test):")
    fwd, bwd, step = run(model, criterion, images, labels, device, opt, scaler, empty_cache=True)
    summ("fwd", fwd); summ("bwd", bwd); summ("opt", step)

    print("\nWith torch.compile (if available):")
    try:
        mc = torch.compile(model)
        opt2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, mc.parameters()), lr=5e-5, weight_decay=0.01)
        scaler2 = torch.amp.GradScaler("cuda")
        t0 = time.perf_counter()
        fwd, bwd, step = run(mc, criterion, images, labels, device, opt2, scaler2)
        print(f"  (compile+run took {time.perf_counter()-t0:.1f}s)")
        summ("fwd", fwd); summ("bwd", bwd); summ("opt", step)
    except Exception as e:
        print(f"  compile unavailable: {type(e).__name__}: {e}")

    print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated()/1e6:.0f} MiB")


if __name__ == "__main__":
    main()
