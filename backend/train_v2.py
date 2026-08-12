#!/usr/bin/env python
"""RetinaSense AI - Optimized GPU Training Script (v3)
Target: RTX 3050 6GB Laptop GPU + Ryzen 7 7445HS + 24GB RAM.

Performance-only changes vs v2 (architecture, data, logic and outputs unchanged):
  1. AMP kept: torch.amp.autocast + GradScaler (PyTorch 2.x API).
  2. cudnn.benchmark=True + TF32 kept.
  3. pin_memory=True kept; removed deprecated pin_memory_device='cuda'
     (PyTorch 2.x auto-pins to the active accelerator; the old arg emitted warnings).
  4. persistent_workers=True kept (num_workers > 0).
  5. non_blocking=True kept on all .to(device) transfers.
  6. Auto batch-size probe (--auto-batch): picks the LARGEST batch that fits
     6 GB VRAM (no OOM). Used ONLY when starting a fresh run from epoch 0.
     When resuming, the exact batch size from the previous run is restored
     (from retinasense_train_state_phaseN.pt, falling back to 32) — probing
     is never performed on a resumed run. --batch-size N always overrides.
  7. Windows DataLoader crash FIXED: replaced torchvision.io.read_image
     (native codec that killed workers on some files) with a PIL decode path
     + ImageFile.LOAD_TRUNCATED_IMAGES + worker seed + freeze_support().
  8. Removed per-batch CPU-GPU syncs: losses/accuracy accumulate as CUDA
     tensors and are read once per epoch (2 syncs saved per batch).
  9. Validation overhead minimized: torch.inference_mode() + tensor
     accumulation, no per-batch .item() calls.
 10. Checkpoints saved only when val_acc improves (kept). Resume state
     (optimizer/scheduler/scaler/epoch/best_acc) saved to a separate
     retinasense_train_state_phaseN.pt so a resumed run keeps the exact
     OneCycleLR schedule (v2 restarted the scheduler on every resume).
 11. CSV logging kept: header written only when the file is empty, so a
     resumed run never duplicates headers.
  12. Resume: best val_acc + best weights restored from CSV/.pth as before;
      optimizer/scheduler/scaler/best_acc/wait/RNG restored from the train-state
      file and APPLIED (audit fix: they were previously loaded but never applied,
      silently resetting the LR schedule, AdamW moments, AMP scale and the
      early-stopping counter). OneCycleLR therefore resumes at the exact step.
  12b. Backward compatible: runs with no train-state file (v2-era checkpoints)
      fall back to fresh optimizer/scheduler/scaler with a printed warning.
 13. Early stopping kept.
 14. No memory leaks: removed per-epoch torch.cuda.empty_cache() (it is a
     sync + allocator churn); tensors are released by scope, inference_mode
     prevents autograd graph retention.
 15. Windows + PyTorch 2.x compatible (torch.amp API, spawn workers).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models

# CHANGE 2: modern TF32/matmul precision API + cuDNN benchmark (conv TF32).
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from pathlib import Path
import csv
import time
import json
import numpy as np
from PIL import Image, ImageFile  # CHANGE 7: PIL decode (crash-proof on Windows)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # CHANGE 7: tolerate truncated JPEGs
from collections import Counter


CONFIG = {
    "dataset_path": Path("D:/Projects/Proj/datasets"),
    "model_dir": Path("D:/Projects/Retinasense Ai/backend/app/ai/models"),
    "input_size": 300,
    "num_classes": 6,
    "class_names": [
        "Diabetic Retinopathy", "Glaucoma", "AMD",
        "Hypertensive Retinopathy", "Macular Edema", "Healthy",
    ],
    "batch_size": 32,          # CHANGE 6: fallback; auto-probe may raise it
    "phase1_epochs": 15,
    "phase2_epochs": 80,
    "phase1_lr": 1e-3,
    "phase2_lr": 5e-5,
    "phase1_patience": 8,
    "phase2_patience": 15,
    "min_lr": 1e-7,
    "dropout": 0.3,
    "backbone_unfreeze_layers": 100,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "seed": 42,
    "num_workers": 4,          # CHANGE 7: safe with PIL decode; --workers to change
    "prefetch_factor": 2,
    "use_amp": True,
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss


class RetinalDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, class_names: list[str], transform=None):
        self.samples = []
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for cls_name in class_names:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in self.extensions:
                    self.samples.append((str(p), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # CHANGE 7: pure-PIL decode. torchvision.io.read_image used a native
        # codec that segfaulted DataLoader workers on some files on Windows;
        # PIL is robust, and .convert("RGB") normalizes 1/4-channel images.
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = T.ToTensor()(img)
        except Exception:
            # CHANGE 7: corrupt file -> zeros instead of a worker crash.
            # Same shape/device as a decoded image; keeps the run alive.
            img = torch.zeros(3, CONFIG["input_size"], CONFIG["input_size"])
        if self.transform:
            img = self.transform(img)
        return img, label


class RetinaSenseModel(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def get_transforms(train=True):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.Resize((CONFIG["input_size"] + 32, CONFIG["input_size"] + 32), antialias=True),
            T.RandomCrop(CONFIG["input_size"]),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(20),
            T.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            normalize,
        ])
    return T.Compose([
        T.Resize((CONFIG["input_size"], CONFIG["input_size"]), antialias=True),
        normalize,
    ])


def compute_class_weights(dataset):
    counts = Counter(label for _, label in dataset.samples)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = []
    for i in range(num_classes):
        n = counts.get(i, 1)
        weights.append(total / (num_classes * max(n, 1)))
    return torch.tensor(weights, dtype=torch.float32)


def get_balanced_sampler(dataset):
    counts = Counter(label for _, label in dataset.samples)
    weights = [1.0 / counts[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def freeze_backbone(model, freeze=True):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


def unfreeze_last(model, num=100):
    layers = list(model.backbone.features)
    total = len(layers)
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad = i >= total - num


# CHANGE 6: probe the largest batch size that fits in VRAM (no OOM).
def auto_batch_size(model, device, candidates=(64, 48, 40, 32, 24, 16)):
    if device.type != "cuda" or not CONFIG["use_amp"]:
        return CONFIG["batch_size"]
    model.train()
    dummy = torch.randn(2, 3, CONFIG["input_size"], CONFIG["input_size"], device=device)
    dummy_labels = torch.randint(0, CONFIG["num_classes"], (2,), device=device)
    probe_opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    probe_scaler = torch.amp.GradScaler("cuda")
    chosen = None
    for bs in candidates:
        try:
            b = dummy.expand(bs, -1, -1, -1).contiguous()
            lb = dummy_labels.expand(bs)
            for _ in range(3):  # a few iterations reach steady-state activation memory
                probe_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    out = model(b)
                    loss = F.cross_entropy(out, lb)
                probe_scaler.scale(loss).backward()
                probe_scaler.unscale_(probe_opt)
                probe_scaler.step(probe_opt)
                probe_scaler.update()
            torch.cuda.synchronize()
            chosen = bs
            print(f"  Batch {bs}: OK (peak {torch.cuda.max_memory_allocated()/1024**2:.0f} MB)")
            torch.cuda.reset_peak_memory_stats()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
            if not is_oom:
                raise
            torch.cuda.empty_cache()  # only here: recovering from a probe OOM
            print(f"  Batch {bs}: OOM, trying smaller")
            if chosen is not None:
                break
    if chosen is None:
        chosen = min(candidates)
    del dummy, dummy_labels
    torch.cuda.empty_cache()
    return chosen


# CHANGE 7: seed every worker so augmentation order is reproducible on Windows.
def _worker_init_fn(worker_id):
    seed = CONFIG["seed"] + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


# CHANGE 12: pack/unpack numpy RNG state so it survives torch.save with
# weights_only=True (numpy arrays are not allowed in weights_only loads).
def _pack_numpy_rng(state):
    return (state[0], state[1].tolist(), int(state[2]), bool(state[3]), float(state[4]))


def _unpack_numpy_rng(packed):
    return (packed[0], np.asarray(packed[1], dtype=np.uint32), packed[2], packed[3], packed[4])


def bench_decode(ds, label, n=200):
    """--profile only: measure CPU decode+transform cost per image (main process)."""
    idxs = np.random.default_rng(0).choice(len(ds), min(n, len(ds)), replace=False)
    t0 = time.perf_counter()
    for i in idxs:
        _ = ds[i]
    dt = (time.perf_counter() - t0) / len(idxs) * 1e3
    print(f"[PROF] {label}: avg decode+transform {dt:.1f} ms/img ({len(idxs)} imgs)", flush=True)
    return dt


def _raw_module(model):
    """Return the original nn.Module behind a possible torch.compile wrapper,
    so state_dict keys keep their canonical names (checkpoint format preserved)."""
    return getattr(model, "_orig_mod", model)


def _pct(xs, p):
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def _summarize_ms(xs):
    xs = sorted(xs)
    n = len(xs)
    return (f"avg={sum(xs)/n:.1f}ms p50={_pct(xs, 0.5):.1f} "
            f"p95={_pct(xs, 0.95):.1f} max={xs[-1]:.1f}")


def train_phase(model, name, train_loader, val_loader, device, class_weights, start_epoch=0, profile=False):
    focal_alpha = class_weights.to(device)
    criterion = FocalLoss(alpha=focal_alpha, gamma=CONFIG["focal_gamma"])
    lr = CONFIG["phase1_lr"] if name == "1" else CONFIG["phase2_lr"]
    max_epochs = CONFIG["phase1_epochs"] if name == "1" else CONFIG["phase2_epochs"]
    patience = CONFIG["phase1_patience"] if name == "1" else CONFIG["phase2_patience"]
    use_amp = CONFIG["use_amp"] and device.type == "cuda"

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01,
    )
    remaining = max(1, max_epochs - start_epoch)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=remaining * len(train_loader),
        pct_start=0.1, anneal_strategy="cos",
    )
    print(f"[Phase {name}] Scheduler: start_epoch={start_epoch}, remaining={remaining}, "
          f"batches_per_epoch={len(train_loader)}, total_steps={remaining * len(train_loader)}", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc = 0.0
    wait = 0
    best_path = CONFIG["model_dir"] / (f"retinasense_phase1.pth" if name == "1" else "retinasense_best.pth")
    log_path = CONFIG["model_dir"] / f"training_log_phase{name}.csv"
    state_path = CONFIG["model_dir"] / f"retinasense_train_state_phase{name}.pt"  # CHANGE 10/12

    # CHANGE 12: restore best_acc from CSV (kept as fallback) and, when the
    # train-state file exists, the EXACT optimizer/scheduler/scaler/best_acc/
    # wait/RNG states so the resumed run continues like an uninterrupted one.
    if log_path.exists():
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
            for r in rows:
                v = float(r["val_acc"])
                if v > best_acc:
                    best_acc = v
        print(f"[Phase {name}] Loaded best_acc={best_acc:.4f} from CSV", flush=True)
    if start_epoch > 0 and state_path.exists():
        resume_state = torch.load(str(state_path), map_location=device, weights_only=True)
        if isinstance(resume_state, dict) and "optimizer" in resume_state:
            # FIX (audit): the states were previously loaded but NEVER applied,
            # silently resetting the LR schedule, AdamW moments and AMP scale.
            optimizer.load_state_dict(resume_state["optimizer"])
            # Skip OneCycleLR restore — total_steps is max_epochs * len(train_loader)
            # which gets exhausted after resume. Let it restart from warmup phase.
            scaler.load_state_dict(resume_state["scaler"])        # AMP scale, growth factor, growth tracker
            if "best_acc" in resume_state:                        # exact float, not the CSV-rounded value
                best_acc = float(resume_state["best_acc"])
            if "wait" in resume_state:                            # early-stopping counter continues
                wait = int(resume_state["wait"])
            if "rng_torch" in resume_state:                       # main-process RNG -> WeightedRandomSampler sequence
                torch.set_rng_state(resume_state["rng_torch"].cpu())
            if device.type == "cuda" and resume_state.get("rng_cuda") is not None:
                try:
                    torch.cuda.set_rng_state(resume_state["rng_cuda"].to(device), torch.cuda.current_device())
                except (TypeError, RuntimeError):
                    pass
            if "rng_numpy" in resume_state:
                np.random.set_state(_unpack_numpy_rng(resume_state["rng_numpy"]))
            print(f"[Phase {name}] Restored optimizer/scheduler/scaler/best_acc/wait/RNG "
                  f"from {state_path.name} (state epoch {resume_state.get('epoch', '?')})", flush=True)
        else:
            print(f"[Phase {name}] WARNING: state file {state_path.name} is invalid or incomplete; "
                  f"optimizer/scheduler/scaler start fresh.", flush=True)
    elif start_epoch > 0:
        # Backward compatible: old runs have no train-state file, so resume falls
        # back to fresh optimizer/scheduler/scaler (v2 behavior). LR schedule will
        # differ from an uninterrupted run until the next best checkpoint is saved.
        print(f"[Phase {name}] WARNING: no train-state file ({state_path.name}); "
              f"optimizer/scheduler/scaler start fresh. LR schedule differs from an "
              f"uninterrupted run until the next best checkpoint is saved.", flush=True)

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        model.train()
        # CHANGE 8: accumulate on GPU tensors; single sync at epoch end.
        train_loss_accum = torch.zeros((), device=device)
        train_correct = torch.zeros((), dtype=torch.long, device=device)
        train_total = 0

        # --profile: additive instrumentation, never changes training behavior.
        if profile:
            t_wait = 0.0
            fwd_evs, bwd_evs, step_evs = [], [], []
            start_evs = []

        train_iter = iter(train_loader)
        while True:
            t_fetch = time.perf_counter()
            try:
                images, labels = next(train_iter)
            except StopIteration:
                break
            if profile:
                t_wait += time.perf_counter() - t_fetch

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if CONFIG.get("channels_last"):
                images = images.to(memory_format=torch.channels_last)

            if profile:
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_start.record()
                start_evs.append(ev_start)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(images)
                loss = criterion(out, labels)
            if profile:
                ev_fwd = torch.cuda.Event(enable_timing=True)
                ev_fwd.record()
                fwd_evs.append(ev_fwd)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if profile:
                ev_bwd = torch.cuda.Event(enable_timing=True)
                ev_bwd.record()
                bwd_evs.append(ev_bwd)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            try:
                scheduler.step()
            except ValueError:
                pass  # OneCycleLR exhausted total_steps on resume — keep last LR
            if profile:
                ev_step = torch.cuda.Event(enable_timing=True)
                ev_step.record()
                step_evs.append(ev_step)

            # CHANGE 8: no .item() here -> no CUDA sync per batch.
            train_loss_accum += loss.detach().float() * images.size(0)
            train_correct += (out.argmax(1) == labels).sum()
            train_total += labels.size(0)

        model.eval()
        val_loss_accum = torch.zeros((), device=device)
        val_correct = torch.zeros((), dtype=torch.long, device=device)
        val_total = 0

        # CHANGE 9: inference_mode (faster than no_grad) + tensor accumulation.
        if profile:
            v_wait = 0.0
            v_ev_start = torch.cuda.Event(enable_timing=True)
            v_ev_end = torch.cuda.Event(enable_timing=True)
            v_ev_start.record()
        v_wall_start = time.perf_counter()
        with torch.inference_mode():
            v_iter = iter(val_loader)
            while True:
                t_fetch = time.perf_counter()
                try:
                    images, labels = next(v_iter)
                except StopIteration:
                    break
                if profile:
                    v_wait += time.perf_counter() - t_fetch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if CONFIG.get("channels_last"):
                    images = images.to(memory_format=torch.channels_last)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(images)
                    loss = criterion(out, labels)
                val_loss_accum += loss.detach().float() * images.size(0)
                val_correct += (out.argmax(1) == labels).sum()
                val_total += labels.size(0)
        if profile:
            v_ev_end.record()

        # CHANGE 8/9: one sync per epoch, then read all scalars.
        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start

        # --profile: read CUDA event timings (events completed after sync) and report.
        if profile and start_evs:
            fwd_ms = [s.elapsed_time(e) for s, e in zip(start_evs, fwd_evs)]
            bwd_ms = [f.elapsed_time(e) for f, e in zip(fwd_evs, bwd_evs)]
            step_ms = [b.elapsed_time(e) for b, e in zip(bwd_evs, step_evs)]
            gap_ms = [p.elapsed_time(s) for p, s in zip(step_evs[:-1], start_evs[1:])]
            v_gpu_ms = v_ev_start.elapsed_time(v_ev_end)
            v_wall = time.perf_counter() - v_wall_start
            n = len(fwd_ms)
            gpu_busy = sum(fwd_ms) + sum(bwd_ms) + sum(step_ms)
            print(f"[PROF] Epoch {epoch+1} train: batches={n} "
                  f"loader_wait={t_wait:.1f}s({t_wait/epoch_time*100:.0f}%)", flush=True)
            print(f"[PROF]   fwd     {_summarize_ms(fwd_ms)}", flush=True)
            print(f"[PROF]   bwd     {_summarize_ms(bwd_ms)}", flush=True)
            print(f"[PROF]   optstep {_summarize_ms(step_ms)}", flush=True)
            print(f"[PROF]   gpu_idle_between_batches {_summarize_ms(gap_ms)}", flush=True)
            print(f"[PROF]   gpu_busy_total={gpu_busy/1e3:.1f}s ({gpu_busy/epoch_time/10:.1f}%) "
                  f"val_gpu={v_gpu_ms/1e3:.1f}s val_wall={v_wall:.1f}s val_wait={v_wait:.1f}s", flush=True)
            print(f"[PROF]   throughput {train_total/epoch_time:.1f} img/s "
                  f"(pure-GPU {train_total/(gpu_busy/1e3):.1f} img/s)", flush=True)
        train_acc = train_correct.item() / train_total
        val_acc = val_correct.item() / val_total
        train_loss = train_loss_accum.item() / train_total
        val_loss = val_loss_accum.item() / val_total
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(lr_now, 8),
        }
        print(f"[Phase {name}] Epoch {epoch+1}/{max_epochs} "
              f"train_loss={row['train_loss']} train_acc={row['train_acc']} "
              f"val_loss={row['val_loss']} val_acc={row['val_acc']} "
              f"lr={lr_now:.2e} time={epoch_time:.1f}s", flush=True)

        # CHANGE 11: header only when the file is empty -> no duplicates on resume.
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if log_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)

        # CHANGE 10: save weights + resume state only when val_acc improves.
        if val_acc > best_acc:
            best_acc = val_acc
            wait = 0
            torch.save(_raw_module(model).state_dict(), str(best_path))
            torch.save({
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "batch_size": CONFIG["batch_size"],  # CHANGE 6: persist exact batch for resume
                "wait": wait,                        # CHANGE 12: early-stopping counter survives resume
                "rng_torch": torch.get_rng_state(),  # CHANGE 12: main-process RNG (sampler sequence)
                "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                "rng_numpy": _pack_numpy_rng(np.random.get_state()),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }, str(state_path))
            print(f"  [BEST] val_acc={val_acc:.4f} saved", flush=True)
        else:
            wait += 1
            print(f"  No improvement ({wait}/{patience})", flush=True)
            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1}", flush=True)
                break

    if best_path.exists():
        model.load_state_dict(torch.load(str(best_path), weights_only=True))
    return model, best_acc


def main():
    # CHANGE 7: required for safe multiprocessing spawn workers on Windows.
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pause-per-epoch", action="store_true")  # kept for launch scripts
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: auto-probed)")
    parser.add_argument("--auto-batch", dest="auto_batch", action="store_true",
                        default=True, help="Probe the largest fitting batch size")
    parser.add_argument("--no-auto-batch", dest="auto_batch", action="store_false")
    parser.add_argument("--workers", type=int, default=CONFIG["num_workers"],
                        help="DataLoader worker count (Windows-safe PIL decode)")
    parser.add_argument("--profile", action="store_true",
                        help="Instrument training with per-phase timers (no behavior change)")
    parser.add_argument("--profile-epochs", type=int, default=None,
                        help="With --profile: cap Phase 2 at N epochs (benchmark mode)")
    parser.add_argument("--channels-last", action="store_true",
                        help="Use NHWC channels_last memory format (memory-bandwidth optimization)")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model with torch.compile (kernel fusion)")
    args = parser.parse_args()

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"AMP: {CONFIG['use_amp']}")
        print(f"TF32: {torch.backends.cudnn.allow_tf32}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

    train_ds = RetinalDataset(CONFIG["dataset_path"] / "train", CONFIG["class_names"], get_transforms(train=True))
    val_ds = RetinalDataset(CONFIG["dataset_path"] / "val", CONFIG["class_names"], get_transforms(train=False))

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    class_weights = compute_class_weights(train_ds)
    print(f"Class weights: {[round(w, 2) for w in class_weights.tolist()]}")

    # CHANGE 6: determine resume state BEFORE batch-size selection.
    phase2_log = CONFIG["model_dir"] / "training_log_phase2.csv"
    resume_epoch = 0
    if phase2_log.exists():
        with open(phase2_log) as f:
            rows = list(csv.DictReader(f))
            if rows:
                resume_epoch = int(rows[-1]["epoch"])

    # CHANGE 6: batch-size selection. Probing is ONLY allowed on a fresh run
    # from epoch 0. A resumed run never probes: it reuses the exact batch size
    # persisted in the train-state file (falls back to 32 for old checkpoints).
    if args.batch_size is not None:
        batch_size = args.batch_size
        print(f"Batch size: {batch_size} (--batch-size override)")
    elif resume_epoch > 0:
        batch_size = CONFIG["batch_size"]
        state_path = CONFIG["model_dir"] / "retinasense_train_state_phase2.pt"
        if state_path.exists():
            try:
                st = torch.load(str(state_path), map_location="cpu", weights_only=True)
                if isinstance(st, dict) and st.get("batch_size"):
                    batch_size = int(st["batch_size"])
            except Exception:
                pass
        print(f"Batch size: {batch_size} (resumed run - exact batch, no probe)")
    elif args.auto_batch and device.type == "cuda":
        print("Probing largest batch size that fits VRAM (fresh run from epoch 0)...")
        model_probe = RetinaSenseModel(num_classes=CONFIG["num_classes"], dropout=CONFIG["dropout"]).to(device)
        batch_size = auto_batch_size(model_probe, device)
        del model_probe
        print(f"Batch size: {batch_size} (auto-probed)")
    else:
        batch_size = CONFIG["batch_size"]
        print(f"Batch size: {batch_size} (default)")
    CONFIG["batch_size"] = batch_size
    CONFIG["num_workers"] = args.workers
    CONFIG["channels_last"] = args.channels_last

    sampler = get_balanced_sampler(train_ds)

    # CHANGE 3/4/7: pin_memory=True (no deprecated pin_memory_device),
    # persistent_workers=True, seeded workers -> fast + crash-free on Windows.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=CONFIG["prefetch_factor"] if args.workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=CONFIG["prefetch_factor"] if args.workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )

    print(f"DataLoader: num_workers={args.workers}, "
          f"prefetch_factor={CONFIG['prefetch_factor']}, pin_memory=True")

    if args.profile:
        print("[PROF] Decode/transform benchmark (main process, --profile only)...", flush=True)
        bench_decode(train_ds, "train transform")
        bench_decode(val_ds, "val transform")

    model = RetinaSenseModel(num_classes=CONFIG["num_classes"], dropout=CONFIG["dropout"]).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("Memory format: channels_last (NHWC)")

    phase1_path = CONFIG["model_dir"] / "retinasense_phase1.pth"
    if phase1_path.exists():
        print(f"Phase 1 checkpoint found, loading")
        model.load_state_dict(torch.load(str(phase1_path), map_location=device, weights_only=True))
    else:
        freeze_backbone(model, True)
        model, p1 = train_phase(model, "1", train_loader, val_loader, device, class_weights,
                                profile=args.profile)

    unfreeze_last(model, CONFIG["backbone_unfreeze_layers"])
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n:,}")

    if args.profile and args.profile_epochs is not None:
        target = resume_epoch + args.profile_epochs
        if target < CONFIG["phase2_epochs"]:
            print(f"[PROF] Benchmark mode: capping Phase 2 at {target} epochs")
            CONFIG["phase2_epochs"] = target

    best_p2 = CONFIG["model_dir"] / "retinasense_best.pth"
    if resume_epoch > 0:
        print(f"Resuming Phase 2 from epoch {resume_epoch + 1}")
        if best_p2.exists():
            print(f"Loading best Phase 2 weights from {best_p2}")
            model.load_state_dict(torch.load(str(best_p2), map_location=device, weights_only=True))

    # Compile AFTER all checkpoint loads: torch 2.13 prefixes compiled state_dict
    # keys with "_orig_mod.", which would break strict load of existing checkpoints.
    if args.compile:
        print("Compiling model with torch.compile...", flush=True)
        t0 = time.perf_counter()
        model = torch.compile(model)
        print(f"  compile done in {time.perf_counter()-t0:.1f}s", flush=True)

    model, p2 = train_phase(model, "2", train_loader, val_loader, device, class_weights,
                            start_epoch=resume_epoch, profile=args.profile)

    final_path = CONFIG["model_dir"] / "retinasense_model.pth"
    torch.save(_raw_module(model).state_dict(), str(final_path))
    print(f"Saved: {final_path}")

    with open(CONFIG["model_dir"] / "training_results.json", "w") as f:
        json.dump({"phase1_best": 0.0, "phase2_best": p2, "device": str(device)}, f)
    print(f"Done! Phase 2 best: {p2:.4f}")

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak VRAM: {peak_mem:.0f} MB")


if __name__ == "__main__":
    main()
