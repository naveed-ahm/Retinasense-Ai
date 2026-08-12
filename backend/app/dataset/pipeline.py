import json
import time
from pathlib import Path
from app.dataset.config import DatasetConfig
from app.dataset.validator import validate_images
from app.dataset.preprocessor import batch_preprocess
from app.dataset.splitter import split_dataset
from app.dataset.statistics import generate_statistics


def run_pipeline(
    config: DatasetConfig | None = None,
    output_report: str | None = None,
    execute_split: bool = False,
) -> dict:
    if config is None:
        config = DatasetConfig()
    start = time.time()
    report = {"pipeline": "RetinaSense Dataset Pipeline", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "config": str(config), "phases": {}}
    print("[Phase 1/4] Validating images...")
    validation = validate_images(config)
    report["phases"]["validation"] = {
        "total": validation["total"],
        "valid": validation["valid"],
        "corrupted": validation["corrupted"],
        "missing": validation["missing"],
        "empty": validation["empty"],
        "duplicates": validation["duplicates"],
    }
    if validation["valid"] == 0:
        report["phases"]["error"] = "No valid images to process"
        report["elapsed_seconds"] = time.time() - start
        if output_report:
            Path(output_report).write_text(json.dumps(report, indent=2))
        return report
    print(f"  Found {validation['valid']} valid images, {validation['corrupted']} corrupted, {validation['duplicates']} duplicates")
    print("[Phase 2/4] Preprocessing images...")
    preprocessed = 0
    errors = 0
    for disease_name, disease_dir in config.disease_dirs.items():
        if not disease_dir.exists():
            continue
        dest = config.root / "preprocessed" / disease_name
        results = batch_preprocess(disease_dir, dest, config)
        for r in results:
            if r["status"] == "ok":
                preprocessed += 1
            else:
                errors += 1
        print(f"  {disease_name}: {len(results)} images ({sum(1 for r in results if r['status']=='ok')} ok, {sum(1 for r in results if r['status']=='error')} errors)")
    report["phases"]["preprocessing"] = {"preprocessed": preprocessed, "errors": errors}
    print(f"[Phase 3/4] Splitting dataset (execute={execute_split})...")
    split = split_dataset(config, dry_run=not execute_split)
    report["phases"]["split"] = split
    print(f"  Train/Val/Test split computed across {len(split.get('splits', {}))} categories")
    print("[Phase 4/4] Generating dataset statistics...")
    stats = generate_statistics(config)
    report["phases"]["statistics"] = stats
    for cat, cat_stats in stats.get("categories", {}).items():
        if "pixel_mean" in cat_stats:
            print(f"  {cat}: {cat_stats['count']} images, "
                  f"size {cat_stats['width']['mean']:.0f}x{cat_stats['height']['mean']:.0f}, "
                  f"mean RGB=({cat_stats['pixel_mean']['r']:.1f},{cat_stats['pixel_mean']['g']:.1f},{cat_stats['pixel_mean']['b']:.1f})")
    report["elapsed_seconds"] = round(time.time() - start, 2)
    print(f"\nPipeline completed in {report['elapsed_seconds']}s")
    if output_report:
        Path(output_report).write_text(json.dumps(report, indent=2))
        print(f"Report saved to {output_report}")
    return report
