"""
RetinaSense Dataset Management CLI.

Usage:
    python dataset_cli.py validate       # Validate all images in dataset dirs
    python dataset_cli.py preprocess     # Preprocess (resize, CLAHE, normalize)
    python dataset_cli.py split          # Show train/val/test split (dry run)
    python dataset_cli.py split --exec   # Actually execute the split (copies files)
    python dataset_cli.py stats          # Generate dataset statistics
    python dataset_cli.py pipeline       # Run full pipeline (validate -> preprocess -> split -> stats)
    python dataset_cli.py pipeline --exec --report report.json
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.dataset.config import DatasetConfig
from app.dataset.validator import validate_images
from app.dataset.preprocessor import batch_preprocess
from app.dataset.splitter import split_dataset
from app.dataset.statistics import generate_statistics
from app.dataset.pipeline import run_pipeline


def cmd_validate():
    config = DatasetConfig()
    result = validate_images(config)
    print(json.dumps({k: v for k, v in result.items() if k != "images" and k != "hash_map"}, indent=2))
    if result["corrupted"] or result["duplicates"]:
        for img in result.get("images", []):
            if img.get("error"):
                print(f"  ISSUE: {img['path']} -> {img['error']}")


def cmd_preprocess():
    config = DatasetConfig()
    total = 0
    for disease_name, disease_dir in config.disease_dirs.items():
        if not disease_dir.exists():
            print(f"Skipping {disease_name} (not found)")
            continue
        dest = config.root / "preprocessed" / disease_name
        results = batch_preprocess(disease_dir, dest, config)
        ok = sum(1 for r in results if r["status"] == "ok")
        err = sum(1 for r in results if r["status"] == "error")
        total += ok
        print(f"{disease_name}: {ok} preprocessed, {err} errors")
    print(f"\nTotal preprocessed: {total}")


def cmd_split():
    config = DatasetConfig()
    dry_run = "--exec" not in sys.argv
    result = split_dataset(config, dry_run=dry_run)
    print(json.dumps(result, indent=2))


def cmd_stats():
    config = DatasetConfig()
    result = generate_statistics(config)
    print(json.dumps(result, indent=2, default=str))


def cmd_pipeline():
    config = DatasetConfig()
    execute = "--exec" in sys.argv
    report_path = None
    if "--report" in sys.argv:
        idx = sys.argv.index("--report")
        if idx + 1 < len(sys.argv):
            report_path = sys.argv[idx + 1]
    result = run_pipeline(config, output_report=report_path, execute_split=execute)
    print(f"\nStatus: {len(result['phases'])} phases completed in {result.get('elapsed_seconds', '?')}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    command = sys.argv[1]
    cmds = {
        "validate": cmd_validate,
        "preprocess": cmd_preprocess,
        "split": cmd_split,
        "stats": cmd_stats,
        "pipeline": cmd_pipeline,
    }
    if command in cmds:
        cmds[command]()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
