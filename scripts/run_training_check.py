"""Check a few resume-training steps on explicitly supplied local author data.

This verifies execution, not reproduction of paper metrics. Inputs are read only;
outputs are written to a separate candidate output directory.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def metadata(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--roi-cache", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "local_training_check")
    args = parser.parse_args()
    if min(args.steps, args.data_factor, args.patch_size) <= 0:
        parser.error("steps, data-factor and patch-size must be positive")
    checkpoint, data_dir, roi = [p.resolve() for p in (args.checkpoint, args.data_dir, args.roi_cache)]
    output = args.output_dir.resolve()
    if output == data_dir or data_dir in output.parents:
        parser.error("output-dir must be outside the input dataset")
    for path in (checkpoint, roi):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    image_dir = data_dir / ("images" if args.data_factor == 1 else f"images_{args.data_factor}")
    if not image_dir.is_dir() or not any(image_dir.rglob("*")):
        parser.error(f"pre-resized image directory is required; no resizing will be performed: {image_dir}")
    inputs_before = {str(path): metadata(path) for path in (checkpoint, roi)}
    image_dir_before = metadata(image_dir)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = int(state["step"])
    final_step = step + args.steps
    max_steps = final_step + 1
    del state
    gc.collect()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CKPT_PATH=str(checkpoint), DATA_DIR=str(data_dir),
               ROI_CACHE_PATH=str(roi), RESULT_DIR=str(output),
               MAX_STEPS=str(max_steps), SAVE_STEPS=str(max_steps), EVAL_STEPS="",
               DATA_FACTOR=str(args.data_factor), PATCH_SIZE=str(args.patch_size),
               LOAD_OPTIMIZER_STATE="1", LOAD_STRATEGY_STATE="1", ROI_FIXED_FINETUNE="1",
               NORMALIZE_WORLD_SPACE="1", USE_BILATERAL_GRID="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, str(ROOT / "examples" / "train_irregular_region_filter.py")]
    with (output / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    inputs_after = {str(path): metadata(path) for path in (checkpoint, roi)}
    saved = output / "ckpts" / f"ckpt_{final_step}_rank0.pt"
    summary = {"paper_metrics_reproduced": False, "steps_requested": args.steps,
               "input_step": step, "expected_final_step": final_step,
               "input_metadata_before": inputs_before, "input_metadata_after": inputs_after,
               "input_metadata_unchanged": inputs_before == inputs_after,
               "pre_resized_image_directory": str(image_dir),
               "image_directory_metadata_unchanged": image_dir_before == metadata(image_dir),
               "exit_code": result.returncode, "output_checkpoint": str(saved)}
    if result.returncode == 0 and saved.is_file():
        state = torch.load(saved, map_location="cpu", weights_only=False)
        summary["observed_final_step"] = int(state["step"])
        summary["all_splats_finite"] = all(bool(torch.isfinite(value).all())
                                           for value in state["splats"].values())
        del state
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if (result.returncode or not summary.get("all_splats_finite") or
            summary.get("observed_final_step") != final_step or not summary["input_metadata_unchanged"]):
        raise RuntimeError(f"Training execution check failed; see {output / 'run.log'}")


if __name__ == "__main__":
    main()
