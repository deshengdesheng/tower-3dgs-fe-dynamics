"""Run the terrain CLI on synthetic Gaussian planes without project data."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def checkpoint(points):
    means = torch.tensor(points, dtype=torch.float32)
    n = len(means)
    quats = torch.zeros((n, 4))
    quats[:, 0] = 1
    return {"splats": {"means": means, "quats": quats,
                       "scales": torch.full((n, 3), -3.),
                       "opacities": torch.full((n,), 2.),
                       "sh0": torch.full((n, 1, 3), .3),
                       "shN": torch.zeros((n, 15, 3))}}


def main():
    out = ROOT / "outputs" / "terrain_smoke"
    out.mkdir(parents=True, exist_ok=True)
    # Multiple samples per grid support the source-feature and height statistics.
    points = [[x + dx, y + dy, 0.] for x in np.arange(0, 2.1, .5)
              for y in np.arange(0, 2.1, .5) for dx, dy in [(0., 0.), (.04, .02), (.02, .04)]]
    reference = [[x, y, 0.] for x in np.arange(-2, 4.1, .25)
                 for y in np.arange(-2, 4.1, .25)]
    input_file, reference_file = out / "input.pt", out / "reference.pt"
    output_file = out / "expanded.pt"
    torch.save(checkpoint(points), input_file)
    torch.save(checkpoint(reference), reference_file)
    command = [sys.executable, str(ROOT / "extended_more_layers_param.py"),
               "--input", str(input_file), "--kriging", str(reference_file),
               "--output", str(output_file), "--grid-size", ".5",
               "--expand-layers", "1", "--observer-center", "1,1,0",
               "--max-multiplier", "2", "--no-multiprocessing",
               "--top-80-percent", "1", "--bottom-82-percent", "1",
               "--min-distance", "1", "--max-candidates", "4",
               "--max-candidates-detail", "4", "--search-radius", "2",
               "--max-neighbors", "4", "--spatial-threshold", "0",
               "--similarity-threshold", "0"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    (out / "run.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Terrain CLI failed; see {out / 'run.log'}")
    result = torch.load(output_file, map_location="cpu", weights_only=False)["splats"]
    count = len(result["means"])
    finite = all(bool(torch.isfinite(value).all()) for value in result.values())
    summary = {"synthetic_data_only": True, "device": "cpu", "input_count": len(points),
               "output_count": count, "added_count": count - len(points),
               "all_tensors_finite": finite, "command": command}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if count <= len(points) or not finite:
        raise AssertionError(f"Terrain smoke did not expand correctly: {summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
