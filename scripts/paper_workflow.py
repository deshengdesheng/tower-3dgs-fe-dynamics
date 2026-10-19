#!/usr/bin/env python3
"""Inspect and run the paper-aligned four-step workflow manifest.

The manifest is deliberately conservative: Steps I--III are documented as
requiring external/private scientific inputs, and Step IV is only marked
``partial_verified`` because the saved tower run verifies the registered
displacement handoff and translation-only dynamic rendering.  This entry
point never upgrades those statuses to a full end-to-end claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "workflow" / "paper_workflow.json"
ALLOWED_STATUSES = {"not_reproduced_in_candidate", "partial_verified", "verified"}


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("workflow manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported workflow manifest schema_version")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("workflow manifest must contain a non-empty steps list")
    seen: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("each workflow step must be an object")
        step_id = str(step.get("id", ""))
        if not step_id or step_id in seen:
            raise ValueError(f"invalid or duplicate step id: {step_id!r}")
        seen.add(step_id)
        status = step.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"step {step_id} has unsupported status: {status!r}")
        for key in ("components", "external_inputs", "commands", "expected_outputs", "evidence"):
            if not isinstance(step.get(key, []), list):
                raise ValueError(f"step {step_id}.{key} must be a list")
    policy = payload.get("status_policy")
    if not isinstance(policy, dict) or policy.get("end_to_end_reproduction") != "not_reproduced_in_candidate":
        raise ValueError("manifest must explicitly declare that end-to-end reproduction is not complete")
    return payload


def _matches(repo_root: Path, pattern: str) -> bool:
    candidate = repo_root / pattern
    if any(char in pattern for char in "*?["):
        return any(path.is_file() for path in repo_root.glob(pattern))
    return candidate.is_file()


def _component_refs(step: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Yield (path, availability, scope) for top-level and nested components."""
    refs: list[tuple[str, str, str]] = []
    for component in step.get("components", []):
        if isinstance(component, str):
            refs.append((component, "bundled", step["id"]))
        else:
            refs.append((str(component.get("path", "")), str(component.get("availability", "bundled")), step["id"]))
    for substep in step.get("substeps", []):
        if not isinstance(substep, dict):
            continue
        scope = f"{step['id']}{substep.get('id', '')}"
        for component in substep.get("components", []):
            if isinstance(component, str):
                refs.append((component, "bundled", scope))
            else:
                refs.append((str(component.get("path", "")), str(component.get("availability", "bundled")), scope))
    return refs


def _display_path(path: Path) -> str:
    """Render a manifest path without failing for a custom manifest location."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def check_manifest(manifest_path: Path) -> int:
    payload = _load_manifest(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    for step in payload["steps"]:
        step_id = step["id"]
        for relative, availability, component_scope in _component_refs(step):
            if availability == "bundled" and (not relative or not _matches(REPO_ROOT, relative)):
                errors.append(f"Step {component_scope}: missing bundled component {relative!r}")
        for evidence in step.get("evidence", []):
            if not _matches(REPO_ROOT, evidence):
                errors.append(f"Step {step_id}: missing evidence {evidence!r}")
        if step["status"] == "not_reproduced_in_candidate":
            warnings.append(f"Step {step_id}: external inputs required; not reproduced by this candidate")
        elif step["status"] == "partial_verified":
            warnings.append(f"Step {step_id}: partial verification only; see verified_scope/limitations")
    if errors:
        print("WORKFLOW CHECK: FAIL")
        for message in errors:
            print(f"  ERROR: {message}")
        return 1
    print("WORKFLOW CHECK: PASS")
    print(f"  manifest: {_display_path(manifest_path)}")
    print("  end-to-end reproduction: NOT COMPLETE (explicit manifest status)")
    for message in warnings:
        print(f"  NOTE: {message}")
    return 0


def show_manifest(payload: dict[str, Any]) -> None:
    print(payload["title"])
    print(f"End-to-end status: {payload['status_policy']['end_to_end_reproduction']}")
    for step in payload["steps"]:
        print(f"Step {step['id']}: {step['title']} [{step['status']}]")
        print(f"  {step['paper_scope']}")
        if step.get("verified_scope"):
            print(f"  Verified scope: {step['verified_scope']}")
        if step.get("limitations"):
            print(f"  Limitation: {step['limitations']}")


def print_plan(payload: dict[str, Any]) -> None:
    print("Execution plan (commands are templates; external inputs are not bundled):")
    for step in payload["steps"]:
        print(f"\nStep {step['id']} [{step['status']}]")
        for command in step.get("commands", []):
            print(f"  {command}")
        if not step.get("commands"):
            print("  No bundled command: requires the author-side FE/wind workflow.")


def run_step_iv(args: argparse.Namespace) -> int:
    if not args.runtime_dir:
        raise SystemExit("--run-step IV requires --runtime-dir")
    runtime_dir = args.runtime_dir.expanduser().resolve()
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_dynamic_gs.py"),
        "--runtime-dir",
        str(runtime_dir),
        "--frames",
        str(args.frames),
        "--resolution",
        args.resolution,
        "--device",
        args.device,
    ]
    if args.output_dir:
        command.extend(("--output-dir", str(args.output_dir.expanduser().resolve())))
    print("Step IV mode: partial verified handoff (translation-only dynamic renderer)")
    print("[run]", " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true", help="validate manifest components and saved evidence")
    parser.add_argument("--show", action="store_true", help="show the four paper stages and explicit limitations")
    parser.add_argument("--plan", action="store_true", help="print execution templates without running them")
    parser.add_argument("--run-step", choices=("IV", "iv"), help="run only the verified Step-IV handoff")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--resolution", default="768x1152")
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    payload = _load_manifest(manifest_path)
    selected = args.check or args.show or args.plan or args.run_step
    if not selected:
        args.show = True
    if args.check:
        result = check_manifest(manifest_path)
        if result:
            return result
    if args.show:
        show_manifest(payload)
    if args.plan:
        print_plan(payload)
    if args.run_step:
        return run_step_iv(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
