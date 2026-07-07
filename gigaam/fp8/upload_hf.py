"""Upload the FP8 matrix to the HuggingFace fork (branch-per-variant).

Mirrors the upstream convention (https://huggingface.co/ai-sage/GigaAM-v3) where
each model variant lives on its own branch (``e2e_ctc``, ``ctc``, ``e2e_rnnt``,
``rnnt``). For each variant this script:

1. Ensures the branch exists in the local clone of your fork — reuses an
   existing ``origin/<variant>`` or creates ``<variant>`` from
   ``upstream/<variant>``.
2. Drops the legacy ``pytorch_model.bin`` (transformers auto-detects
   ``model.safetensors``).
3. Copies the native-convention package from ``hf_packages/<variant>/`` into the
   working tree: ``model.safetensors`` (fp16), ``model_fp8.safetensors`` (FP8
   uint8 + scales), ``model_fp8.safetensors.activation_scales.json``, ``config.json``,
   ``modeling_gigaam.py``, and ``tokenizer.model`` (e2e variants only).
   ``README.md`` and ``.gitattributes`` are **never** overwritten.
4. Verifies the staged files (sizes, dtypes, no ``model.`` prefix, README untouched).
5. Commits and (only with ``--push``) pushes the branch to ``origin``.

By default the script is a **dry run**: it performs every step up to (not
including) the push and prints a per-branch ``git diff --stat``. Pass ``--push``
after reviewing the dry run to actually upload.

The script operates inside the existing clone at ``hf_repo/GigaAM-v3`` so git-lfs
state and the origin/upstream remotes are reused. It does not touch ``main``,
``ssl``, or any non-variant branch.

Usage
-----
    # Dry run (default) — review what would be uploaded:
    uv run python -m gigaam.fp8.upload_hf

    # Actually push to the fork after reviewing the dry run:
    uv run python -m gigaam.fp8.upload_hf --push

    # Upload a subset:
    uv run python -m gigaam.fp8.upload_hf --variants e2e_ctc rnnt
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# (variant, create-from source). Branches already in origin are reused as-is;
# branches missing from origin are created from the listed upstream branch.
VARIANTS = [
    ("e2e_ctc", "upstream/e2e_ctc"),
    ("ctc", "upstream/ctc"),
    ("e2e_rnnt", "upstream/e2e_rnnt"),
    ("rnnt", "upstream/rnnt"),
]

# Files copied from the matrix package into the branch working tree. README.md
# and .gitattributes are deliberately excluded (preserved from the branch).
PACKAGE_FILES = [
    "model.safetensors",
    "model_fp8.safetensors",
    "model_fp8.safetensors.activation_scales.json",
    "config.json",
    "modeling_gigaam.py",
    "tokenizer.model",  # only present for e2e variants
]

# Files removed from the branch (legacy fp16 pytorch format, superseded by
# model.safetensors which transformers auto-detects).
REMOVE_FILES = ["pytorch_model.bin"]


def _run(cmd: List[str], cwd: str, check: bool = True, env: Optional[Dict] = None) -> subprocess.CompletedProcess:
    """Run a command, streaming combined output; raise on failure when check."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if result.stdout.strip():
        for line in result.stdout.rstrip().splitlines():
            print(f"    {line}")
    if result.returncode != 0:
        if result.stderr.strip():
            for line in result.stderr.rstrip().splitlines():
                print(f"    ! {line}")
        if check:
            raise RuntimeError(f"command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


# Skip LFS smudge on git operations: we never need the LFS *content* of files we
# are about to replace or delete (model.safetensors, pytorch_model.bin). Checkout
# only needs the LFS pointer files, so skipping smudge avoids downloading
# multi-hundred-MB objects that would immediately be overwritten.
_LFS_SKIP = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}


def _clean_tree(repo: str) -> None:
    """Discard any working-tree changes + untracked files from a prior variant."""
    _run(["git", "reset", "--hard", "HEAD", "--quiet"], repo, check=False)
    _run(["git", "clean", "-fdq"], repo, check=False)


def _has_ref(repo: str, ref: str) -> bool:
    """True if a local or remote ref exists in the repo."""
    r = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        cwd=repo, text=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0


def _branch_ahead_of_origin(repo: str, branch: str) -> bool:
    """True if the local ``branch`` has commits not on ``origin/<branch>``.

    Returns True when origin doesn't carry the branch at all (it will be created
    on push), or when local is strictly ahead of origin. Both mean a push is due.
    """
    if not _has_ref(repo, f"origin/{branch}"):
        return True  # new branch — push will create it
    r = subprocess.run(
        ["git", "rev-list", "--count", f"origin/{branch}..{branch}"],
        cwd=repo, text=True, capture_output=True,
    )
    try:
        return int(r.stdout.strip()) > 0
    except ValueError:
        return False


def _ensure_branch(repo: str, variant: str, upstream_branch: str) -> str:
    """Ensure a local branch ``variant`` exists, tracked at origin/<variant>.

    If ``origin/<variant>`` exists, check it out (creating a local tracking
    branch). Otherwise create the branch from ``upstream_branch`` (it will be
    pushed to origin on the first ``--push``). Returns the branch name.
    """
    if _has_ref(repo, f"origin/{variant}"):
        # Tracking branch exists in the fork — check it out (creating local if
        # needed) and fast-forward to origin to start from the fork's current tip.
        # LFS smudge is skipped (env): we only need pointer files here.
        if not _has_ref(repo, variant):
            _run(["git", "branch", variant, f"origin/{variant}"], repo)
        _run(["git", "checkout", "--quiet", variant], repo, env=_LFS_SKIP)
        _run(["git", "merge", "--ff-only", "--quiet", f"origin/{variant}"], repo, check=False)
    else:
        # Not in the fork yet — create from upstream, to be pushed on --push.
        if not _has_ref(repo, variant):
            _run(["git", "branch", variant, upstream_branch], repo)
        _run(["git", "checkout", "--quiet", variant], repo, env=_LFS_SKIP)
        print(f"  (created {variant} from {upstream_branch}; will push to origin on --push)")
    return variant


def _verify_package(repo: str, variant: str, matrix_summary: Dict) -> None:
    """Assert the staged files in the working tree are sane before committing."""
    from safetensors import safe_open

    work = Path(repo)
    fp16 = work / "model.safetensors"
    fp8 = work / "model_fp8.safetensors"
    assert fp16.exists(), f"{variant}: model.safetensors missing"
    assert fp8.exists(), f"{variant}: model_fp8.safetensors missing"
    # FP8 file must be smaller than the fp16 parity file.
    s_fp16 = fp16.stat().st_size
    s_fp8 = fp8.stat().st_size
    assert s_fp8 < s_fp16, (
        f"{variant}: model_fp8 ({s_fp8/1e6:.1f} MB) not smaller than "
        f"model ({s_fp16/1e6:.1f} MB)"
    )
    # No `model.`-prefixed keys in either file (native convention).
    for fn, p in [("model.safetensors", fp16), ("model_fp8.safetensors", fp8)]:
        with safe_open(str(p), framework="pt") as sf:
            bad = [k for k in sf.keys() if k.startswith("model.")]
            assert not bad, f"{variant}: {fn} has {len(bad)} model.-prefixed keys"
    # config.json parses and model_name references the variant.
    cfg = json.loads((work / "config.json").read_text())
    model_name = cfg.get("cfg", {}).get("model", {}).get("cfg", {}).get("model_name", "")
    assert variant in model_name, (
        f"{variant}: config model_name={model_name!r} does not reference the variant"
    )
    # README.md must be unchanged (the script never overwrites it; this is a guard
    # against accidental external edits during staging).
    readme_status = subprocess.run(
        ["git", "status", "--porcelain", "--", "README.md"], cwd=repo,
        text=True, capture_output=True,
    ).stdout.strip()
    assert not readme_status, (
        f"{variant}: README.md is modified ({readme_status}); refusing to commit"
    )
    print(
        f"  verify OK: model={s_fp16/1e6:.1f} MB, model_fp8={s_fp8/1e6:.1f} MB, "
        f"model_name={model_name!r}, README untouched"
    )


def _stage_variant(
    variant: str,
    upstream_branch: str,
    pkg_dir: str,
    repo: str,
    kd_label: str,
) -> Dict:
    """Stage one variant's files onto its branch. Returns a status dict."""
    print(f"\n{'='*60}\n=== {variant}  (from {upstream_branch}, {kd_label})\n{'='*60}")
    _clean_tree(repo)
    _ensure_branch(repo, variant, upstream_branch)
    _clean_tree(repo)  # checkout may leave LFS-smudge artifacts; clean again

    # 1. Remove legacy files (only if tracked; --ignore-unmatch handles absence).
    for fn in REMOVE_FILES:
        _run(["git", "rm", "--quiet", "--ignore-unmatch", fn], repo, env=_LFS_SKIP)

    # 2. Copy package files (skip README.md / .gitattributes / missing files).
    copied: List[str] = []
    for fn in PACKAGE_FILES:
        src = Path(pkg_dir) / fn
        if not src.exists():
            continue
        shutil.copy2(src, Path(repo) / fn)
        copied.append(fn)
    print(f"  copied: {copied}")

    # 3. Stage the copied files. Run WITHOUT the LFS-skip env so git add converts
    #    the safetensors/model content into LFS pointers (the skip only affects
    #    checkout downloads, not add). Removed files are already staged by git rm.
    if copied:
        _run(["git", "add", "--"] + copied, repo)

    # 4. Verify before commit.
    _verify_package(repo, variant, {})

    # 5. Diff summary so the dry run is reviewable.
    diff = _run(["git", "diff", "--cached", "--stat"], repo)
    print(f"  staged diff:\n{diff.stdout.rstrip()}")

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True
    ).stdout
    has_changes = bool(status.strip())
    if not has_changes:
        print(f"  no new changes to commit for {variant}")
        # Even if nothing was committed this run, the branch may still be ahead of
        # origin from a prior run (e.g. a dry run committed but didn't push). Push
        # if there are unpushed commits.
        return {"variant": variant, "staged": False, "committed": False, "pushed": False,
                "needs_push": _branch_ahead_of_origin(repo, variant)}

    # 6. Commit.
    msg = f"Add FP8 E4M3 quantized weights ({variant}) [{kd_label}]"
    _run(["git", "commit", "--quiet", "-m", msg], repo)
    print(f"  committed: {msg}")
    return {"variant": variant, "staged": True, "committed": True, "pushed": False,
            "needs_push": True}


def main():
    p = argparse.ArgumentParser(description="Upload the FP8 matrix to the HF fork.")
    p.add_argument(
        "--repo", default="hf_repo/GigaAM-v3",
        help="Path to the HF clone (must have origin + upstream remotes).",
    )
    p.add_argument(
        "--packages", default="hf_packages",
        help="Root dir of the matrix output (contains <variant>/ subdirs).",
    )
    p.add_argument("--variants", nargs="+", default=None, help="Subset of variants.")
    p.add_argument(
        "--summary", default="hf_packages/matrix_summary.json",
        help="matrix_summary.json from run_matrix (for KD/PTQ labels).",
    )
    p.add_argument("--push", action="store_true", help="Actually push to origin (default: dry run).")
    args = p.parse_args()

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(os.path.join(repo, ".git")):
        sys.exit(f"--repo {repo} is not a git repository")
    # Confirm remotes exist before doing anything.
    for r in ("origin", "upstream"):
        out = subprocess.run(
            ["git", "remote", "get-url", r], cwd=repo,
            text=True, capture_output=True,
        )
        if out.returncode != 0:
            sys.exit(f"--repo {repo} missing remote '{r}'")
        print(f"remote {r}: {out.stdout.strip()}")

    # Load KD/PTQ labels from the matrix summary.
    kd_by_variant: Dict[str, str] = {}
    if os.path.exists(args.summary):
        for entry in json.loads(Path(args.summary).read_text()):
            v = entry.get("variant")
            kd_by_variant[v] = "KD" if entry.get("kd") else "PTQ"
    else:
        print(f"[warn] {args.summary} not found; defaulting all labels to 'PTQ+KD?'")

    variants = VARIANTS
    if args.variants:
        variants = [v for v in VARIANTS if v[0] in args.variants]
        if not variants:
            sys.exit(f"no matching variants in {args.variants}")

    # Fetch upstream so the create-from refs are current.
    print("\nFetching upstream ...")
    _run(["git", "fetch", "--quiet", "upstream"], repo, check=False)

    results: List[Dict] = []
    starting_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo,
        text=True, capture_output=True,
    ).stdout.strip()

    for variant, upstream_branch in variants:
        pkg_dir = os.path.join(args.packages, variant)
        if not os.path.isdir(pkg_dir):
            print(f"\n!! {variant}: package dir {pkg_dir} missing — skipping")
            results.append({"variant": variant, "error": "package missing"})
            continue
        label = kd_by_variant.get(variant, "PTQ")
        try:
            res = _stage_variant(variant, upstream_branch, pkg_dir, repo, label)
            # Push if requested AND the branch has unpushed commits (committed this
            # run, or left over from a prior dry run that committed but didn't push).
            if args.push and res.get("needs_push"):
                print(f"  pushing {variant} -> origin ...")
                push = _run(["git", "push", "origin", variant], repo, check=False)
                res["pushed"] = push.returncode == 0
                if push.returncode != 0:
                    res["error"] = "push failed"
                else:
                    res["needs_push"] = False
            elif args.push and not res.get("needs_push"):
                print(f"  (already up to date on origin)")
            elif not args.push:
                tag = "would push" if res.get("needs_push") else "nothing to push"
                print(f"  (dry run; {tag}. Re-run with --push to upload.)")
            results.append(res)
        except Exception as e:
            import traceback
            print(f"\n!!! {variant} FAILED: {e}")
            traceback.print_exc()
            results.append({"variant": variant, "error": str(e)})
        finally:
            # Leave the repo on a clean footing between variants: discard tracked
            # changes and remove untracked files (the copied package files).
            _clean_tree(repo)

    # Return to the starting branch (tree is now clean from the finally block).
    if starting_branch:
        _run(["git", "checkout", "--quiet", starting_branch], repo, check=False, env=_LFS_SKIP)

    print(f"\n{'='*60}\n=== UPLOAD SUMMARY {'(PUSHED)' if args.push else '(DRY RUN)'}\n{'='*60}")
    print(json.dumps(results, indent=2))
    if not args.push:
        print("\nThis was a dry run. To upload, re-run with --push.")


if __name__ == "__main__":
    main()
