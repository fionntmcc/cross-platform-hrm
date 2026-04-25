# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Publish trained model artifacts to a GitHub Release.

Creates a git tag, pushes it, waits for the Actions workflow to create the
draft release, uploads all model files, then publishes the release.

No external dependencies — uses only Python stdlib.

Usage:
    python scripts/publish_models.py --tag v1.0.0 --token ghp_xxxxx

    # Or set env var to avoid typing token on screen:
    $env:GITHUB_TOKEN = "ghp_xxxxx"
    python scripts/publish_models.py --tag v1.0.0

    # Dry run (tag + push only, skip upload/publish):
    python scripts/publish_models.py --tag v1.0.0 --dry-run

Get a token at: https://github.com/settings/tokens
Required scope: repo  (for private repos) or public_repo (for public repos)
"""

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "fionntmcc/cross-platform-hrm"
API_BASE = f"https://api.github.com/repos/{REPO}"
MODEL_DIR = Path(__file__).parent.parent / "model"

# GitHub API helpers


def _headers(token: str) -> dict:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get(url: str, token: str) -> dict | list:
    req = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _post(url: str, token: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={**_headers(token), "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _patch(url: str, token: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={**_headers(token), "Content-Type": "application/json"},
        method="PATCH",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get_release_by_tag(tag: str, token: str) -> dict | None:
    """Return release for a tag, or None if it does not exist."""
    try:
        return _get(f"{API_BASE}/releases/tags/{tag}", token)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _create_draft_release(tag: str, token: str, notes: str) -> dict:
    """Create a draft release for an existing tag."""
    body = {
        "tag_name": tag,
        "name": f"Model Release {tag}",
        "body": notes,
        "draft": True,
    }
    return _post(f"{API_BASE}/releases", token, body)


def _upload(upload_url: str, token: str, path: Path) -> dict:
    """Upload a single file to the GitHub release upload endpoint."""
    # upload_url looks like: https://uploads.github.com/repos/.../releases/123/assets{?name,label}
    url = upload_url.split("{")[0] + f"?name={path.name}"
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()
    req = urllib.request.Request(
        url,
        data=data,
        headers={**_headers(token), "Content-Type": mime, "Content-Length": str(len(data))},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# Git helpers


def run_git(*args: str, cwd: Path) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        print(f"git {' '.join(args)} failed:\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()


# Release logic


def wait_for_draft_release(tag: str, token: str, timeout: int = 90) -> dict:
    """Poll until the Actions workflow creates the draft release for this tag."""
    url = f"{API_BASE}/releases/tags/{tag}"
    deadline = time.time() + timeout
    print(f"Waiting for GitHub Actions to create the draft release for {tag} ", end="", flush=True)
    while time.time() < deadline:
        try:
            r = _get(url, token)
            print(" found!")
            return r
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(".", end="", flush=True)
                time.sleep(5)
            else:
                raise
    print()
    print(
        f"Timed out after {timeout}s waiting for release.\n"
        "Check https://github.com/fionntmcc/cross-platform-hrm/actions\n"
        "Then re-run with --skip-tag to just upload assets to the existing release.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag, upload and publish a GitHub model release.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tag", required=True, metavar="TAG", help="Release tag, e.g. v1.0.0")
    parser.add_argument(
        "--message",
        metavar="MSG",
        default=None,
        help="Annotated tag message / release description (default: auto-generated).",
    )
    parser.add_argument(
        "--token",
        metavar="TOKEN",
        default=None,
        help="GitHub PAT with repo/public_repo scope. Defaults to GITHUB_TOKEN env var.",
    )
    parser.add_argument(
        "--puzzle",
        choices=["sudoku_4x4", "sudoku_9x9"],
        default=None,
        help="Upload only models for this puzzle type.",
    )
    parser.add_argument(
        "--skip-tag",
        action="store_true",
        help="Skip creating/pushing the git tag (use if already pushed).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create and push the tag only — skip upload and publish.",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token and not args.dry_run:
        print(
            "Error: GitHub token required.\n"
            "Set GITHUB_TOKEN env var or pass --token.\n"
            "Get one at: https://github.com/settings/tokens (scope: public_repo)",
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path(__file__).parent.parent

    # 1. Collect all files currently present in model/
    model_files = sorted(f for f in MODEL_DIR.iterdir() if f.is_file())
    if args.puzzle:
        model_files = [f for f in model_files if args.puzzle in f.name]

    if not model_files:
        print(f"No model files found in {MODEL_DIR}", file=sys.stderr)
        sys.exit(1)

    total_mb = sum(f.stat().st_size for f in model_files) / 1_048_576
    print(f"Files to publish ({len(model_files)}, {total_mb:.1f} MB total):")
    for f in model_files:
        print(f"  {f.name} ({f.stat().st_size / 1_048_576:.1f} MB)")
    print()

    release_notes = args.message or (
        f"Model release {args.tag}\n\n"
        "Contains trained SimplifiedHRM (L-Module Only) checkpoints:\n"
        + "\n".join(f"  - {f.name}" for f in model_files)
    )

    # 2. Create + push annotated git tag
    if not args.skip_tag:
        # Check tag doesn't already exist
        existing = run_git("tag", "-l", args.tag, cwd=root)
        if existing:
            print(f"Tag {args.tag} already exists locally — skipping tag creation.")
        else:
            print(f"Creating annotated tag {args.tag} ...")
            run_git("tag", "-a", args.tag, "-m", release_notes, cwd=root)

        print(f"Pushing tag {args.tag} to origin ...")
        run_git("push", "origin", args.tag, cwd=root)
        print("  Pushed. GitHub Actions will create a draft release shortly.")
    else:
        print("Skipping tag creation (--skip-tag).")

    if args.dry_run:
        print("\nDry run complete — no upload performed.")
        if args.skip_tag:
            print(f"  python {__file__} --tag {args.tag} --skip-tag")
        else:
            print(f"  python {__file__} --tag {args.tag} --skip-tag")
        return

    # 3. Resolve the release shell that assets will be attached to.
    # If --skip-tag is used, do not wait for Actions; use an existing release
    # immediately or create the draft release directly.
    if args.skip_tag:
        release = _get_release_by_tag(args.tag, token)
        if release is None:
            print(f"No release found for {args.tag}. Creating draft release directly ...")
            try:
                release = _create_draft_release(args.tag, token, release_notes)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                print(f"Failed to create draft release ({e.code}): {body}", file=sys.stderr)
                sys.exit(1)
    else:
        release = wait_for_draft_release(args.tag, token)

    release_id = release["id"]
    upload_url = release["upload_url"]
    print(f"Release URL: {release['html_url']}")
    print()

    # 4. Upload model files
    print("Uploading assets ...")
    for f in model_files:
        size_mb = f.stat().st_size / 1_048_576
        print(f"  {f.name} ({size_mb:.1f} MB) ... ", end="", flush=True)
        try:
            _upload(upload_url, token, f)
            print("done")
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            print(f"FAILED ({e.code}): {body}", file=sys.stderr)
            sys.exit(1)

    # 5. Publish (undraft) the release
    print("\nPublishing release ...")
    updated = _patch(f"{API_BASE}/releases/{release_id}", token, {"draft": False})
    print(f"Published: {updated['html_url']}")
    print("\nDone! Download with:")
    print(f"  python scripts/download_model.py --tag {args.tag}")


if __name__ == "__main__":
    main()
