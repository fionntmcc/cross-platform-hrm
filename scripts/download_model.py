"""
Download trained model artifacts from a GitHub Release.

Fetches .pt and .json files from a tagged release of fionntmcc/cross-platform-hrm
and saves them into the model/ directory.

Usage:
    # Download from the latest release
    python scripts/download_model.py

    # Download from a specific release tag
    python scripts/download_model.py --tag v1.0.0

    # Download a specific puzzle's model only
    python scripts/download_model.py --puzzle sudoku_4x4

    # List available releases
    python scripts/download_model.py --list

    # Use a GitHub token (avoids rate-limiting on private repos or heavy use)
    python scripts/download_model.py --token ghp_xxxxx
    # or set the GITHUB_TOKEN environment variable

Environment:
    GITHUB_TOKEN  Optional personal access token to raise API rate limits.
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

REPO = "fionntmcc/cross-platform-hrm"
API_BASE = f"https://api.github.com/repos/{REPO}"
MODEL_DIR = Path(__file__).parent.parent / "model"

# File patterns to download from each release
ASSET_EXTENSIONS = {".pt", ".pth", ".json"}


def _headers(token: str | None) -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    tok = token or os.environ.get("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _get(url: str, token: str | None) -> dict | list:
    req = urllib.request.Request(url, headers=_headers(token))
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"GitHub API error {e.code}: {body}", file=sys.stderr)
        sys.exit(1)


def get_release(tag: str | None, token: str | None) -> dict:
    if tag is None or tag == "latest":
        url = f"{API_BASE}/releases/latest"
    else:
        url = f"{API_BASE}/releases/tags/{tag}"
    return _get(url, token)


def list_releases(token: str | None) -> None:
    releases = _get(f"{API_BASE}/releases", token)
    if not releases:
        print("No releases found.")
        return
    print(f"{'TAG':<20} {'TITLE':<40} {'ASSETS'}")
    print("-" * 72)
    for r in releases:
        n_assets = len(r.get("assets", []))
        draft = " [draft]" if r.get("draft") else ""
        print(f"{r['tag_name']:<20} {r['name']:<40} {n_assets}{draft}")


def download_asset(asset: dict, dest_dir: Path, token: str | None) -> None:
    url = asset["browser_download_url"]
    name = asset["name"]
    dest = dest_dir / name
    size_mb = asset["size"] / 1_048_576

    print(f"  Downloading {name} ({size_mb:.1f} MB) ...", end=" ", flush=True)

    req = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 20):  # 1 MiB chunks
            f.write(chunk)

    print("done")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download trained HRM model artifacts from a GitHub Release.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tag",
        metavar="TAG",
        default=None,
        help="Release tag to download from (default: latest).",
    )
    parser.add_argument(
        "--puzzle",
        choices=["sudoku_4x4", "sudoku_9x9"],
        default=None,
        help="Only download models for this puzzle type.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=str(MODEL_DIR),
        help=f"Directory to save models into (default: {MODEL_DIR}).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available releases and exit.",
    )
    parser.add_argument(
        "--token",
        metavar="TOKEN",
        default=None,
        help="GitHub personal access token (or set GITHUB_TOKEN env var).",
    )
    args = parser.parse_args()

    if args.list:
        list_releases(args.token)
        return

    # Fetch release metadata
    release = get_release(args.tag, args.token)
    tag = release["tag_name"]
    title = release["name"]
    assets = release.get("assets", [])
    is_draft = release.get("draft", False)

    print(f"Release : {tag} — {title}" + (" [DRAFT]" if is_draft else ""))
    print(f"Assets  : {len(assets)} file(s)")

    if is_draft:
        print("Warning: this is a draft release; model files may not be attached yet.")

    # Filter to relevant file types
    model_assets = [a for a in assets if Path(a["name"]).suffix in ASSET_EXTENSIONS]

    # Optionally filter by puzzle type
    if args.puzzle:
        model_assets = [a for a in model_assets if args.puzzle in a["name"]]

    if not model_assets:
        print("No matching model assets found in this release.", file=sys.stderr)
        print("You can attach files with: gh release upload <tag> model/*.pt model/*.json")
        sys.exit(1)

    # Download
    dest_dir = Path(args.output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {dest_dir}\n")
    for asset in model_assets:
        download_asset(asset, dest_dir, args.token)

    print(f"\nDone. {len(model_assets)} file(s) downloaded to {dest_dir}/")


if __name__ == "__main__":
    main()
