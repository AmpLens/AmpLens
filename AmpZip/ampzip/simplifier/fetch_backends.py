"""Download the optional SFT and CDS code and model checkpoints.

Repositories are checked out at pinned commits. The patched SFT files, CDS
checkpoints and CDS datasets are checked against stored SHA-256 digests.

Layout after `ampzip fetch-backends --cds`:

    <dest>/                       davidshih17/RL_amplitudes_claude @ RL_COMMIT (+ the patch)
    <dest>/models/sft_*pt_500k/   the SFT checkpoints, tracked upstream
    <dest>/spinorhelicity/        aureliendersy/spinorhelicity @ SH_COMMIT
    <dest>/spinorhelicity/*.pth   the three CDS checkpoints (with --cds)
    <dest>/spinorhelicity/data/n{4,5,6}_i1_3/data.prefix.counts.test_1000   (with --cds-data)

The destination defaults to AMPZIP_BACKENDS or the user data directory from _paths.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import _paths

RL_URL = "https://github.com/davidshih17/RL_amplitudes_claude"
RL_COMMIT = "9701092"
SH_URL = "https://github.com/aureliendersy/spinorhelicity"
SH_COMMIT = "6057a838d2bea6b5fd6324d2182ae7fbe676fa5e"

PATCH = Path(__file__).with_name("patches") / "sft_eval_extensions.patch"
MARKER = ".ampzip-backends.json"

# The three files the patch touches, at RL_COMMIT (before) and after the patch.
SFT_EVAL_SHA256 = {
    "src/eval_sft_4pt.py": ("aa212944a9323e4c23716e9109097a7bcebc950b5bf37a10ab7d73fc8eca2a27",
                            "fbde9ed8aa676b02a49e648e1821ad6085f80a6332478ba87b203d1307629750"),
    "src/eval_sft_5pt.py": ("b25ff342c794d8f1999d28445a25a0317450382ace7a2d59b7195dc52265a6a8",
                            "ec694f00b460980f3251b0b0f72159ef0c983e49a3376157b7d195a1532c4e74"),
    "src/eval_sft_6pt.py": ("931836f1cea22c64341a9f192b7f0d2e02a0c6d17416a601126f909a5ad93857",
                            "ac86286183e3cc9e150ab3fc30c03aeea3f103361b8e79a38e1bfc77342c4452"),
}

# CDS checkpoints: file name -> (Google Drive id from the spinorhelicity README, sha256, bytes).
CDS_WEIGHTS = {
    "n4_i1_3_l1000_10Mex.pth": ("1lq46Hc_eF8khsoC4k-lsMjZpaY0Pco29",
                                "befd35656f09b9e729ed1d11c04d6e7974ba2fa712e0213ea781996ffbca8710", 296900245),
    "n5_simplifier.pth": ("1hSyRKcsMjxgVZ3DB1EkwD0OFHxoId4FZ",
                          "2d63bccadd6168d85b567068b01fb5ae57c78d45fefe085a8d988842f6eda13b", 297014933),
    "n6_simplifier.pth": ("1JFhH-6UPXvLdFb3Gd-DFhCS7hGmgXQJw",
                          "f551a15a6533d47f0a70dd59122fe0b3ec2a3adf6f8638315d51ac68b02c7393", 297158485),
}


# CDS dataset archives: n -> (Google Drive id from the spinorhelicity README, zip sha256, zip
# bytes, member holding the test split, test-split sha256). Only the test split is kept.
CDS_DATA = {
    4: ("18idYRLv1Kzt4dcGS7s5YYONAbMsQTxD9",
        "347868658d6cfeda0db98aec48ad10080275309cf2b06e33bdcb35c8128d9584", 438247637,
        "n4_i1_3/data.prefix.counts.test_1000",
        "df4698a044438430a48b856e0b98a6cc5348f9de27326f0535810c42cd3eeb1b"),
    5: ("1eLbgC9F5j0zPcN3xV8-e8fH-BK1t3utF",
        "4263605ae98c7ed7ab019940f54e4ad96a699291d3dbca85bf6fe67c18e116c5", 919052880,
        "n5_i1_3/data.prefix.counts.test_1000",
        "9e0505d6dd4e60ae1cb4def8cbe7612c0f84ccdc53336ed96adc33074fbfff61"),
    6: ("1nrGG92Gl7xlW6l754SnZzuFL_ZfswtOC",
        "a363c8be221fdbc1517ff65566eea6e14e0f738718d042fc6eda02a5771a0471", 1329261749,
        "data.prefix.counts.test_1000",          # this archive has no top-level directory
        "3410ee6c077c62de53ece7403f2127ba968e466291238770c686bb4ddb06b36e"),
}


class FetchError(RuntimeError):
    pass


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, cwd: Optional[Path] = None) -> str:
    if shutil.which("git") is None:
        raise FetchError("Git is required to download the backend repositories. Add it to PATH.")
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise FetchError(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def _head(repo: Path) -> str:
    return _git("rev-parse", "HEAD", cwd=repo)


def _log(msg: str) -> None:
    print(f"[fetch-backends] {msg}", flush=True)


def default_dest() -> Path:
    env = os.environ.get("AMPZIP_BACKENDS")
    return Path(env).expanduser() if env else _paths.FETCH_DIR / "RL_amplitudes_claude"


def clone_pinned(url: str, commit: str, dest: Path, what: str) -> None:
    """Clone `url` into `dest` (or reuse an existing clone) and check out `commit`, detached."""
    if (dest / ".git").exists() or (dest / ".git").is_file():
        head = _head(dest)
        if head.startswith(commit) or commit.startswith(head[: len(commit)]):
            _log(f"{what}: already at {commit} in {dest}")
            return
        _log(f"{what}: existing clone at {head[:7]}, fetching {commit}")
        _git("fetch", "--quiet", "origin", cwd=dest)
    elif dest.exists() and any(dest.iterdir()):
        raise FetchError(f"{dest} is not a Git repository. Choose another directory with --dest.")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _log(f"{what}: cloning {url} -> {dest}")
        _git("clone", "--quiet", url, str(dest))
    _git("checkout", "--quiet", "--detach", commit, cwd=dest)
    _log(f"{what}: checked out {_head(dest)[:7]}")


def apply_sft_patch(dest: Path) -> str:
    """Apply the SFT rollout patch if needed and return its status."""
    before = {f: _sha256(dest / f) for f in SFT_EVAL_SHA256}
    if all(before[f] == SFT_EVAL_SHA256[f][1] for f in SFT_EVAL_SHA256):
        return "already applied"
    for f, (pristine, _) in SFT_EVAL_SHA256.items():
        if before[f] != pristine:
            raise FetchError(
                f"Checksum mismatch for {dest / f} (SHA-256 {before[f][:12]}…). "
                "Patch skipped. Choose another download directory with --dest."
            )
    if not PATCH.exists():
        raise FetchError(f"patch file missing from the installation: {PATCH}")
    _git("apply", "--unidiff-zero", str(PATCH), cwd=dest)
    after = {f: _sha256(dest / f) for f in SFT_EVAL_SHA256}
    bad = [f for f in after if after[f] != SFT_EVAL_SHA256[f][1]]
    if bad:
        raise FetchError(f"patched files have unexpected digests: {bad}")
    return "applied"


def fetch_cds_weights(sh_dir: Path, force: bool = False) -> List[str]:
    try:
        import gdown
    except ImportError as e:
        raise FetchError("gdown is needed to download the CDS weights: pip install 'ampzip[cds]'") from e
    done = []
    for name, (drive_id, sha, size) in CDS_WEIGHTS.items():
        out = sh_dir / name
        if out.exists() and not force:
            if out.stat().st_size == size and _sha256(out) == sha:
                _log(f"cds: {name} present and verified")
                done.append(name)
                continue
            _log(f"cds: file verification failed for {name}. Downloading again.")
        _log(f"cds: downloading {name} ({size / 1e6:.0f} MB) from Google Drive id {drive_id}")
        got = gdown.download(id=drive_id, output=str(out), quiet=False)
        if not got or not out.exists():
            raise FetchError(f"download of {name} failed")
        if out.stat().st_size != size or _sha256(out) != sha:
            raise FetchError(f"{name}: downloaded file does not match the pinned sha256 / size")
        done.append(name)
    return done


def fetch_cds_data(sh_dir: Path, zips_dir: Optional[Path] = None) -> List[int]:
    """Put the CDS test split for n = 4, 5, 6 under sh_dir/data/; verify archive and file digests.

    Archives are read from `zips_dir` when given (any copy of the authors' zips), otherwise
    downloaded to a temporary directory that is removed afterwards."""
    import tempfile
    import zipfile
    done = []
    tmp = None
    try:
        for n, (drive_id, zsha, zsize, member, tsha) in CDS_DATA.items():
            out = sh_dir / "data" / f"n{n}_i1_3" / "data.prefix.counts.test_1000"
            if out.exists() and _sha256(out) == tsha:
                _log(f"cds data: n={n} test split present and verified")
                done.append(n)
                continue
            if zips_dir is not None:
                zp = Path(zips_dir) / f"n{n}_i1_3.zip"
                if not zp.exists():
                    raise FetchError(f"{zp} not found")
            else:
                try:
                    import gdown
                except ImportError as e:
                    raise FetchError("gdown is needed to download the CDS data: pip install 'ampzip[cds]'") from e
                tmp = tmp or Path(tempfile.mkdtemp(prefix="ampzip-cds-"))
                zp = tmp / f"n{n}_i1_3.zip"
                _log(f"cds data: downloading n={n} archive ({zsize / 1e6:.0f} MB) from Google Drive id {drive_id}")
                if not gdown.download(id=drive_id, output=str(zp), quiet=False) or not zp.exists():
                    raise FetchError(f"download of the n={n} archive failed")
            if zp.stat().st_size != zsize or _sha256(zp) != zsha:
                raise FetchError(f"{zp}: archive does not match the pinned sha256 / size")
            out.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp) as z, z.open(member) as src, open(out, "wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
            if _sha256(out) != tsha:
                raise FetchError(f"{out}: extracted test split does not match the pinned sha256")
            _log(f"cds data: n={n} test split extracted and verified")
            done.append(n)
            if tmp is not None:
                zp.unlink()
    finally:
        if tmp is not None:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()
    return done


def write_marker(dest: Path, patch_status: str, cds: List[str], cds_data: List[int]) -> None:
    info = {"rl_url": RL_URL, "rl_commit": _head(dest), "sh_url": SH_URL,
            "sh_commit": _head(dest / "spinorhelicity"), "sft_patch": patch_status,
            "cds_weights": cds, "cds_test_splits": cds_data}
    (dest / MARKER).write_text(json.dumps(info, indent=1) + "\n")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ampzip fetch-backends",
        description="Download the code and model checkpoints for SFT. "
                    "Add --cds to download the CDS checkpoints.",
    )
    p.add_argument("--dest", type=str, default=None,
                   help="Download directory. Defaults to AMPZIP_BACKENDS or the user data directory.")
    p.add_argument("--cds", action="store_true",
                   help="Also download the three CDS checkpoints (about 890 MB).")
    p.add_argument("--force", action="store_true", help="Download the CDS checkpoints again.")
    p.add_argument("--cds-data", action="store_true",
                   help="Download the CDS test datasets used in the paper. "
                        "Downloads 2.7 GB and retains 26 MB.")
    p.add_argument("--cds-zips", type=str, default=None,
                   help="Use local CDS dataset archives n{4,5,6}_i1_3.zip from this directory.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    dest = Path(args.dest).expanduser() if args.dest else default_dest()
    try:
        clone_pinned(RL_URL, RL_COMMIT, dest, "RL_amplitudes_claude")
        clone_pinned(SH_URL, SH_COMMIT, dest / "spinorhelicity", "spinorhelicity")
        status = apply_sft_patch(dest)
        _log(f"sft patch: {status}")
        cds: List[str] = fetch_cds_weights(dest / "spinorhelicity", force=args.force) if args.cds else []
        data: List[int] = []
        if args.cds_data or args.cds_zips:
            data = fetch_cds_data(dest / "spinorhelicity",
                                  Path(args.cds_zips).expanduser() if args.cds_zips else None)
        write_marker(dest, status, cds, data)
    except FetchError as e:
        print(f"[fetch-backends] ERROR: {e}", file=sys.stderr)
        return 1
    _log(f"Downloads complete. Backends saved to {dest}")
    if not args.dest and not os.environ.get("AMPZIP_BACKENDS"):
        _log("AmpZip finds this directory automatically.")
    else:
        _log(f"Run AmpZip with AMPZIP_BACKENDS set to {dest}")
    _log("Dependencies: pip install 'ampzip[sft]' for SFT, or 'ampzip[cds]' for CDS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
