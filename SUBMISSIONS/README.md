# Packaging submissions

Downstream packaging for ReAgent v0.2.0. One directory per ecosystem.

ReAgent is a Python package with a pinned interpreter, not a standalone
binary, which is what shapes every decision below.

| | Status | Verified |
|---|---|---|
| [`aur/`](aur/) | Ready to submit | Built and 127 tests run on this machine |
| [`chocolatey/`](chocolatey/) | Ready to review | Structure only, no Windows machine |
| winget | Not submitted | See below |

## Why the interpreter is bundled

`requires-python` is `>=3.10,<3.12`, and that ceiling is not arbitrary:

| Dependency | Supports |
|---|---|
| aizynthfinder 4.4.1 | 3.10 to 3.12 |
| rdkit 2023.9.6 | manylinux and win wheels, no 3.13+ |
| onnxruntime | 3.11+ |

Arch ships Python 3.14. Chocolatey's default Python is also newer. Neither can
run ReAgent, so both packages create their own 3.11 virtualenv rather than
installing into system site-packages.

There is a second reason on Arch specifically: RDKit and AiZynthFinder have no
Arch package at all, in either the official repositories or the AUR, so there
is nothing to put in `depends`.

## aur/

Installs a self-contained venv to `/opt/reagent` and puts two wrappers on PATH,
`reagent` and `reagent-download-data`.

Dependency versions come from `requirements.lock`, generated from the
repository's `uv.lock` and pinned by digest. `pip --require-hashes` refuses
anything that does not match, so the build cannot drift if a release is yanked
or re-uploaded.

Build and install:

```sh
cd aur
makepkg -si
```

Note that `python311` is itself an AUR package, so an AUR helper is the easier
route. The package is large: 395 MB compressed, 1.5 GB installed. That is
RDKit, ONNX Runtime, SciPy, PyArrow and llvmlite, all vendored.

`namcap` is clean on the PKGBUILD. On the built package it reports ELF files
outside `/usr` and unstripped binaries. Both are inherent: the venv lives in
`/opt` by design, and `!strip` is set deliberately, since the `.so` files come
from upstream manylinux wheels and stripping vendored wheels risks breaking
them for no size win.

## chocolatey/

Depends on the `python311` package, creates a venv under the package's `tools`
directory, and shims `reagent` and `reagent-download-data`.

The wheel is fetched from the GitHub release with a pinned sha256, not from
PyPI. `reagent` on PyPI is Facebook's reinforcement learning library, an
unrelated project, so `pip install reagent` fetches the wrong thing.

Dependency versions are pinned in `tools/requirements.txt` from the same
`uv.lock`, but without hashes. That differs from the AUR lock on purpose: the
export resolves per platform, and requiring hashes would reject the Windows
wheels pip correctly selects.

Build and install it locally with:

```powershell
cd SUBMISSIONS\chocolatey
choco pack
choco install reagent -s ".;https://community.chocolatey.org/api/v2/" -y
```

Both sources are needed. The local folder holds `reagent.nupkg`; the community
feed holds the `python311` dependency, which `-s .` alone cannot resolve.

The `Packaging` workflow runs exactly this on a Windows runner whenever
`SUBMISSIONS/` changes, so the package is no longer untested.

## winget

Not submitted, deliberately.

winget installs an artifact: an msi, an exe, or a zip containing one. It has no
concept of a Python package, and the release ships a wheel and an sdist. Making
a winget submission real would mean adding a PyInstaller build to CI to produce
a portable zip, which for RDKit plus ONNX Runtime is a 300 to 500 MB artifact
that no one here can test.

There is also a fit question. winget expects an application that works once
installed. ReAgent needs a 760 MB model download and a stock-cache build before
it can plan anything, which is not what the store front-end implies.

Revisit if a bundled Windows build ever becomes worth maintaining.

## Updating for a new release

Both packages carry the version in three places, and all three must move
together:

| File | What to change |
|---|---|
| `aur/PKGBUILD` | `pkgver`, and the sdist sha256 in `sha256sums` |
| `aur/requirements.lock` | regenerate, see below |
| `chocolatey/reagent.nuspec` | `<version>`, `<releaseNotes>` |
| `chocolatey/tools/chocolateyinstall.ps1` | wheel URL and `checksum` |
| `chocolatey/tools/requirements.txt` | regenerate, see below |

Regenerate both dependency files from `uv.lock`:

```sh
NO_COLOR=1 uv export --no-dev --no-emit-project --format requirements-txt \
    --no-header --quiet > SUBMISSIONS/aur/requirements.lock

NO_COLOR=1 uv export --no-dev --no-emit-project --format requirements-txt \
    --no-header --no-hashes --quiet \
  | sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' \
  > SUBMISSIONS/chocolatey/tools/requirements.txt
```

Then refresh the AUR checksums and metadata:

```sh
cd SUBMISSIONS/aur
sha256sum requirements.lock reagent.sh reagent-download-data.sh
curl -sL https://github.com/rugbedbugg/ReAgent/releases/download/vX.Y.Z/reagent-X.Y.Z.tar.gz | sha256sum
makepkg --printsrcinfo > .SRCINFO
```

`.SRCINFO` is generated, never edited by hand. The AUR rejects a push where it
disagrees with the PKGBUILD.
