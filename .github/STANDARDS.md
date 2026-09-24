# Repository standards baseline

This is ReAgent's reference implementation of the shared CI/CD/README baseline.
Reuse the structure when adapting another GitHub-published repository, with its
own runtime pins and meaningful checks. CI is required; CD is optional.

| Area | Required structure | ReAgent implementation |
| --- | --- | --- |
| CI triggers | Pull requests and maintained branches, optional manual/reusable entry | `workflows/ci.yml` |
| Setup | Declared toolchain, isolated dependencies, appropriate caches | mise tasks, uv environment, Python 3.10/3.11 matrix |
| Validation | Blocking lint, tests, build/package checks as applicable | Ruff, pytest, sdist tests, wheel smoke, metadata checks |
| Permissions | Read-only validation; narrowly scoped publish writes | Workflow defaults and publish job override |
| Scheduling | Cancel obsolete CI; preserve active publication | CI ref groups, serialized manual packaging, release tag groups |
| Release | Verify version, require CI, reuse checked artifacts | `workflows/release.yml` calls `ci.yml` |
| Provenance | Checksums tied to validated build, verify remote artifacts | SHA256SUMS uploaded with distributions and checked after download |
| Package publication | Validate installers and protect credentials | Manual Chocolatey environment; tested nupkg artifact reuse |
| README | Purpose, install, quick start, usage/configuration, development/tests, license, CI/release links | Root README |

Adapt this baseline to research/data/profile repositories without fake tests or
irrelevant product sections. Do not change version constraints to adopt it.
For future workflow edits, validate YAML/action expressions, execute the affected
mise tasks, and require the GitHub matrix to pass before integration. Check that
release artifacts come from those checks, not a second untested build.

A checksum manifest records artifact integrity; it is not a signed provenance
attestation. Environment reviewer and branch-protection settings are hosted
configuration and must be verified separately when changing those controls.
