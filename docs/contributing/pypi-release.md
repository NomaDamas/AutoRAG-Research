# PyPI Release

AutoRAG-Research publishes Python distributions through the `release-main` GitHub Actions workflow at
`.github/workflows/on-release-main.yml`. Publishing a GitHub Release starts the workflow; publishing a tag or
running a workflow manually does not.

## Trusted Publisher Setup

The PyPI project must have this GitHub Trusted Publisher configured:

| Field | Value |
|-------|-------|
| Owner | `NomaDamas` |
| Repository | `AutoRAG-Research` |
| Workflow | `on-release-main.yml` |
| Environment | `pypi` |

The `pypi` environment name must also exist in GitHub and must match the publisher configuration exactly. Keep any
required reviewers on that environment limited to release maintainers who are authorized to approve a PyPI publish.

### Maintainer permissions

- The release actor needs GitHub repository **write** access to create and publish a release.
- A protected `pypi` environment may require approval from one of its configured reviewers before the publish job can
  run.
- A PyPI project owner, or another account authorized to manage the project's publishing settings, must configure the
  Trusted Publisher. After configuration, the workflow's trusted publisher identity is the upload principal; maintainers
  do not need a PyPI token.
- The workflow needs `id-token: write` only on the publish job. No long-lived PyPI credential is required.

## Release Procedure

### 1. Use a PEP 440 version tag

The workflow passes the GitHub release tag to `uv version --frozen`, so the tag must be a valid
[PEP 440](https://peps.python.org/pep-0440/) version.
Use a version such as `0.2.0` or `0.2.0rc1`; do not use labels such as `latest` or `release-0.2.0`.

Check the tag and release notes before publishing the GitHub Release. Once the release is published, the workflow
checks out that tag and derives the package version from it.

### 2. Publish the GitHub Release

The `published` release event runs the workflow in this order:

1. **Build once.** The `build` job checks out the release tag, sets the version, runs `uv build --no-sources`, and
   uploads the resulting `dist/` directory as the `python-package-distributions` artifact.
2. **Publish the same artifact.** The `publish` job downloads that artifact, enters the `pypi` environment, and
   calls `pypa/gh-action-pypi-publish` with `id-token: write`. PyPI Trusted Publishing exchanges the GitHub OIDC
   identity for short-lived publish authorization; the workflow does not use `PYPI_TOKEN`.
3. **Attach provenance.** The publish action is configured with `attestations: true`, so the uploaded distributions
   receive [PEP 740](https://peps.python.org/pep-0740/) digital attestations tied to the publishing identity.
4. **Deploy documentation.** The existing documentation deployment runs only after the publish job succeeds.

The publish job does not rebuild the package. The uploaded artifact is the release input and must be treated as
immutable once it has been handed to the publish job.

## Verify PyPI Provenance

For a published distribution, copy its direct file URL from the PyPI release page and run the PyPI attestation
verifier:

```bash
WHEEL_DIRECT_URL="https://files.pythonhosted.org/.../autorag_research-<version>-<tags>.whl"
pypi-attestations verify pypi \
  --repository https://github.com/NomaDamas/AutoRAG-Research \
  "$WHEEL_DIRECT_URL"
```

The command downloads the distribution and its provenance, checks that the trusted publisher identifies this
repository, and cryptographically verifies the distribution against its attestations. Repeat the check for the source
archive when both distribution types are present. The provenance JSON for a file is also available through the PyPI
Integrity API at:

```text
https://pypi.org/integrity/autorag-research/<version>/<filename>/provenance
```

The repository, workflow, and environment shown in the provenance must correspond to the configured publisher tuple
above. A missing or mismatched provenance record is a release incident; do not treat a successful upload alone as
proof of provenance.

## Updating Action SHAs

All external actions in the release workflow are pinned to full commit SHAs. Update a pin as follows:

1. Review the upstream action release notes and the commit that the release is intended to reference.
2. Resolve the selected ref to a full 40-character commit SHA. For an annotated tag, use the peeled `^{}` result:

   ```bash
   git ls-remote https://github.com/actions/checkout.git \
     refs/tags/v4 'refs/tags/v4^{}'
   ```

   For a maintained branch, query the exact `refs/heads/<branch>` ref. For example, resolve the PyPA publish action's
   `release/v1` branch with:

   ```bash
   git ls-remote https://github.com/pypa/gh-action-pypi-publish.git \
     refs/heads/release/v1
   ```

   Branches move, so review the resolved commit and its upstream changes before pinning that commit.

3. Update the `uses:` line and its version comment in `.github/workflows/on-release-main.yml`.
4. Update the matching entry in `AUDITED_ACTIONS` in `tests/test_release_workflow.py` so the workflow and its audit
   test remain in agreement.
5. Run both checks from the repository root:

   ```bash
   uv run pytest tests/test_release_workflow.py
   actionlint .github/workflows/on-release-main.yml
   ```

Do not replace a reviewed SHA with a moving branch or tag reference, and do not update only the workflow while leaving
the audit test stale.

## Tokenless Recovery

If the publish job fails because PyPI cannot match the GitHub identity, repair the configuration rather than adding a
credential:

1. Compare the PyPI publisher with the exact tuple `NomaDamas` / `AutoRAG-Research` / `on-release-main.yml` /
   `pypi`.
2. Check that the publish job still names the GitHub environment `pypi`, has `id-token: write`, and invokes the PyPA
   publish action without a password.
3. If the `build` job succeeded, rerun the failed `publish` job from the same workflow run. It must download and use
   the existing `python-package-distributions` artifact; do not rerun a modified build to produce replacement files.

Never restore `PYPI_TOKEN` or add another long-lived PyPI token path as a recovery measure. If the publisher or
environment mismatch cannot be repaired safely, stop the release and escalate with the workflow run and PyPI project
details.

## Partial or Failed Releases

Treat every distribution filename and its bytes as immutable after any file may have reached PyPI:

- Preserve the successful build artifact and its hashes.
- Inspect which files are already present on PyPI before retrying.
- Retry only with the same build artifact and do not rebuild, overwrite, or silently replace an existing filename.
- If the original artifact cannot safely complete the release, stop and investigate instead of creating a second build
  for the same version.

## Rollback

Do not delete a PyPI version and do not reuse its version number for different package contents. Roll back a bad
release by yanking the affected version when appropriate, then publish the fix as a new valid PEP 440 version with a
new build and new attestations.

## References

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [PyPI digital attestations](https://docs.pypi.org/attestations/)
- [PyPI Integrity API](https://docs.pypi.org/api/integrity/)
- [PEP 440](https://peps.python.org/pep-0440/)
- [PEP 740](https://peps.python.org/pep-0740/)
