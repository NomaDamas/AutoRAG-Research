import re
from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "on-release-main.yml"
SHA_REFERENCE = re.compile(r"^[^@\s]+@[0-9a-fA-F]{40}$")
AUDITED_ACTIONS = {
    "actions/checkout": ("34e114876b0b11c390a56381ad16ebd13914f8d5", "v4"),
    "astral-sh/setup-uv": ("d0cc045d04ccac9d8b7881df0226f9e82c39688e", "v6"),
    "actions/upload-artifact": ("ea165f8d65b6e75b540449e92b4886f43607fa02", "v4"),
    "actions/download-artifact": ("d3f86a106a0bac45b974a628896c90dbdf5c8093", "v4"),
    "pypa/gh-action-pypi-publish": ("cef221092ed1bacb1cc03d23a2d87d1d172e277b", "release/v1"),
}


@pytest.fixture
def workflow_source() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


@pytest.fixture
def workflow(workflow_source: str):
    return yaml.load(workflow_source, Loader=yaml.BaseLoader)  # noqa: S506


def _steps(workflow: dict, job_name: str) -> list[dict]:
    return workflow["jobs"].get(job_name, {}).get("steps", [])


def _run_lines(workflow: dict, job_name: str) -> list[str]:
    return [
        line.strip() for step in _steps(workflow, job_name) for line in step.get("run", "").splitlines() if line.strip()
    ]


def _steps_using(workflow: dict, job_name: str, action: str) -> list[dict]:
    return [step for step in _steps(workflow, job_name) if step.get("uses", "").split("@", maxsplit=1)[0] == action]


def _external_uses(workflow: dict) -> list[str]:
    return [
        step["uses"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "uses" in step and not step["uses"].startswith("./")
    ]


def test_release_workflow_defaults_to_denying_permissions(workflow: dict) -> None:
    # Given the release workflow is loaded with its top-level policy.
    # When the default permission grant is inspected.
    # Then jobs begin with no ambient GitHub token permissions.
    assert workflow.get("permissions") == {}


def test_release_workflow_has_build_publish_deploy_graph(workflow: dict) -> None:
    # Given the release workflow job definitions.
    jobs = workflow["jobs"]

    # When the release dependency graph is inspected.
    # Then the workflow builds, publishes, and deploys in that order.
    assert set(jobs) == {"build", "publish", "deploy-docs"}
    assert "needs" not in jobs.get("build", {})
    assert jobs.get("publish", {}).get("needs") == "build"
    assert jobs.get("deploy-docs", {}).get("needs") == "publish"


def test_release_workflow_scopes_permissions_per_job(workflow: dict) -> None:
    # Given the build, publish, and deploy jobs.
    jobs = workflow["jobs"]

    # When each job's explicit permission scope is inspected.
    # Then every job receives only the access required for its release stage.
    assert jobs.get("build", {}).get("permissions") == {"contents": "read"}
    assert jobs.get("publish", {}).get("permissions") == {"id-token": "write"}
    assert jobs.get("deploy-docs", {}).get("permissions") == {"contents": "write"}


def test_release_workflow_publishes_through_the_pypi_environment(workflow: dict) -> None:
    # Given the publish job's deployment environment.
    environment = workflow["jobs"].get("publish", {}).get("environment", {})

    # When its protection boundary and deployment URL are inspected.
    # Then Trusted Publishing is gated by the named PyPI environment.
    assert environment == {
        "name": "pypi",
        "url": "https://pypi.org/p/autorag-research",
    }


def test_release_workflow_does_not_use_token_secrets_or_sed(workflow_source: str) -> None:
    # Given the unchanged release workflow source.
    # When the source is searched for legacy mutation and token-publishing paths.
    # Then neither path is present in the workflow contract.
    assert "PYPI_TOKEN" not in workflow_source
    assert "UV_PUBLISH_TOKEN" not in workflow_source
    assert "sed" not in workflow_source


def test_release_workflow_versions_once_from_the_release_tag(workflow: dict) -> None:
    # Given the build job's shell commands.
    version_commands = [line for line in _run_lines(workflow, "build") if re.match(r"^uv\s+version\b", line)]

    # When the release-version command is counted.
    # Then exactly one frozen command derives the project version from the release tag.
    assert version_commands == ['uv version --frozen -- "$RELEASE_VERSION"']


def test_release_workflow_passes_the_release_tag_through_step_env(workflow: dict) -> None:
    # Given the build job's release-version step.
    version_steps = [step for step in _steps(workflow, "build") if re.match(r"^uv\s+version\b", step.get("run", ""))]

    # When the command inputs are inspected.
    # Then the untrusted release tag reaches the shell only through an environment variable.
    assert len(version_steps) == 1
    assert version_steps[0].get("env") == {"RELEASE_VERSION": "${{ github.event.release.tag_name }}"}


def test_release_workflow_keeps_expressions_out_of_run_commands(workflow: dict) -> None:
    # Given every shell command in the release workflow.
    run_commands = [
        step.get("run", "") for job in workflow["jobs"].values() for step in job.get("steps", []) if "run" in step
    ]

    # When shell command source is inspected.
    # Then GitHub expressions are not interpolated directly into shell commands.
    assert run_commands
    assert all("${{" not in command for command in run_commands)


def test_release_workflow_disables_setup_uv_caching(workflow: dict) -> None:
    # Given the build and documentation setup-uv steps.
    setup_uv_steps = [
        step for job_name in ("build", "deploy-docs") for step in _steps_using(workflow, job_name, "astral-sh/setup-uv")
    ]

    # When the cache policy is inspected.
    # Then every setup-uv invocation explicitly disables caching.
    assert len(setup_uv_steps) == 2
    assert all(step.get("with", {}).get("enable-cache") == "false" for step in setup_uv_steps)


def test_release_workflow_does_not_persist_build_checkout_credentials(workflow: dict) -> None:
    # Given the build job checkout step.
    checkout_steps = _steps_using(workflow, "build", "actions/checkout")

    # When its credential persistence policy is inspected.
    # Then the build workspace cannot retain the GitHub token for later commands.
    assert len(checkout_steps) == 1
    assert checkout_steps[0].get("with", {}).get("persist-credentials") == "false"


def test_release_workflow_serializes_each_release_tag(workflow: dict) -> None:
    # Given the workflow-level concurrency policy.
    concurrency = workflow.get("concurrency", {})

    # When release grouping and cancellation behavior are inspected.
    # Then each release tag has an independent, non-canceling concurrency group.
    assert concurrency == {
        "group": "release-${{ github.event.release.tag_name }}",
        "cancel-in-progress": "false",
    }


def test_release_workflow_builds_the_release_ref_with_uv_0_9_7(workflow: dict) -> None:
    # Given the build job's checkout and uv setup steps.
    checkout_steps = _steps_using(workflow, "build", "actions/checkout")
    setup_uv_steps = _steps_using(workflow, "build", "astral-sh/setup-uv")

    # When their immutable build inputs are inspected.
    # Then the release tag is checked out and the audited uv version is installed.
    assert len(checkout_steps) == 1
    assert checkout_steps[0].get("with", {}).get("ref") == "${{ github.event.release.tag_name }}"
    assert len(setup_uv_steps) == 1
    assert setup_uv_steps[0].get("with", {}).get("version") == "0.9.7"


def test_release_workflow_builds_once_without_sources(workflow: dict) -> None:
    # Given the build job's shell commands.
    build_commands = [line for line in _run_lines(workflow, "build") if re.match(r"^uv\s+build\b", line)]

    # When the package-build command is counted.
    # Then exactly one source-independent uv build is required.
    assert build_commands == ["uv build --no-sources"]


def test_release_workflow_transfers_the_built_distribution(workflow: dict) -> None:
    # Given the build and publish job steps.
    upload_steps = _steps_using(workflow, "build", "actions/upload-artifact")
    download_steps = _steps_using(workflow, "publish", "actions/download-artifact")

    # When the artifact handoff is inspected.
    # Then the publish job downloads the distribution produced by the build job.
    assert len(upload_steps) == 1
    assert len(download_steps) == 1
    assert upload_steps[0]["with"]["path"] == "dist"
    assert download_steps[0]["with"]["name"] == upload_steps[0]["with"]["name"]
    assert download_steps[0]["with"]["path"] == "dist"


def test_release_workflow_uses_pypa_trusted_publishing_with_attestations(workflow: dict) -> None:
    # Given the publish job steps.
    publisher_steps = _steps_using(workflow, "publish", "pypa/gh-action-pypi-publish")

    # When the package publisher is inspected.
    # Then PyPI Trusted Publishing is used and attestations are not disabled.
    assert len(publisher_steps) == 1
    publisher = publisher_steps[0]
    assert re.fullmatch(r"pypa/gh-action-pypi-publish@[0-9a-fA-F]{40}", publisher["uses"])
    assert publisher.get("with", {}).get("packages-dir") == "dist"
    assert publisher.get("with", {}).get("attestations") == "true"


def test_release_workflow_deploys_docs_with_pinned_uv_environment(workflow: dict) -> None:
    # Given the documentation deployment job.
    checkout_steps = _steps_using(workflow, "deploy-docs", "actions/checkout")
    setup_uv_steps = _steps_using(workflow, "deploy-docs", "astral-sh/setup-uv")
    run_lines = _run_lines(workflow, "deploy-docs")

    # When its toolchain and commands are inspected.
    # Then it installs Python and locked dependencies before the existing MkDocs deployment.
    assert len(checkout_steps) == 1
    assert len(setup_uv_steps) == 1
    assert checkout_steps[0].get("with", {}).get("persist-credentials", "true") == "true"
    assert setup_uv_steps[0].get("with", {}).get("version") == "0.9.7"
    assert setup_uv_steps[0].get("with", {}).get("python-version")
    assert "uv sync --frozen --all-extras" in run_lines
    assert "uv run mkdocs gh-deploy --force" in run_lines


def test_release_workflow_pins_every_external_action_to_audited_commit(
    workflow: dict,
    workflow_source: str,
) -> None:
    # Given every external action reference in the release workflow.
    external_uses = _external_uses(workflow)

    # When each reference and its review comment are checked.
    # Then every external action uses exactly an audited commit with its version comment.
    assert external_uses
    assert all(SHA_REFERENCE.fullmatch(reference) for reference in external_uses)
    expected_references = {
        f"{action}@{sha}": version_comment for action, (sha, version_comment) in AUDITED_ACTIONS.items()
    }
    assert set(external_uses) == set(expected_references)

    for reference, version_comment in expected_references.items():
        pinned_lines = re.findall(
            rf"(?m)^\s*uses:\s+{re.escape(reference)}\s+#\s+{re.escape(version_comment)}\s*$",
            workflow_source,
        )
        assert len(pinned_lines) == external_uses.count(reference)
