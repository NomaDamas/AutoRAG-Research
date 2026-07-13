from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_project_urls_use_canonical_organization_mapping() -> None:
    # Given the project's machine-readable TOML metadata.
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        project_metadata = tomllib.load(pyproject_file)

    # When the project URLs are read from the parsed metadata.
    project_urls = project_metadata["project"]["urls"]

    # Then the exact canonical URL mapping is present.
    assert project_urls == {
        "Homepage": "https://nomadamas.github.io/AutoRAG-Research/",
        "Repository": "https://github.com/NomaDamas/AutoRAG-Research",
        "Documentation": "https://nomadamas.github.io/AutoRAG-Research/",
    }
