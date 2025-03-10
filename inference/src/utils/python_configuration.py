import ast
import configparser
from io import StringIO
from typing import List, Optional

import tomllib  # type: ignore[import-not-found]

from src.agents.python_baseline.constants import BUILD_BACKEND_TO_DM_MAP


def get_dm_from_pyproject_toml(file_contents: str) -> Optional[str]:
    # https://packaging.python.org/en/latest/guides/writing-pyproject-toml
    try:
        toml_contents = tomllib.loads(file_contents)
        build_system = toml_contents.get("build-system")
        if build_system is None:
            return None
        build_backend = build_system.get("build-backend")
        if build_backend in BUILD_BACKEND_TO_DM_MAP:
            return BUILD_BACKEND_TO_DM_MAP[build_backend]
        return None
    except tomllib.TOMLDecodeError:
        return None


def get_extras_from_pyproject_toml_poetry(file_contents: str) -> List[str]:
    """
    Parse the given pyproject.toml contents and return the names of all
    optional Poetry dependency groups.

    This function handles:
      1. The newer syntax, e.g.:

         [tool.poetry.group.dev]
         optional = true
         [tool.poetry.group.dev.dependencies]
         pytest = "^6.2"

      2. The older syntax for dev dependencies, e.g.:

         [tool.poetry.dev-dependencies]
         pytest = "^6.0.0"
         pytest-mock = "*"

         This older style will be treated as a dev group
         and considered optional by default.

         https://python-poetry.org/docs/1.8/managing-dependencies/#dependency-groups
    """
    try:
        data = tomllib.loads(file_contents)
        poetry_config = data.get("tool", {}).get("poetry", {})
        optional_groups = []

        # 1) Handle older syntax for dev dependencies.
        #    If [tool.poetry.dev-dependencies] exists, treat that as an optional "dev" group.
        if "dev-dependencies" in poetry_config:
            optional_groups.append("dev")

        # 2) Handle the newer group-based syntax.
        groups = poetry_config.get("group", {})
        for group_name, group_config in groups.items():
            # If the group's metadata includes "optional = true", consider it optional.
            if isinstance(group_config, dict) and group_config.get("optional", False) is True:
                optional_groups.append(group_name)

        return optional_groups
    except tomllib.TOMLDecodeError:
        return []


def get_extras_from_pyproject_toml(file_contents: str) -> List[str]:
    # https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#dependencies-optional-dependencies
    try:
        toml_contents = tomllib.loads(file_contents)
        optional_dependencies = toml_contents.get("project", {}).get("optional-dependencies", [])
        if not optional_dependencies:
            return []
        return [dep for dep in optional_dependencies]
    except tomllib.TOMLDecodeError:
        return []


def get_extras_from_setup_py(file_contents: str) -> List[str]:
    # https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies
    tree = ast.parse(file_contents)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup":
            for keyword in node.keywords:
                if keyword.arg == "extras_require" and isinstance(keyword.value, ast.Dict):
                    return [key.s for key in keyword.value.keys if isinstance(key, ast.Constant)]

    return []


def get_extras_from_setup_cfg(file_contents: str) -> List[str]:
    config = configparser.ConfigParser()
    config.read(StringIO(file_contents))
    if "options.extras_require" in config:
        return list(config["options.extras_require"].keys())
    else:
        return []
