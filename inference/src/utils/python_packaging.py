import logging
import re
from typing import List, Optional

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def is_poetry_constraint(specifier: str) -> bool:
    """
    Check if the given specifier uses Poetry-style syntax (^ or ~).
    """
    return bool(re.match(r"^[\^~]", specifier))


def translate_poetry_specifier(specifier: str) -> str:
    """
    Translate a Poetry-style specifier to a PEP 440-compatible specifier.
    """
    match = re.match(r"^\^(\d+)\.(\d+)", specifier)
    if match:
        major, minor = match.groups()
        return f">={major}.{minor},<{int(major) + 1}.0"

    match = re.match(r"^~(\d+)\.(\d+)", specifier)
    if match:
        major, minor = match.groups()
        return f">={major}.{minor},<{major}.{int(minor) + 1}"

    return specifier


def normalize_specifier(specifier: str) -> str:
    """
    Normalize a version specifier string.
    If it's a bare version (e.g., '3.8.12'), add '=='.
    """
    if is_poetry_constraint(specifier):
        return translate_poetry_specifier(specifier)

    if specifier and not any(specifier.startswith(op) for op in ["==", ">=", "<=", "~=", "^", ">", "<"]):
        return f"=={specifier}"
    return specifier


def filter_valid_versions(raw_versions: List[str]) -> List[str]:
    """
    Filter and return only PEP 440-compliant Python versions.
    """
    valid_versions = []
    for version in raw_versions:
        if re.match(r"^\d+\.\d+(\.\d+)?$", version):
            valid_versions.append(version)
    return valid_versions


def select_python_version(requirements: List[str], raw_versions: List[str]) -> Optional[str]:
    """
    Select the optimal Python version based on given requirements.

    Args:
        requirements (List[str]): A list of Python version requirement strings.
        raw_versions (List[str]): A raw list of available versions from pyenv.

    Returns:
        Optional[str]: The best matching Python version, or None if no match is found.
    """
    try:
        valid_versions = filter_valid_versions(raw_versions)
        combined_specifier = SpecifierSet(",".join([normalize_specifier(requirement) for requirement in requirements]))
        matching_versions = [v for v in valid_versions if Version(v) in combined_specifier]

        if not matching_versions:
            return None

        return max(matching_versions, key=Version)

    except Exception as e:
        print(f"Error: {e}")
        return None


def is_python_version_compatible(requirement: str, installed_version: str) -> bool:
    """
    Check if the installed Python version satisfies the version requirement.

    Args:
        requirement (str): Version requirement string (e.g., '>=3.8,<3.11').
        installed_version (str): Installed Python version (e.g., '3.9.7').

    Returns:
        bool: True if the installed version satisfies the requirement, False otherwise.
    """
    try:
        specifiers = SpecifierSet(normalize_specifier(requirement))
        version = Version(installed_version)
        return version in specifiers
    except Exception as e:
        logging.exception(f"Couldn't match Python version requirement and installed version: {e}")
        return False
