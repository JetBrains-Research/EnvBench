from textwrap import dedent
from typing import Dict, List

import pytest

from env_setup_utils.markdown import extract_all_headings


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            dedent("""
                         # First-level heading
                         
                         something
                         
                         ## Second-level heading 1
                         
                         something
                         
                         ## Second-level heading 2
                         
                         something
                         
                         ### Third-level heading
                         
                         something"""),
            [
                {"heading": "Second-level heading 1", "contents": "something", "heading_level": 2},
                {
                    "heading": "Second-level heading 2",
                    "contents": "something\n\n### Third-level heading\n\nsomething",
                    "heading_level": 2,
                },
            ],
        )
    ],
)
def test_specified_heading_level(test_input: str, expected: List[Dict[str, str]]):
    assert extract_all_headings(test_input, heading_level=2) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            dedent("""
                         # First-level heading

                         something

                         ## Second-level heading 1

                         something

                         ## Second-level heading 2

                         something

                         ### Third-level heading

                         something"""),
            [
                {
                    "heading": "First-level heading",
                    "contents": "something\n\n## Second-level heading 1\n\nsomething\n\n## Second-level heading 2\n\nsomething\n\n### Third-level heading\n\nsomething",
                    "heading_level": 1,
                },
                {"heading": "Second-level heading 1", "contents": "something", "heading_level": 2},
                {
                    "heading": "Second-level heading 2",
                    "contents": "something\n\n### Third-level heading\n\nsomething",
                    "heading_level": 2,
                },
                {"heading": "Third-level heading", "contents": "something", "heading_level": 3},
            ],
        ),
        (
            dedent("""
                         # First-level heading 1

                         something

                         ## Second-level heading 1

                         something

                         # First-level heading 2

                         something

                         ### Third-level heading

                         something"""),
            [
                {
                    "heading": "First-level heading 1",
                    "contents": "something\n\n## Second-level heading 1\n\nsomething",
                    "heading_level": 1,
                },
                {"heading": "Second-level heading 1", "contents": "something", "heading_level": 2},
                {
                    "heading": "First-level heading 2",
                    "contents": "something\n\n### Third-level heading\n\nsomething",
                    "heading_level": 1,
                },
                {"heading": "Third-level heading", "contents": "something", "heading_level": 3},
            ],
        ),
    ],
)
def test_all_heading_level(test_input: str, expected: List[Dict[str, str]]):
    assert extract_all_headings(test_input) == expected
