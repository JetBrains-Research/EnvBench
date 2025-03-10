from textwrap import dedent
from typing import Dict, List

import pytest

from env_setup_utils.markdown import extract_headings_with_keywords


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            dedent("""
                    # Build heading 1
                    
                    ## Build heading 2
                    
                    something 1
                    
                    ## Random heading 1
                    
                    something 2
                    
                    # Build heading 3
                    
                    something 3
                    
                    # Random heading 2
                    
                    ## Build heading 4
                    
                    something 4
                    
                    # Random heading 3
                    
                    ## Random heading 4
                    
                    something 5
                    """),
            [
                {
                    "heading": "Build heading 1",
                    "contents": "## Build heading 2\n\nsomething 1\n\n## Random heading 1\n\nsomething 2",
                    "heading_level": 1,
                },
                {
                    "heading": "Build heading 3",
                    "contents": "something 3",
                    "heading_level": 1,
                },
                {
                    "heading": "Build heading 4",
                    "contents": "something 4",
                    "heading_level": 2,
                },
            ],
        )
    ],
)
def test_all_heading_level(test_input: str, expected: List[Dict[str, str]]):
    assert extract_headings_with_keywords(test_input, {"build"}) == expected
