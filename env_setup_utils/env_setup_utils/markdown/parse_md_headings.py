import re
from typing import List, Optional, Set, TypedDict

import mistune
from bs4 import BeautifulSoup  # type: ignore[import-untyped]
from markdownify import markdownify as md  # type: ignore[import-untyped]


class Heading(TypedDict):
    heading: str
    contents: str
    heading_level: int


def is_heading(name: Optional[str]) -> bool:
    if name is None:
        return False
    return bool(re.match(r"^h[1-7]$", name))


def extract_all_headings(md_text: str, heading_level: Optional[int] = None) -> List[Heading]:
    """Parses markdown and extracts content under all specified headings."""
    html = mistune.html(md_text)
    soup = BeautifulSoup(html, "html.parser")  # type: ignore[reportArgumentType]
    headings = soup.find_all(
        [f"h{heading_level}"] if heading_level else [f"h{heading_level}" for heading_level in range(1, 7)]
    )

    results: List[Heading] = []
    for heading in headings:
        heading_text = heading.text.strip()
        content = []
        sibling = heading.find_next_sibling()
        while sibling and (not is_heading(sibling.name) or int(sibling.name[1:]) > int(heading.name[1:])):
            content.append(md(str(sibling), heading_style="ATX"))
            sibling = sibling.find_next_sibling()
        if len(content) > 0:
            results.append(
                {"heading": heading_text, "contents": "".join(content).strip(), "heading_level": int(heading.name[1:])}
            )
    return results


def extract_headings_with_keywords(md_text: str, keywords: Set[str]) -> List[Heading]:
    """Parses markdown and extracts content under all headings containing specified keywords."""
    all_headings = extract_all_headings(md_text)
    all_headings.sort(key=lambda x: x["heading_level"])

    headings_with_keywords: List[Heading] = []
    i = 0
    included_heading_levels: Set[int] = set()
    while i < len(all_headings):
        cur_heading = all_headings[i]
        if any(keyword in cur_heading["heading"].lower() for keyword in keywords):
            if not included_heading_levels or cur_heading["heading_level"] in included_heading_levels:
                headings_with_keywords.append(cur_heading)
            else:
                if any(
                    cur_heading["contents"] in build_heading["contents"] for build_heading in headings_with_keywords
                ):
                    i += 1
                    continue
                headings_with_keywords.append(cur_heading)

            included_heading_levels.add(cur_heading["heading_level"])
        i += 1

    return headings_with_keywords
