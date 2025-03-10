from typing import List, Optional, Tuple, no_type_check

# copied from: https://github.com/coinse/installamatic/blob/0ba511dd49a08ed41d495c22d0e063f3e73464e3/install_test/agent/functions.py#L15C1-L15C74
NON_NL = [".py", "requirements", ".toml", ".yaml", "Dockerfile", ".lock"]


# copied from: https://github.com/coinse/installamatic/blob/0ba511dd49a08ed41d495c22d0e063f3e73464e3/install_test/agent/functions.py#L166-L206
@no_type_check
def get_headings(file: str) -> Optional[List[Tuple[str, str]]]:
    "get a list of all section heading, section content pairs from the given file"
    lines = file.split("\n")
    headings = []
    code_block = False
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("```"):
            code_block = not code_block
        elif line.startswith("#") and not code_block:
            headings.append((line, i))
    if len(headings) == 0:
        return None

    headings = [
        (
            heading[0].replace("#", "").strip(),
            heading[1],
            (3 if heading[0].startswith("###") else 2 if heading[0].startswith("##") else 1),
        )
        for heading in headings
    ]

    sections = [("", lines[: headings[0][1]])]
    sections = sections + [
        (
            heading[0],
            "\n".join(lines[heading[1] + 1 : (headings[i + 1][1] if i + 1 < len(headings) else None)]),
        )
        for i, heading in enumerate(headings[:-1])
    ]
    return sections


# copied from: https://github.com/coinse/installamatic/blob/0ba511dd49a08ed41d495c22d0e063f3e73464e3/install_test/agent/functions.py#L209C1-L227C20
@no_type_check
def get_headings_rst(file: str) -> Optional[List[Tuple[str, str]]]:
    "get a list of all section heading, section content pairs from the given file"
    lines = file.split("\n")
    headings = [
        i
        for i, line in enumerate(lines[1:])
        if not line.startswith(" ")
        and line.strip() != ""
        and (all(ch == "=" for ch in line.strip()) or all(ch == "-" for ch in line.strip()))
    ]

    sections = []
    if headings[0] != 0:
        sections = [("", "\n".join(lines[: headings[0]]))]
    sections.extend((lines[prev], "\n".join(lines[prev + 2 : curr])) for prev, curr in zip(headings, headings[1:]))
    return sections
