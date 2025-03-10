import re
import shlex
from typing import Optional


def add_flag_to_command(command: str, target_flag: str, target_flag_long: Optional[str] = None) -> str:
    split_command = shlex.split(command)

    flags_indices = []
    for i, part in enumerate(split_command):
        if re.match(r"^-\w*$", part):
            flags_indices.append(i)
            if target_flag in part:
                return command
        if target_flag_long:
            if re.match(r"^--\w*$", part):
                if target_flag_long in part:
                    return command

    if not flags_indices:
        split_command.insert(-1, f"-{target_flag}")
        return shlex.join(split_command)

    split_command[flags_indices[0]] += target_flag
    return shlex.join(split_command)
