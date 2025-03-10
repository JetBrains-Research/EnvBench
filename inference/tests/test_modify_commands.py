from src.utils.modify_commands import add_flag_to_command


def test_modify_pyenv_command():
    # -f is not there
    assert add_flag_to_command("pyenv install 3.9.7", target_flag="f") == "pyenv install -f 3.9.7"
    assert add_flag_to_command("pyenv install -smth 3.9.7", target_flag="f") == "pyenv install -smthf 3.9.7"
    assert add_flag_to_command("pyenv install -v -smth 3.9.7", target_flag="f") == "pyenv install -vf -smth 3.9.7"

    # -f is there
    assert add_flag_to_command("pyenv install -f 3.9.7", target_flag="f") == "pyenv install -f 3.9.7"
    assert add_flag_to_command("pyenv install -fsmth 3.9.7", target_flag="f") == "pyenv install -fsmth 3.9.7"
    assert add_flag_to_command("pyenv install -smthf 3.9.7", target_flag="f") == "pyenv install -smthf 3.9.7"
    assert add_flag_to_command("pyenv install -smthfsmth 3.9.7", target_flag="f") == "pyenv install -smthfsmth 3.9.7"
    assert add_flag_to_command("pyenv install -v -f -smth 3.9.7", target_flag="f") == "pyenv install -v -f -smth 3.9.7"

    # --force is there
    assert (
        add_flag_to_command("pyenv install --force 3.9.7", target_flag="f", target_flag_long="force")
        == "pyenv install --force 3.9.7"
    )
