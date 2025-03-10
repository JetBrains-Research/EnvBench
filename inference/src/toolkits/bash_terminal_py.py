from .bash_terminal import BashTerminalToolkit


class PythonBashTerminalToolkit(BashTerminalToolkit):
    def initial_commands(self):
        return super().initial_commands() + ['eval "$(pyenv init -)"']
