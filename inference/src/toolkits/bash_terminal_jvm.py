from .bash_terminal import BashTerminalToolkit


class JVMBashTerminalToolkit(BashTerminalToolkit):
    def initial_commands(self):
        # hack: adding sdkman as a command
        #  ideally bash should be in interactive mode
        return super().initial_commands() + [
            "[ -f '/root/.sdkman/bin/sdkman-init.sh' ] && source /root/.sdkman/bin/sdkman-init.sh || true"
        ]
