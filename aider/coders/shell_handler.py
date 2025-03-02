from aider.io import ConfirmGroup
from aider.run_cmd import run_cmd

class ShellHandler:
    def __init__(self, coder):
        self.coder = coder
        self.io = coder.io
        self.root = coder.root
        self.suggest_shell_commands = coder.suggest_shell_commands

    def run_shell_commands(self):
        if not self.suggest_shell_commands:
            return ""
        return self._process_commands()

    def _process_commands(self):
        done = set()
        group = ConfirmGroup(set(self.coder.shell_commands))
        accumulated_output = ""
        for command in self.coder.shell_commands:
            if command in done:
                continue
            done.add(command)
            output = self._handle_command(command, group)
            if output:
                accumulated_output += output + "\n\n"
        return accumulated_output

    def _handle_command(self, command, group):
        commands = command.strip().splitlines()
        command_count = sum(1 for cmd in commands if cmd.strip() and not cmd.strip().startswith("#"))
        prompt = "Run shell command?" if command_count == 1 else "Run shell commands?"
        if not self.io.confirm_ask(prompt, subject="\n".join(commands), group=group, allow_never=True):
            return

        return self._execute_commands(commands)

    def _execute_commands(self, commands):
        accumulated_output = ""
        for cmd in commands:
            cmd = cmd.strip()
            if not cmd or cmd.startswith("#"):
                continue
            exit_status, output = self._run_single_command(cmd)
            if output:
                accumulated_output += f"Output from {cmd}\n{output}\n"
        return self._confirm_and_return_output(accumulated_output)

    def _run_single_command(self, command):
        self.io.tool_output()
        self.io.tool_output(f"Running {command}")
        self.io.add_to_input_history(f"/run {command.strip()}")
        return run_cmd(command, error_print=self.io.tool_error, cwd=self.root)

    def _confirm_and_return_output(self, output):
        if output.strip() and self.io.confirm_ask("Add command output to the chat?", allow_never=True):
            num_lines = len(output.strip().splitlines())
            line_plural = "line" if num_lines == 1 else "lines"
            self.io.tool_output(f"Added {num_lines} {line_plural} of output to the chat.")
            return output
