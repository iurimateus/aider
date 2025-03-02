from aider.coders.chat_chunks import ChatChunks

class MessageFormatter:
    @staticmethod
    def get_platform_info(coder):
        import platform
        import os
        from datetime import datetime

        platform_text = f"- Platform: {platform.platform()}\n"
        shell_var = "COMSPEC" if os.name == "nt" else "SHELL"
        shell_val = os.getenv(shell_var)
        platform_text += f"- Shell: {shell_var}={shell_val}\n"

        user_lang = MessageFormatter.get_user_language(coder)
        if user_lang:
            platform_text += f"- Language: {user_lang}\n"

        dt = datetime.now().astimezone().strftime("%Y-%m-%d")
        platform_text += f"- Current date: {dt}\n"

        if coder.repo:
            platform_text += "- The user is operating inside a git repository\n"

        if coder.lint_cmds:
            if coder.auto_lint:
                platform_text += (
                    "- The user's pre-commit runs these lint commands, don't suggest running"
                    " them:\n"
                )
            else:
                platform_text += "- The user prefers these lint commands:\n"
            for lang, cmd in coder.lint_cmds.items():
                if lang is None:
                    platform_text += f"  - {cmd}\n"
                else:
                    platform_text += f"  - {lang}: {cmd}\n"

        if coder.test_cmd:
            if coder.auto_test:
                platform_text += "- The user's pre-commit runs this test command, don't suggest running them: "
            else:
                platform_text += "- The user prefers this test command: "
            platform_text += coder.test_cmd + "\n"

        return platform_text

    @staticmethod
    def get_user_language(coder):
        if coder.chat_language:
            return coder.chat_language

        import locale
        import os

        try:
            lang = locale.getlocale()[0]
            if lang:
                return lang  # Return the full language code, including country
        except Exception:
            pass

        for env_var in ["LANG", "LANGUAGE", "LC_ALL", "LC_MESSAGES"]:
            lang = os.environ.get(env_var)
            if lang:
                return lang.split(".")[
                    0
                ]  # Return language and country, but remove encoding if present

        return None

    @staticmethod
    def fmt_system_prompt(coder, prompt):
        lazy_prompt = coder.gpt_prompts.lazy_prompt if coder.main_model.lazy else ""
        platform_text = MessageFormatter.get_platform_info(coder)

        if coder.suggest_shell_commands:
            shell_cmd_prompt = coder.gpt_prompts.shell_cmd_prompt.format(
                platform=platform_text
            )
            shell_cmd_reminder = coder.gpt_prompts.shell_cmd_reminder.format(
                platform=platform_text
            )
        else:
            shell_cmd_prompt = coder.gpt_prompts.no_shell_cmd_prompt.format(
                platform=platform_text
            )
            shell_cmd_reminder = coder.gpt_prompts.no_shell_cmd_reminder.format(
                platform=platform_text
            )

        if coder.fence[0] == "`" * 4:
            quad_backtick_reminder = "\nIMPORTANT: Use *quadruple* backticks ```` as fences, not triple backticks!\n"
        else:
            quad_backtick_reminder = ""

        prompt = prompt.format(
            fence=coder.fence,
            quad_backtick_reminder=quad_backtick_reminder,
            lazy_prompt=lazy_prompt,
            platform=platform_text,
            shell_cmd_prompt=shell_cmd_prompt,
            shell_cmd_reminder=shell_cmd_reminder,
        )

        if coder.main_model.system_prompt_prefix:
            prompt = coder.main_model.system_prompt_prefix + prompt

        return prompt

    @staticmethod
    def format_chat_chunks(coder):
        chunks = ChatChunks()
        main_sys = MessageFormatter.fmt_system_prompt(
            coder, coder.gpt_prompts.main_system
        )

        if coder.main_model.use_system_prompt:
            chunks.system = [{"role": "system", "content": main_sys}]
        else:
            chunks.system = [
                {"role": "user", "content": main_sys},
                {"role": "assistant", "content": "Ok."},
            ]

        chunks.examples = MessageFormatter._get_example_messages(coder)
        chunks.done = coder.done_messages
        chunks.repo = coder.get_repo_messages()
        chunks.readonly_files = MessageFormatter._get_readonly_files_messages(coder)
        chunks.chat_files = MessageFormatter._get_chat_files_messages(coder)
        chunks.cur = list(coder.cur_messages)

        MessageFormatter._add_reminder_prompt(coder, chunks)
        return chunks

    @staticmethod
    def _get_example_messages(coder):
        if not coder.gpt_prompts.example_messages:
            return []

        return [
            dict(
                role=msg["role"],
                content=MessageFormatter.fmt_system_prompt(coder, msg["content"]),
            )
            for msg in coder.gpt_prompts.example_messages
        ]

    @staticmethod
    def _get_readonly_files_messages(coder):
        messages = []
        read_only_content = coder.get_read_only_files_content()

        if read_only_content:
            messages += [
                {
                    "role": "user",
                    "content": coder.gpt_prompts.read_only_files_prefix
                    + read_only_content,
                },
                {
                    "role": "assistant",
                    "content": "Ok, I will use these files as references.",
                },
            ]

        images_msg = coder.get_images_message(coder.abs_read_only_fnames)
        if images_msg:
            messages += [
                images_msg,
                {
                    "role": "assistant",
                    "content": "Ok, I will use these images as references.",
                },
            ]

        return messages

    @staticmethod
    def _get_chat_files_messages(coder):
        if coder.abs_fnames:
            content = coder.gpt_prompts.files_content_prefix + coder.get_files_content()
            reply = coder.gpt_prompts.files_content_assistant_reply
        else:
            content = coder.gpt_prompts.files_no_full_files
            reply = "Ok."

        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": reply},
        ]

        images_msg = coder.get_images_message(coder.abs_fnames)
        if images_msg:
            messages += [images_msg, {"role": "assistant", "content": "Ok."}]

        return messages

    @staticmethod
    def _add_reminder_prompt(coder, chunks):
        if not coder.gpt_prompts.system_reminder:
            return

        reminder_content = MessageFormatter.fmt_system_prompt(
            coder, coder.gpt_prompts.system_reminder
        )
        if coder.main_model.reminder == "sys":
            chunks.reminder = [{"role": "system", "content": reminder_content}]
        elif (
            coder.main_model.reminder == "user"
            and chunks.cur
            and chunks.cur[-1]["role"] == "user"
        ):
            new_content = f"{chunks.cur[-1]['content']}\n\n{reminder_content}"
            chunks.cur[-1]["content"] = new_content
