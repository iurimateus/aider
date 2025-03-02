from aider.repo import ANY_GIT_ERROR


class GitManager:
    def __init__(self, coder):
        self.coder = coder
        self.repo = coder.repo
        self.io = coder.io
        self.show_diffs = coder.show_diffs
        self.gpt_prompts = coder.gpt_prompts
        
    def show_auto_commit_outcome(self, res):
        commit_hash, commit_message = res
        self.coder.last_aider_commit_hash = commit_hash
        self.coder.aider_commit_hashes.add(commit_hash)
        if self.show_diffs:
            self.coder.commands.cmd_diff()

    def auto_commit(self, edited, context=None):
        if not self.repo or not self.coder.auto_commits or self.coder.dry_run:
            return

        if not context:
            context = self._get_context_from_history()

        try:
            res = self.repo.commit(fnames=edited, context=context, aider_edits=True)
            if res:
                self.show_auto_commit_outcome(res)
                commit_hash, commit_message = res
                return self.gpt_prompts.files_content_gpt_edits.format(
                    hash=commit_hash,
                    message=commit_message,
                )
            return self.gpt_prompts.files_content_gpt_no_edits
        except ANY_GIT_ERROR as err:
            self.io.tool_error(f"Unable to commit: {str(err)}")

    def _get_context_from_history(self):
        return self.coder.get_context_from_history(self.coder.cur_messages)

    def dirty_commit(self):
        if not self.coder.need_commit_before_edits:
            return
        if not self.coder.dirty_commits:
            return
        if not self.repo:
            return
        self.repo.commit(fnames=self.coder.need_commit_before_edits)
