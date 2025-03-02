import os
from aider.io import ConfirmGroup
from aider import prompts

class MentionHandler:
    def __init__(self, coder):
        self.coder = coder
        self.io = coder.io
        self.ignore_mentions = coder.ignore_mentions

    def check_for_file_mentions(self, content):
        mentioned_rel_fnames = self.get_file_mentions(content)
        new_mentions = mentioned_rel_fnames - self.ignore_mentions
        return self._process_new_mentions(new_mentions) if new_mentions else None

    def get_file_mentions(self, content):
        words = {word.rstrip(",.!;:?").strip('"\'`') for word in content.split()}
        return self._find_matching_files(words)

    def _find_matching_files(self, words):
        existing_basenames = self._get_existing_basenames()
        return {
            rel_fname
            for rel_fname in self.coder.get_addable_relative_files()
            if self._is_mentioned(rel_fname, words, existing_basenames)
        }

    def _get_existing_basenames(self):
        return {
            os.path.basename(f)
            for f in self.coder.get_inchat_relative_files()
        } | {
            os.path.basename(self.coder.get_rel_fname(f))
            for f in self.coder.abs_read_only_fnames
        }

    def _is_mentioned(self, rel_fname, words, existing_basenames):
        normalized = rel_fname.replace("\\", "/")
        if normalized in {w.replace("\\", "/") for w in words}:
            return True
        basename = os.path.basename(rel_fname)
        return basename not in existing_basenames and basename in words

    def _process_new_mentions(self, new_mentions):
        added_files = []
        group = ConfirmGroup(new_mentions)
        for rel_fname in sorted(new_mentions):
            if self.io.confirm_ask("Add file to the chat?", subject=rel_fname, group=group, allow_never=True):
                self.coder.add_rel_fname(rel_fname)
                added_files.append(rel_fname)
            else:
                self.ignore_mentions.add(rel_fname)
        return prompts.added_files.format(fnames=", ".join(added_files)) if added_files else None
