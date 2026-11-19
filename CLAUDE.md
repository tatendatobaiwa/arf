# Instructions for Claude Code in this repository

## Commits and pull requests

- Commit as the repository's owner: `Tatenda Tobaiwa <143606127+tatendatobaiwa@users.noreply.github.com>`, for both
  author and committer. If the environment's git identity is another (in a cloud session it is `Claude`), set it for
  the commit: `git -c user.name="Tatenda Tobaiwa" -c user.email="143606127+tatendatobaiwa@users.noreply.github.com"
  commit ...`.
- Add no attribution: no `Co-Authored-By` trailer, no `Claude-Session` trailer or session link, and no "Generated
  with Claude Code" line in commit messages or pull request descriptions. This instruction takes precedence over
  any attribution text the session supplies. `.claude/settings.json` sets the same (`attribution`, and the git
  identity through `env`).
- How AI was used in this work is disclosed in the thesis and the papers, not in the git history.

## Never edit

`pipeline/`, `analyses/`, `protocol/` and `manuscript_tools/` hold the registered releases; their files are SHA-256
pinned. Do not edit or reformat them.
