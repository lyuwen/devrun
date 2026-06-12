# Workflow Orchestration Toolkit Guidelines

It is meant to be a toolkit for human and AI agents to use to run complicated but deterministic workflows.
The workflow orchestration should use a stateful heartbeat system, which each pulse, it checks the state of the jobs and promote them to the next stage when the current stage is completed.

Read @AGENT.md in project root to learn the currect software archetexture and implementation status. Treat thisnfile as your long-term memory and update the project archetecture and design choices proactively.

The "examples" directory contains the examples for the workflows, read the sub-level CLAUDE.local.md for details. And treat this directory as the playground for testing. But remember not to edit the existing files.

The "docs" directory is used for project documentation that includes the documentation of the toolkit as well as the implementation plans and review reports. Keep it organized so that we have a clear history and vision ahead.

The toolkit is designed to be used by both human and AI agents to formalize LLM related data synthesis, training and evaluation workflows into standardized and parametrizable modules, so that it is both flexible and contained. AI agent is expected to access the toolkit with custom agents and skills. Therefore, in the design process, simulate realistic usage environments and proactively ask user for use cases, environment information/status and expected behaviors to align with user preference.

**IMPORTANT**
1. When impelementing a new feature, plan extensively with subagents and discuss the plan with a reviewer agent. But always keep user in the loop for important decisions or when there is ambiguity.
2. Write comprehensive unit tests, review thoroughly the testing plan to make sure it covers as much aspects as possible.
3. Delegate implementation, review and testing of submodules to agent teams if the workload is significant.
4. The project is already designes with multi-level configuration loading and override to avoid leaking sensitive data, always keep this in mind when designing new features.

# Rules
1. Do not edit production data in-place, always creat a copy and edit the copy.
2. Most of workflow jobs are meant to run on a remote machine, do not assume to run locally. These jobs usually have a default remote point to run on, but do not hard-code the prameters.
3. For testing, create dry-run mode and make it transparent snd verbose.
4. Do not forget to activate remote environments if necessany.
5. **NEVER** disclose sentive user credentials and access keys.
6. Use branches and worktrees to develop mew features, when a feature is production ready, create a pull request with a detailed QA review report.
7. Do not commit information that might disclose user credentials, environments and directory structures into the repository.

# Committing & Repository Hygiene
1. Commit at the end of each development stage (feature complete, refactor done, tests passing) — don't accumulate large uncommitted diffs across unrelated changes.
2. Stage files explicitly by name. Never use `git add -A` / `git add .` — those sweep up untracked junk.
3. Only commit files that are part of the change. Do NOT commit:
   - Sensitive data (credentials, `.env`, tokens, keys, host-specific paths).
   - Random markdown files, scratch notes, brainstorming drafts, or session transcripts.
   - Intermediate testing scripts or one-off experiments — keep those in `examples/` (playground) or delete them.
   - `.pyc`, `__pycache__`, IDE/editor configs, OS metadata (`.DS_Store`, etc.).
4. Keep the project root clean: source under `devrun/`, tests under `tests/`, docs under `docs/`, examples under `examples/`. Never drop loose files at the top level.
5. If a file doesn't fit any existing directory, ask before creating a new top-level one.
