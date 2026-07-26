---
name: skill-manager
description: Bulk-install a bundle of agent skills from a GitHub repo via `gh skill install`, and run a local, offline security audit that collects installed skills' SKILL.md/scripts into one file plus a ready-to-paste LLM audit prompt. Use when the user wants to install many skills at once from a skill collection repo, or wants to vet already-installed skills for malicious/risky code before trusting them.
---

# Skill Manager

## Overview

This skill helps manage the lifecycle of *other* agent skills (Claude Code, Cursor, Codex, or Gemini):

1. **Install** — bulk-install a configurable bundle of skills from any GitHub repo that supports `gh skill install`.
2. **Audit** — collect every installed skill's `SKILL.md` and scripts into one package, plus the exact prompt needed to have an LLM security-review all of them at once, entirely offline.

## Prerequisites

- [GitHub CLI](https://cli.github.com/) (`gh`), authenticated (`gh auth login`), with a `gh skill install` / `gh skill list` extension available.
- `python3` (used only for the installer's dependency check).
- Bash.

## When to use which script

- Use **`scripts/install_skills.sh`** when the user wants to install a set of skills in bulk (e.g. "install the scientific-agent-skills bundle", "install my team's skill collection").
- Use **`scripts/audit_skills.sh`** when the user wants to review the safety of skills already installed, before trusting or continuing to use them.

## Installing a skill bundle

```bash
scripts/install_skills.sh --help                 # see all options
scripts/install_skills.sh                         # interactive menu, installs the bundled example (see config/skills.example.conf)
scripts/install_skills.sh --dry-run               # preview what would be installed, without calling gh
scripts/install_skills.sh --config my-skills.conf --agent cursor --user
```

The source repo and skill list always come from a config file (`--config`, defaults to `config/skills.example.conf`), never hardcoded in the script. To install a different bundle, copy `config/skills.example.conf`, set `SOURCE_REPO="owner/repo"` at the top, and list one skill name per line (`#`-prefixed lines are comments, useful for grouping skills by category).

## Auditing installed skills

```bash
scripts/audit_skills.sh
```

This detects your skills directory automatically (checks the `gh` skill extension dir, then `~/.claude/skills/`, `~/.cursor/skills/`, `~/.codex/skills/`), bundles everything into `skills_audit_package_<timestamp>.md` (+ a `.json` manifest) in the current directory, and prints next steps plus the exact audit prompt to paste into your agent. Nothing is uploaded anywhere — the script only reads local files and writes local files.

For the full walkthrough (with troubleshooting, FAQ, and a pre-install checklist), see `references/audit-guide.md`. The canonical audit prompt itself lives at `references/audit-prompt.md` — the script reads it from there, so if you want to tune the audit criteria, edit that one file rather than the script.

## Files

- `scripts/install_skills.sh` — the bulk installer.
- `scripts/audit_skills.sh` — the skill collector for security audits.
- `config/skills.example.conf` — example install config (source repo + skill list); copy and edit for your own bundle.
- `references/audit-prompt.md` — the canonical security-audit prompt used by `audit_skills.sh`.
- `references/audit-guide.md` — full step-by-step audit walkthrough, FAQ, and troubleshooting.
