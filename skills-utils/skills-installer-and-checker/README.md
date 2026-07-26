# Skill Manager

A small toolkit for managing *other* agent skills (Claude Code, Cursor, Codex, Gemini): bulk-install a bundle of skills from a GitHub repo, and run a local, offline security audit on whatever's already installed before you trust it.

This repository provides one skill package:

- `skill-manager/`
  - `SKILL.md`: overview and usage rules
  - `scripts/install_skills.sh`: configurable bulk installer
  - `scripts/audit_skills.sh`: local security-audit collector
  - `config/skills.example.conf`: example install config (source repo + skill list)
  - `references/`: audit prompt + full step-by-step audit guide

## Installation

Assume you are in the repository root.

### 1) Claude Code

Global:

```bash
mkdir -p "$HOME/.claude/skills"
cp -R skill-manager "$HOME/.claude/skills/"
```

Project-level:

```bash
mkdir -p .claude/skills
cp -R skill-manager .claude/skills/
```

In prompts, explicitly request this skill, for example: `Please use the skill-manager skill to audit my installed skills`.

### 2) Codex

```bash
mkdir -p "$CODEX_HOME/skills"
cp -R skill-manager "$CODEX_HOME/skills/"
```

### 3) Cursor

```bash
mkdir -p "$HOME/.cursor/skills"
cp -R skill-manager "$HOME/.cursor/skills/"
```

### 4) Gemini

```bash
mkdir -p "$HOME/.gemini/skills"
cp -R skill-manager "$HOME/.gemini/skills/"
```

You can also just run the scripts directly without installing them as a skill — see below.

## Usage

### Install a bundle of skills

```bash
cd skill-manager/scripts
./install_skills.sh --help
```

By default, running it with no flags opens an interactive menu and installs the bundled example (a scientific-computing skill collection). Common flags:

```bash
./install_skills.sh --dry-run                                        # preview only, no network calls
./install_skills.sh --config my-skills.conf --agent cursor --user    # install your own bundle for Cursor, user-wide
./install_skills.sh --repo my-org/my-skills --project                # point at a different repo, project-scoped
```

### Configuring your own skill bundle

The installer never hardcodes a skill list — it reads one from a config file (`--config`, defaults to `config/skills.example.conf`). To use your own bundle:

1. Copy `config/skills.example.conf` to a new file.
2. Set `SOURCE_REPO="owner/repo"` at the top (or pass `--repo` at runtime instead).
3. List one skill name per line. Lines starting with `#` are ignored — use them as category labels if you like.
4. Run `install_skills.sh --config path/to/your-file.conf`.

### Audit installed skills for security

```bash
cd skill-manager/scripts
./audit_skills.sh
```

This bundles every installed skill's `SKILL.md` and scripts into a single Markdown file (plus a JSON manifest) in your current directory, and prints the exact prompt to paste into your agent for a security review. The audit is entirely local and offline — nothing is uploaded anywhere except to whichever agent you choose to paste the file into. See `skill-manager/references/audit-guide.md` for the full step-by-step walkthrough, FAQ, and troubleshooting.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).
