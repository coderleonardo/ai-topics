#!/usr/bin/env bash

# ═══════════════════════════════════════════════════════════════════════════════
# SKILL COLLECTOR FOR AGENT SKILL SECURITY AUDITS
#
# Scans your installed agent skills (Claude Code, Cursor, Codex, or the `gh`
# skill extension), bundles each skill's SKILL.md and scripts into one
# Markdown package plus a machine-readable JSON manifest, and prints
# instructions (and the exact prompt) for having an LLM audit them for you.
#
# This script is 100% local and offline: it only reads files already on your
# disk and writes new files next to itself. Nothing is uploaded anywhere.
# ═══════════════════════════════════════════════════════════════════════════════

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="$SCRIPT_DIR/../references/audit-prompt.md"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
ORANGE='\033[0;33m'
NC='\033[0m'

detect_skills_dir() {
    if [ -d "$HOME/.local/share/gh/extensions/skills/" ]; then
        echo "$HOME/.local/share/gh/extensions/skills/"
    elif [ -d "$HOME/.claude/skills/" ]; then
        echo "$HOME/.claude/skills/"
    elif [ -d "$HOME/.cursor/skills/" ]; then
        echo "$HOME/.cursor/skills/"
    elif [ -d "$HOME/.codex/skills/" ]; then
        echo "$HOME/.codex/skills/"
    else
        echo ""
    fi
}

print_header()  { echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }
print_info()    { echo -e "${YELLOW}ℹ️  $1${NC}"; }
print_warning() { echo -e "${ORANGE}⚠️  $1${NC}"; }

if [ ! -f "$PROMPT_FILE" ]; then
    print_error "Audit prompt not found at: $PROMPT_FILE"
    exit 1
fi
AUDIT_PROMPT="$(cat "$PROMPT_FILE")"

SKILLS_DIR=$(detect_skills_dir)

if [ -z "$SKILLS_DIR" ]; then
    print_error "No skills directory found!"
    echo ""
    echo "Looked in:"
    echo "  ~/.local/share/gh/extensions/skills/"
    echo "  ~/.claude/skills/"
    echo "  ~/.cursor/skills/"
    echo "  ~/.codex/skills/"
    exit 1
fi

OUTPUT_FILE="skills_audit_package_$(date +%Y%m%d_%H%M%S).md"

print_header "🔐 SKILL COLLECTOR FOR AGENT SKILL AUDITS"
echo ""
print_info "Detected directory: ${GREEN}${SKILLS_DIR}${NC}"
print_info "Output file: ${GREEN}${OUTPUT_FILE}${NC}"
echo ""

SKILL_COUNT=$(find "$SKILLS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
print_info "Found ${GREEN}${SKILL_COUNT}${NC} skills"
echo ""

# ── Part 1: header with expanded variables ─────────────────────────────────────
cat > "$OUTPUT_FILE" << HEADER
# 🔐 COMPREHENSIVE SKILLS SECURITY AUDIT PACKAGE

**Total Skills:** ${SKILL_COUNT}
**Generated:** $(date)

---

## 📋 TABLE OF CONTENTS

### Skills included:

HEADER

while IFS= read -r skill_path; do
    skill_name=$(basename "$skill_path")
    echo "- [${skill_name}](#skill-${skill_name})" >> "$OUTPUT_FILE"
done < <(find "$SKILLS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

# ── Part 2: audit instructions, sourced from references/audit-prompt.md ────────
{
    echo ""
    echo "---"
    echo ""
    echo "## 🔒 SECURITY AUDIT INSTRUCTIONS"
    echo ""
    echo "### How to use this file:"
    echo ""
    echo "1. **Open your agent (Claude Code, or any assistant you trust with this review)**"
    echo "2. **Open this file in your editor**"
    echo "3. **Select ALL content (Ctrl+A / Cmd+A)**"
    echo "4. **Send this prompt to your agent:**"
    echo ""
    echo '```'
    echo "$AUDIT_PROMPT"
    echo '```'
    echo ""
    echo "---"
    echo ""
} >> "$OUTPUT_FILE"

# ── Part 3: collect all skills ──────────────────────────────────────────────────
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

CURRENT=0
find "$SKILLS_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r skill_path; do
    skill_name=$(basename "$skill_path")
    CURRENT=$((CURRENT + 1))

    echo -ne "\r[$CURRENT/$SKILL_COUNT] $skill_name                    "

    {
        echo "## Skill: $skill_name"
        echo ""
        echo "**Path:** \`$skill_path\`"
        echo ""

        if [ -f "$skill_path/SKILL.md" ]; then
            echo "### SKILL.md"
            echo ""
            echo '```markdown'
            cat "$skill_path/SKILL.md"
            echo '```'
            echo ""
        fi

        if [ -d "$skill_path/scripts" ]; then
            for py_file in "$skill_path"/scripts/*.py; do
                [ -f "$py_file" ] || continue
                echo "### Python Script: $(basename "$py_file")"
                echo ""
                echo '```python'
                cat "$py_file"
                echo '```'
                echo ""
            done
        fi

        for py_file in "$skill_path"/*.py; do
            [ -f "$py_file" ] || continue
            echo "### Python Script: $(basename "$py_file")"
            echo ""
            echo '```python'
            cat "$py_file"
            echo '```'
            echo ""
        done

        for sh_file in "$skill_path"/*.sh "$skill_path"/scripts/*.sh; do
            [ -f "$sh_file" ] || continue
            echo "### Script: $(basename "$sh_file")"
            echo ""
            echo '```bash'
            cat "$sh_file"
            echo '```'
            echo ""
        done

        echo "---"
        echo ""

    } >> "$OUTPUT_FILE"
done

echo -ne "\r                                                  \r"

print_success "Audit package created: ${GREEN}${OUTPUT_FILE}${NC}"
echo ""
print_header "📋 NEXT STEPS"
echo ""
echo "1. Open your agent (e.g. Claude Code)"
echo ""
echo "2. Open the file:"
echo "   ${GREEN}${OUTPUT_FILE}${NC}"
echo ""
echo "3. Select ALL content: Ctrl+A"
echo ""
echo "4. Paste this EXACT prompt into your agent:"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "$AUDIT_PROMPT"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "5. Wait for the analysis (it may take a few minutes)"
echo ""
echo "6. Review the generated security report"
echo ""
print_warning "IMPORTANT: Do not install any skill with a BLOCK verdict"
print_warning "Manually review skills with a REVIEW verdict"
echo ""
print_success "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
print_info "Total skills: ${GREEN}${SKILL_COUNT}${NC}"
echo ""

# ── Part 4: JSON output ─────────────────────────────────────────────────────────
JSON_FILE="skills_audit_package_$(date +%Y%m%d_%H%M%S).json"

mapfile -t SKILL_PATHS < <(find "$SKILLS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
TOTAL=${#SKILL_PATHS[@]}

{
    echo "{"
    echo "  \"audit_info\": {"
    echo "    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "    \"total_skills\": $SKILL_COUNT,"
    echo "    \"directory\": \"$SKILLS_DIR\""
    echo "  },"
    echo "  \"skills\": ["

    for i in "${!SKILL_PATHS[@]}"; do
        skill_path="${SKILL_PATHS[$i]}"
        skill_name=$(basename "$skill_path")

        echo "    {"
        echo "      \"name\": \"$skill_name\","
        echo "      \"path\": \"$skill_path\","
        echo "      \"files\": ["

        mapfile -t FILES < <(find "$skill_path" -type f \( -name "*.py" -o -name "*.sh" -o -name "SKILL.md" \))
        FILE_TOTAL=${#FILES[@]}
        for j in "${!FILES[@]}"; do
            fname=$(basename "${FILES[$j]}")
            if [ $((j + 1)) -lt "$FILE_TOTAL" ]; then
                echo "        \"${fname}\","
            else
                echo "        \"${fname}\""
            fi
        done

        echo "      ]"
        if [ $((i + 1)) -lt "$TOTAL" ]; then
            echo "    },"
        else
            echo "    }"
        fi
    done

    echo "  ]"
    echo "}"
} > "$JSON_FILE"

print_info "JSON file also created: ${GREEN}${JSON_FILE}${NC}"
echo ""
print_header "✅ FILES READY"
echo ""
ls -lh skills_audit_package_*.* 2>/dev/null || echo "Files created in the current directory"
echo ""
