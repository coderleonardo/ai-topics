#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════════
# SKILL COLLECTOR FOR CLAUDE CODE SECURITY AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

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

SKILLS_DIR=$(detect_skills_dir)

if [ -z "$SKILLS_DIR" ]; then
    print_error "Nenhum diretório de skills encontrado!"
    echo ""
    echo "Procurei em:"
    echo "  ~/.local/share/gh/extensions/skills/"
    echo "  ~/.claude/skills/"
    echo "  ~/.cursor/skills/"
    echo "  ~/.codex/skills/"
    exit 1
fi

OUTPUT_FILE="skills_audit_package_$(date +%Y%m%d_%H%M%S).md"

print_header "🔐 SKILL COLLECTOR FOR CLAUDE CODE AUDIT"
echo ""
print_info "Diretório detectado: ${GREEN}${SKILLS_DIR}${NC}"
print_info "Arquivo de saída: ${GREEN}${OUTPUT_FILE}${NC}"
echo ""

# Fix bug #5: tr -d ' ' strips leading whitespace from wc -l
SKILL_COUNT=$(find "$SKILLS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
print_info "Encontradas ${GREEN}${SKILL_COUNT}${NC} skills"
echo ""

# ── Part 1: header with expanded variables (no backticks here, safe to expand) ─
cat > "$OUTPUT_FILE" << HEADER
# 🔐 COMPREHENSIVE SKILLS SECURITY AUDIT PACKAGE

**Total Skills:** ${SKILL_COUNT}
**Generated:** $(date)

---

## 📋 TABLE OF CONTENTS

### Skills Included:

HEADER

# Fix bug #2: write TOC directly instead of multiline sed replacement
while IFS= read -r skill_path; do
    skill_name=$(basename "$skill_path")
    echo "- [${skill_name}](#skill-${skill_name})" >> "$OUTPUT_FILE"
done < <(find "$SKILLS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

# ── Part 2: static audit instructions (single-quoted heredoc — no expansion) ──
cat >> "$OUTPUT_FILE" << 'INSTRUCTIONS'

---

## 🔒 SECURITY AUDIT INSTRUCTIONS FOR CLAUDE CODE

### How to use this file:

1. **Open Claude Code**
2. **Open this file in your editor**
3. **Select ALL content (Ctrl+A)**
4. **Send this prompt to Claude:**

```
You are a professional security auditor. I need you to perform
a COMPREHENSIVE security audit of ALL the skills in this document.

For EACH skill, you must:

1. **Analyze ALL code** for malicious patterns:
   - eval(), exec(), __import__(), compile()
   - pickle.loads(), marshal.loads()
   - subprocess.call(), os.system()
   - socket.*, requests.*, urllib.*
   - base64.b64decode()

2. **Check for obfuscation:**
   - String encoding/decoding chains
   - Dynamic code construction
   - Escape sequences
   - Unusual variable names

3. **Verify capabilities:**
   - Declared tools vs actual code used
   - Unexpected permission requirements
   - File system access patterns

4. **Assess risk:**
   - Data exfiltration attempts
   - Privilege escalation
   - System modifications
   - Supply chain attacks

5. **Provide verdict:**
   - SAFE - No issues detected
   - REVIEW - Needs manual verification
   - BLOCK - Do not install

Format your response as:

### SKILL: [name]
- **Status:** [SAFE/REVIEW/BLOCK]
- **Severity:** [LOW/MEDIUM/HIGH/CRITICAL]
- **Findings:** [List issues or "No issues"]
- **Recommendation:** [Action to take]
- **Details:** [Why this verdict]

---

After reviewing all skills, provide:

## SUMMARY
- Total skills: X
- Safe: X
- Review needed: X
- Block: X
- Risk level: LOW/MEDIUM/HIGH
```

---

INSTRUCTIONS

# ── Part 3: collect all skills ────────────────────────────────────────────────
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

CURRENT=0
find "$SKILLS_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r skill_path; do
    skill_name=$(basename "$skill_path")
    CURRENT=$((CURRENT + 1))  # avoids ((++)) returning false when value was 0

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

        # Fix bug #3: brace expansion must be outside quotes
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

print_success "Arquivo de auditoria criado: ${GREEN}${OUTPUT_FILE}${NC}"
echo ""
print_header "📋 PRÓXIMOS PASSOS"
echo ""
echo "1. Abra Claude Code"
echo ""
echo "2. Abra o arquivo:"
echo "   ${GREEN}${OUTPUT_FILE}${NC}"
echo ""
echo "3. Selecione TODO o conteúdo: Ctrl+A"
echo ""
echo "4. Cole este EXATO prompt no Claude:"
echo ""
cat << 'PROMPT'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are a professional security auditor specializing in code review.
Perform a COMPREHENSIVE security audit of ALL skills in this document.

For EACH skill, analyze:

1. **CRITICAL FUNCTIONS** (auto-block if found without legitimate context):
   - eval(), exec(), __import__(), compile()
   - pickle.loads(), pickle.load()
   - marshal.loads()
   - subprocess.call(), os.system() with shell=True
   - ctypes.CDLL(), ctypes.WinDLL()

2. **HIGH-RISK PATTERNS** (review context required):
   - subprocess.run() with shell=True
   - socket.*, requests.*, urllib with untrusted URLs
   - base64/hex/ROT13 decoding chains
   - Hardcoded API keys or credentials

3. **OBFUSCATION DETECTION**:
   - String encoding chains
   - Dynamic code construction
   - Unusual variable naming
   - Hidden control flow

4. **CAPABILITY MISMATCH**:
   - Declared tools vs actual usage
   - Unexpected privilege requirements
   - Undeclared network access

5. **ATTACK PATTERNS**:
   - Data exfiltration (external URLs + file writes)
   - Privilege escalation
   - System modification
   - Supply chain attacks

VERDICT RULES:
- CRITICAL found → BLOCK
- HIGH + unclear use → REVIEW
- HIGH + clear use → ALLOW
- MEDIUM → ALLOW
- LOW or none → SAFE

RESPONSE FORMAT (for each skill):

### [SKILL_NAME]
- **Verdict:** SAFE | REVIEW | BLOCK
- **Severity:** LOW | MEDIUM | HIGH | CRITICAL
- **Author:** [from SKILL.md]
- **Findings:** [issues or "No issues detected"]
- **Risk:** [what could go wrong]

FINAL SUMMARY:
- Skills audited: X
- Safe: X | Review: X | Block: X
- Overall risk: LOW/MEDIUM/HIGH

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT

echo ""
echo "5. Aguarde a análise (pode levar alguns minutos)"
echo ""
echo "6. Revise o relatório de segurança gerado"
echo ""
print_warning "IMPORTANTE: Não instale nenhuma skill com veredicto BLOCK"
print_warning "Revise manualmente skills com veredicto REVIEW"
echo ""
print_success "Tamanho do arquivo: $(du -h "$OUTPUT_FILE" | cut -f1)"
print_info "Total de skills: ${GREEN}${SKILL_COUNT}${NC}"
echo ""

# ── Part 4: JSON output ───────────────────────────────────────────────────────
# Fix bug #4: collect paths into array first so we can track last element
# and avoid trailing comma in JSON
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

print_info "Arquivo JSON também criado: ${GREEN}${JSON_FILE}${NC}"
echo ""
print_header "✅ ARQUIVOS PRONTOS"
echo ""
ls -lh skills_audit_package_*.* 2>/dev/null || echo "Arquivos criados no diretório atual"
echo ""
