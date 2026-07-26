# 🔐 Skill Auditor — Step-by-Step Guide

A step-by-step guide for running a security audit on your installed agent skills.

---

## 📋 What is a skill audit?

A skill audit checks whether the skills installed in your environment contain:
- ❌ Malicious or dangerous code
- ⚠️ Questionable security patterns
- ✅ Expected, well-documented behavior

The audit is **local, offline, and never sends your data to a third party.**

---

## 🚀 STEP BY STEP — AUDITING YOUR SKILLS

### **STEP 1: Open a terminal**

In VS Code: `Ctrl + \`` (backtick), or via the menu: `View` → `Terminal`. Any other terminal works too.

---

### **STEP 2: Navigate to `skill-manager/scripts`**

```bash
cd /path/to/skill-manager/scripts
```

---

### **STEP 3: Run the collector script**

```bash
bash audit_skills.sh
```

Or with explicit permissions:
```bash
chmod +x audit_skills.sh && ./audit_skills.sh
```

The script will:
- ✅ Detect your installed skills (checks `~/.local/share/gh/extensions/skills/`, `~/.claude/skills/`, `~/.cursor/skills/`, and `~/.codex/skills/`, in that order)
- ✅ Bundle each skill's `SKILL.md` and scripts into one file
- ✅ Generate a `skills_audit_package_YYYYMMDD_HHMMSS.md` (plus a `.json` manifest)
- ✅ Print the full path of the generated file

**Example output:**
```
✓ Found 30 skills
✓ Generated: /home/user/.claude/skills/skill-manager/skills_audit_package_20260601_095615.md
```

---

### **STEP 4: Open the generated file in your editor**

```bash
code skills_audit_package_*.md
```

Or manually: `Ctrl+O` → browse to `skills_audit_package_*.md`, or drag the file into your editor.

---

### **STEP 5: Select ALL of the file's content**

```
Ctrl + A
```

This selects everything so you can copy it.

---

### **STEP 6: Open your agent (e.g. Claude Code)**

Open whichever coding agent you want to run the audit with — a side panel in your editor, or a separate terminal/chat window.

---

### **STEP 7: Paste the file and send the audit prompt**

1. **Paste the content** (Ctrl+V) into your agent's chat
2. **Paste the exact audit prompt.** The script prints it to the terminal after generating the file, and it also lives at [`audit-prompt.md`](./audit-prompt.md) in this same folder — copy it verbatim.
3. **Press Enter** to send

---

### **STEP 8: Wait for the analysis**

Your agent will analyze every skill in the package. Estimated time:
- Up to 10 skills: ~1–2 minutes
- 10–20 skills: ~2–3 minutes
- 20–30 skills: ~3–5 minutes

---

### **STEP 9: Review the report**

Your agent will return a verdict for each skill:

| Symbol | Meaning | Action |
|---|---|---|
| ✅ SAFE | No issues detected | Install with confidence |
| ⚠️ REVIEW | Needs manual review | Ask your agent for details |
| ❌ BLOCK | Not safe | DO NOT install |

**Example report entry:**
```markdown
### scikit-learn
- **Verdict:** ✅ SAFE
- **Severity:** 🟢 LOW
- **Findings:** No issues detected
- **Risk:** None
```

---

### **STEP 10: Make a decision**

Based on the verdict:

#### ✅ Everything SAFE?
Install/keep the skills with confidence.

#### ⚠️ Something REVIEW?
Ask your agent:
```
"What's the exact risk of [SKILL_NAME]? Can I use it safely?"
```

#### ❌ Something BLOCK?
Do NOT install that skill. Look for an alternative or wait for a security fix.

---

## 📖 Interpreting results

### 🟢 **Severity: LOW**
- Expected behavior, no security risk. **Action:** proceed with installation.

### 🟡 **Severity: MEDIUM**
- A documented pattern (e.g. cloud API access), no known vulnerabilities. **Action:** review the documentation; proceed if appropriate.

### 🟠 **Severity: HIGH**
- Potentially dangerous behavior, or a pattern that needs manual review. **Action:** ask your agent about the specific context.

### 🔴 **Severity: CRITICAL**
- Clearly malicious code, or a forbidden function (eval, exec, etc.). **Action:** ❌ DO NOT INSTALL.

---

## ❓ FAQ

**Q: How long does it take to audit my skills?**
A: Depends on the count — up to 10 skills: ~2 minutes; up to 30 skills: ~5 minutes.

**Q: Is my skill data sent anywhere?**
A: No. The audit runs **locally** on your machine. The file is analyzed only by the agent you paste it into.

**Q: Do I need to be online?**
A: Yes, to use a hosted agent. But the file itself is processed locally — nothing is uploaded to the audit script.

**Q: Can I audit only some skills?**
A: Yes. Manually edit the generated `skills_audit_package_*.md` and remove the skills you don't want audited before sending it to your agent.

**Q: What do I do if a skill gets a ⚠️ REVIEW verdict?**
A: Ask your agent: *"What's the exact risk of [SKILL_NAME]? Why does it need review? Is it safe if I avoid feature X?"*

**Q: Can I reuse a previous audit?**
A: Yes, if the skill versions haven't changed. Regenerating monthly or after updates is still recommended.

---

## 🔧 Troubleshooting

**`command not found: bash`**
```bash
# If you're on zsh or another shell:
zsh audit_skills.sh
```

**`Permission denied`**
```bash
chmod +x audit_skills.sh
./audit_skills.sh
```

**No skills detected**
```bash
# Check that your skills actually live in one of:
ls ~/.claude/skills/
ls ~/.cursor/skills/
ls ~/.codex/skills/
ls ~/.local/share/gh/extensions/skills/
```

**Your agent can't process the file**
The file might be too large. Split it into parts (e.g. skills A–K, then L–Z) and audit each separately.

---

## 📋 Audit checklist

Before installing any new skill, run through this checklist:

- [ ] `skills_audit_package_*.md` generated successfully
- [ ] File opened and content copied into your agent
- [ ] Audit prompt sent
- [ ] Full report received
- [ ] Every verdict reviewed
- [ ] No skill with a ❌ BLOCK verdict
- [ ] Skills with ⚠️ REVIEW were followed up on
- [ ] Final decision made

**Done! Your audit is complete. 🔒**
