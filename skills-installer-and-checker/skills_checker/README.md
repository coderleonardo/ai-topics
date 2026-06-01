# 🔐 Auditor de Skills — Claude Code

Um guia passo a passo para realizar auditorias de segurança nas suas skills instaladas do Claude Code.

---

## 📋 O que é uma Auditoria de Skills?

Uma auditoria de skills verifica se as skills instaladas no seu ambiente contêm:
- ❌ Código malicioso ou perigoso
- ⚠️ Padrões de segurança questionáveis
- ✅ Comportamentos esperados e documentados

A auditoria é **local, offline e não envia dados para terceiros**.

---

## 🚀 PASSO A PASSO - VERIFICAÇÃO DE SKILLS

### **PASSO 1: Abra o Terminal do VSCode**

```
Ctrl + ` (acento grave)
```

Ou via menu: `View` → `Terminal`

---

### **PASSO 2: Navegue até a pasta skills_checker**

```bash
cd /caminho/para/skills_checker
```

Ou, se já estiver no repositório raiz:
```bash
cd skills_checker
```

---

### **PASSO 3: Execute o script de coleta**

```bash
bash collect_skills_for_audit.sh
```

Ou com permissões explícitas:
```bash
chmod +x collect_skills_for_audit.sh && ./collect_skills_for_audit.sh
```

O script vai:
- ✅ Detectar suas skills instaladas no `~/.claude/skills/`
- ✅ Consolidar o SKILL.md de cada uma
- ✅ Gerar um arquivo `skills_audit_package_YYYYMMDD_HHMMSS.md`
- ✅ Exibir o caminho completo do arquivo gerado

**Exemplo de saída:**
```
✓ Coletadas 30 skills
✓ Gerado: /home/user/.claude/skills_checker/skills_audit_package_20260601_095615.md
```

---

### **PASSO 4: Abra o arquivo gerado no VSCode**

```bash
code skills_audit_package_*.md
```

Ou manualmente:
- `Ctrl+O` → navegue até `skills_audit_package_*.md`
- Ou clique e arraste o arquivo para o VSCode

---

### **PASSO 5: Selecione TODO o conteúdo do arquivo**

```
Ctrl + A
```

Isso seleciona todo o conteúdo para copiar.

---

### **PASSO 6: Abra Claude Code (painel direito do VSCode)**

- Clique no ícone do Claude no lado direito da barra lateral
- Ou pressione `Ctrl+Shift+I`

Claude Code abrirá em um painel lateral.

---

### **PASSO 7: Cole o arquivo e envie o prompt de auditoria**

1. **Cole o conteúdo** (Ctrl+V) no campo de chat do Claude Code
2. **Digite ou cole este prompt exato:**

```
You are a professional security auditor specializing in code review. 
Perform a COMPREHENSIVE security audit of ALL skills in this document.

For EACH skill, analyze:

1. **CRITICAL FUNCTIONS** (auto-block if found):
   - eval(), exec(), __import__(), compile()
   - pickle.loads(), marshal.loads()
   - subprocess.call(), os.system() with shell=True
   - ctypes.CDLL(), ctypes.WinDLL()

2. **HIGH-RISK PATTERNS**:
   - socket.*, requests.*, urllib with untrusted URLs
   - base64/hex/ROT13 decoding chains
   - Hardcoded credentials

3. **OBFUSCATION DETECTION**:
   - String encoding chains
   - Dynamic code construction
   - Unusual variable naming

4. **CAPABILITY MISMATCH**:
   - Declared tools vs actual code used
   - Unexpected permissions
   - Undeclared network access

5. **ATTACK PATTERNS**:
   - Data exfiltration
   - Privilege escalation
   - System modifications

VERDICT RULES:
- 🔴 CRITICAL found → ❌ BLOCK
- 🟠 HIGH + unclear → ⚠️ REVIEW
- 🟠 HIGH + clear use → ✅ ALLOW
- 🟡 MEDIUM → ✅ ALLOW
- 🟢 LOW or none → ✅ SAFE

For EACH skill provide:

### [SKILL_NAME]
- **Verdict:** ✅ SAFE | ⚠️ REVIEW | ❌ BLOCK
- **Severity:** 🟢 LOW | 🟡 MEDIUM | 🟠 HIGH | 🔴 CRITICAL
- **Findings:** [issues or "No issues detected"]
- **Risk:** [assessment]

Then provide FINAL SUMMARY:
- Skills audited: X
- Safe: X | Review: X | Block: X
- Overall risk level: LOW/MEDIUM/HIGH
```

3. **Pressione Enter** para enviar

---

### **PASSO 8: Aguarde a análise**

Claude analisará todas as skills. Tempo estimado:
- Até 10 skills: ~1–2 minutos
- 10–20 skills: ~2–3 minutos
- 20–30 skills: ~3–5 minutos

Um indicador de progresso aparecerá no painel.

---

### **PASSO 9: Revise o relatório**

Claude fornecerá um relatório com vereditos para cada skill:

| Símbolo | Significado | Ação |
|---------|-------------|------|
| ✅ SAFE | Sem problemas detectados | Instalar com confiança |
| ⚠️ REVIEW | Precisa revisão manual | Perguntar detalhes ao Claude |
| ❌ BLOCK | Não é seguro | NÃO instalar |

**Exemplo de entrada no relatório:**
```markdown
### scikit-learn
- **Verdict:** ✅ SAFE
- **Severity:** 🟢 LOW
- **Findings:** No issues detected
- **Risk:** None
```

---

### **PASSO 10: Tome uma decisão**

Com base no veredicto:

#### ✅ Tudo SAFE?
```bash
# Instale todas as skills com confiança
# Exemplo:
cd ~/.claude/skills
git pull  # ou clone se não tiver ainda
```

#### ⚠️ Tem alguma com REVIEW?
```
Pergunte ao Claude:
"Qual é o risco exato de [SKILL_NAME]? Posso usar?"
```

#### ❌ Tem alguma com BLOCK?
```
NÃO instale essa skill.
Procure uma alternativa ou espere uma atualização de segurança.
```

---

## 📖 Interpretar Resultados

### 🟢 **Severity: LOW**
- Comportamento esperado
- Sem risco de segurança
- **Ação:** Prossiga com instalação

### 🟡 **Severity: MEDIUM**
- Padrão documentado (ex: acesso a API cloud)
- Sem vulnerabilidades conhecidas
- **Ação:** Revisar documentação; prosseguir se apropriado

### 🟠 **Severity: HIGH**
- Comportamento potencialmente perigoso
- Ou padrão que requer revisão manual
- **Ação:** Questionar Claude sobre contexto específico

### 🔴 **Severity: CRITICAL**
- Código claramente malicioso
- Função proibida (eval, exec, etc.)
- **Ação:** ❌ NÃO INSTALAR

---

## 🛠️ RESUMO RÁPIDO DOS COMANDOS

```bash
# 1. Abrir terminal no VSCode
# Ctrl + `

# 2. Entrar na pasta
cd skills_checker

# 3. Executar auditoria
bash collect_skills_for_audit.sh

# 4. Abrir arquivo gerado
code skills_audit_package_*.md

# 5. No editor VSCode
# Ctrl+A (selecionar tudo)
# Ctrl+C (copiar)

# 6. Claude Code (Ctrl+Shift+I)
# Cola o prompt acima

# 7. Aguarde resultado

# 8. Revise vereditos
```

---

## ❓ Perguntas Frequentes

### **P: Quanto tempo leva para auditar minhas skills?**
**R:** Depende da quantidade:
- Até 10 skills: ~2 minutos
- Até 30 skills: ~5 minutos

### **P: Os dados das minha skills são enviados para a Anthropic?**
**R:** Não. A auditoria é feita **localmente** no seu computador. O arquivo é analisado apenas pelo Claude Code rodando no seu terminal.

### **P: Preciso estar online?**
**R:** Sim, para usar Claude Code. Mas o arquivo é processado localmente — não há upload para servidores.

### **P: Posso auditar apenas algumas skills?**
**R:** Sim. Edite manualmente o arquivo `skills_audit_package_*.md` e remova as skills que não quer auditar antes de enviar ao Claude.

### **P: O que faço se uma skill tiver veredicto ⚠️ REVIEW?**
**R:** Converse com Claude:
```
"Qual é o risco exato de [SKILL_NAME]? Por que precisa de revisão?
É seguro instalar se eu [não usar feature X]?"
```

### **P: Posso reutilizar a auditoria anterior?**
**R:** Sim, se a versão das skills não mudou. Mas recomenda-se gerar novo relatório mensalmente ou após atualizações.

---

## 🔧 Troubleshooting

### **Erro: `command not found: bash`**
```bash
# Se estiver no zsh ou outro shell:
zsh collect_skills_for_audit.sh
```

### **Erro: `Permission denied`**
```bash
chmod +x collect_skills_for_audit.sh
./collect_skills_for_audit.sh
```

### **Nenhuma skill foi detectada**
```bash
# Verifique que as skills estão em:
ls ~/.claude/skills/

# Se estiver vazio, instale uma skill primeiro:
# (via /skill-install no Claude Code)
```

### **Claude não consegue processar o arquivo**
O arquivo pode estar muito grande. Divida em partes:
- Parte 1: Skills A–K
- Parte 2: Skills L–Z

---

## 📋 Checklist de Auditoria

Antes de instalar qualquer nova skill, passe por este checklist:

- [ ] Arquivo `skills_audit_package_*.md` gerado com sucesso
- [ ] Arquivo aberto e conteúdo copiado para Claude Code
- [ ] Prompt de auditoria enviado
- [ ] Relatório completo recebido
- [ ] Todos os vereditos revisados
- [ ] Nenhuma skill com ❌ BLOCK
- [ ] Skills com ⚠️ REVIEW foram questionadas
- [ ] Decisão final tomada

**Pronto! Sua auditoria está segura. 🔒**

---

## 📚 Referências

- **Claude Code Docs:** https://claude.ai/code
- **Skill Registry:** `~/.claude/skills/`
- **SECURITY.md** (se existir no repositório)

---

**Última atualização:** 2026-06-01
