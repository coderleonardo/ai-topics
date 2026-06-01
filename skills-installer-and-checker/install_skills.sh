#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════════
# SCIENTIFIC AGENT SKILLS - INSTALADOR AUTOMÁTICO
# Para: Grafos, SDEs, Matemática e Física
# ═══════════════════════════════════════════════════════════════════════════════

# CONFIGURAÇÕES (edite aqui)
AGENT_TARGET="claude-code"  # Opções: cursor, claude-code, codex, gemini
INSTALL_SCOPE="project"     # Opções: project, user
SKIP_PROMPTS=true           # true = não pede confirmação, false = pede
DELAY_BETWEEN_INSTALLS=3    # Delay em segundos entre instalações

# ═══════════════════════════════════════════════════════════════════════════════

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Função para imprimir com cores
print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${CYAN}⚠️  $1${NC}"
}

# Verificar dependências
check_dependencies() {
    print_header "🔍 Verificando dependências"
    
    if ! command -v gh &> /dev/null; then
        print_error "GitHub CLI (gh) não encontrado"
        echo "Instale com: curl -fsSL https://cli.githubusercontent.com/install.sh | bash"
        exit 1
    fi
    print_success "GitHub CLI encontrado"
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 não encontrado"
        echo "Instale com: sudo apt install python3"
        exit 1
    fi
    print_success "Python 3 encontrado"
}

# Verificar autenticação
check_authentication() {
    print_header "🔐 Verificando autenticação GitHub"
    
    if gh auth status &> /dev/null; then
        print_success "Você está autenticado no GitHub"
        return 0
    else
        print_warning "Você NÃO está autenticado no GitHub"
        echo ""
        echo "Sem autenticação, você terá limites severos de rate limit na API."
        echo "Vamos fazer login agora..."
        echo ""
        
        gh auth login || {
            print_error "Falha ao fazer login"
            echo "Tente fazer login manualmente com: gh auth login"
            exit 1
        }
        
        print_success "Login realizado com sucesso!"
    fi
}

# Mostrar configurações
show_config() {
    print_header "⚙️  CONFIGURAÇÕES DE INSTALAÇÃO"
    echo ""
    echo -e "  ${BLUE}Agente alvo:${NC}      ${GREEN}${AGENT_TARGET}${NC}"
    echo -e "  ${BLUE}Escopo:${NC}           ${GREEN}${INSTALL_SCOPE}${NC}"
    echo -e "  ${BLUE}Delay entre skills:${NC}  ${GREEN}${DELAY_BETWEEN_INSTALLS}s${NC}"
    echo ""
}

# Confirmar instalação
confirm_installation() {
    if [ "$SKIP_PROMPTS" = true ]; then
        return 0
    fi
    
    read -p "Continuar com a instalação? (s/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        return 0
    else
        print_error "Instalação cancelada"
        exit 0
    fi
}

# Instalar skill com flags
install_skill() {
    local skill_name=$1
    local flags="--agent $AGENT_TARGET --scope $INSTALL_SCOPE"
    
    print_info "Instalando: ${GREEN}${skill_name}${NC}"
    gh skill install K-Dense-AI/scientific-agent-skills $skill_name $flags
    
    if [ $? -eq 0 ]; then
        print_success "$skill_name instalada com sucesso"
    else
        print_error "Erro ao instalar $skill_name (rate limit ou erro de conexão)"
    fi
    
    # Delay para evitar rate limit
    print_info "Aguardando ${DELAY_BETWEEN_INSTALLS}s antes da próxima instalação..."
    sleep $DELAY_BETWEEN_INSTALLS
}

# Instalar todas as skills
install_all_skills() {
    print_header "📦 INICIANDO INSTALAÇÃO DE SKILLS"
    echo ""
    
    local total_skills=34
    local current=0
    
    # Machine Learning & Deep Learning
    print_info "📊 Instalando Machine Learning Skills..."
    current=$((current + 1))
    echo "[$current/$total_skills] torch-geometric"
    install_skill "torch-geometric"
    
    current=$((current + 1))
    echo "[$current/$total_skills] pytorch-lightning"
    install_skill "pytorch-lightning"
    
    current=$((current + 1))
    echo "[$current/$total_skills] scikit-learn"
    install_skill "scikit-learn"
    
    current=$((current + 1))
    echo "[$current/$total_skills] transformers"
    install_skill "transformers"
    
    current=$((current + 1))
    echo "[$current/$total_skills] stable-baselines3"
    install_skill "stable-baselines3"
    
    current=$((current + 1))
    echo "[$current/$total_skills] pymc"
    install_skill "pymc"
    
    current=$((current + 1))
    echo "[$current/$total_skills] shap"
    install_skill "shap"
    
    current=$((current + 1))
    echo "[$current/$total_skills] aeon"
    install_skill "aeon"
    
    current=$((current + 1))
    echo "[$current/$total_skills] umap-learn"
    install_skill "umap-learn"
    
    current=$((current + 1))
    echo "[$current/$total_skills] deepchem"
    install_skill "deepchem"
    
    current=$((current + 1))
    echo "[$current/$total_skills] pymoo"
    install_skill "pymoo"
    
    current=$((current + 1))
    echo "[$current/$total_skills] pufferlib"
    install_skill "pufferlib"
    
    echo ""
    # Matemática & Simulação
    print_info "📐 Instalando Skills de Matemática..."
    current=$((current + 1))
    echo "[$current/$total_skills] sympy"
    install_skill "sympy"
    
    current=$((current + 1))
    echo "[$current/$total_skills] statsmodels"
    install_skill "statsmodels"
    
    current=$((current + 1))
    echo "[$current/$total_skills] statistical-analysis"
    install_skill "statistical-analysis"
    
    current=$((current + 1))
    echo "[$current/$total_skills] simpy"
    install_skill "simpy"
    
    current=$((current + 1))
    echo "[$current/$total_skills] dask"
    install_skill "dask"
    
    echo ""
    # Física & Ciências Exatas
    print_info "🌌 Instalando Skills de Física..."
    current=$((current + 1))
    echo "[$current/$total_skills] astropy"
    install_skill "astropy"
    
    current=$((current + 1))
    echo "[$current/$total_skills] qiskit"
    install_skill "qiskit"
    
    current=$((current + 1))
    echo "[$current/$total_skills] cirq"
    install_skill "cirq"
    
    current=$((current + 1))
    echo "[$current/$total_skills] pennylane"
    install_skill "pennylane"
    
    current=$((current + 1))
    echo "[$current/$total_skills] qutip"
    install_skill "qutip"
    
    current=$((current + 1))
    echo "[$current/$total_skills] pymatgen"
    install_skill "pymatgen"
    
    echo ""
    # Análise de Grafos
    print_info "🔗 Instalando Skills de Grafos..."
    current=$((current + 1))
    echo "[$current/$total_skills] networkx"
    install_skill "networkx"
    
    echo ""
    # Visualização
    print_info "📊 Instalando Skills de Visualização..."
    current=$((current + 1))
    echo "[$current/$total_skills] scientific-visualization"
    install_skill "scientific-visualization"
    
    current=$((current + 1))
    echo "[$current/$total_skills] matplotlib"
    install_skill "matplotlib"
    
    current=$((current + 1))
    echo "[$current/$total_skills] seaborn"
    install_skill "seaborn"
    
    current=$((current + 1))
    echo "[$current/$total_skills] exploratory-data-analysis"
    install_skill "exploratory-data-analysis"
    
    echo ""
    # Dados & Processamento
    print_info "⚙️  Instalando Skills de Processamento de Dados..."
    current=$((current + 1))
    echo "[$current/$total_skills] polars"
    install_skill "polars"
    
    current=$((current + 1))
    echo "[$current/$total_skills] vaex"
    install_skill "vaex"
}

# Verificação pós-instalação
post_install_check() {
    print_header "✅ VERIFICAÇÃO PÓS-INSTALAÇÃO"
    echo ""
    
    print_info "Listando skills instaladas..."
    gh skill list 2>/dev/null || echo "  (Nenhuma skill listada - verifique manualmente)"
    
    echo ""
    print_info "Testando imports Python..."
    python3 -c "import sympy; print('  ✓ SymPy')" 2>/dev/null || echo "  ⚠ SymPy (instale com: pip install sympy)"
    python3 -c "import networkx; print('  ✓ NetworkX')" 2>/dev/null || echo "  ⚠ NetworkX (instale com: pip install networkx)"
    python3 -c "import numpy; print('  ✓ NumPy')" 2>/dev/null || echo "  ⚠ NumPy (instale com: pip install numpy)"
}

# Menu de configuração
interactive_config() {
    print_header "⚙️  CONFIGURAÇÃO INTERATIVA"
    echo ""
    
    echo "Escolha o agente alvo:"
    echo "  1) Claude Code (padrão)"
    echo "  2) Cursor"
    echo "  3) Codex"
    echo "  4) Gemini"
    read -p "Opção (1-4): " agent_choice
    
    case $agent_choice in
        1) AGENT_TARGET="claude-code" ;;
        2) AGENT_TARGET="cursor" ;;
        3) AGENT_TARGET="codex" ;;
        4) AGENT_TARGET="gemini" ;;
        *) AGENT_TARGET="claude-code" ;;
    esac
    
    echo ""
    echo "Escolha o escopo de instalação:"
    echo "  1) Project (apenas neste projeto)"
    echo "  2) User (para todos os projetos do usuário)"
    read -p "Opção (1-2): " scope_choice
    
    case $scope_choice in
        1) INSTALL_SCOPE="project" ;;
        2) INSTALL_SCOPE="user" ;;
        *) INSTALL_SCOPE="project" ;;
    esac
    
    echo ""
    echo "Configurar delay entre instalações? (padrão: 3s)"
    read -p "Segundos (Enter para 3): " delay_choice
    
    if [ -z "$delay_choice" ]; then
        DELAY_BETWEEN_INSTALLS=3
    else
        DELAY_BETWEEN_INSTALLS=$delay_choice
    fi
    
    echo ""
    SKIP_PROMPTS=true
}

# Menu principal
show_menu() {
    print_header "🚀 SCIENTIFIC AGENT SKILLS INSTALLER"
    echo ""
    echo "  1) Instalar com configurações padrão (Claude Code, Project, 3s delay)"
    echo "  2) Configurar manualmente"
    echo "  3) Sair"
    echo ""
    read -p "Escolha uma opção (1-3): " menu_choice
    
    case $menu_choice in
        1)
            AGENT_TARGET="claude-code"
            INSTALL_SCOPE="project"
            DELAY_BETWEEN_INSTALLS=3
            ;;
        2)
            interactive_config
            ;;
        3)
            print_info "Saindo..."
            exit 0
            ;;
        *)
            print_error "Opção inválida"
            exit 1
            ;;
    esac
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    # Se executado sem argumentos, mostrar menu
    if [ $# -eq 0 ]; then
        show_menu
    else
        # Processar argumentos
        while [[ $# -gt 0 ]]; do
            case $1 in
                --agent)
                    AGENT_TARGET="$2"
                    shift 2
                    ;;
                --scope)
                    INSTALL_SCOPE="$2"
                    shift 2
                    ;;
                --project)
                    INSTALL_SCOPE="project"
                    shift
                    ;;
                --user)
                    INSTALL_SCOPE="user"
                    shift
                    ;;
                --delay)
                    DELAY_BETWEEN_INSTALLS="$2"
                    shift 2
                    ;;
                --skip-prompts)
                    SKIP_PROMPTS=true
                    shift
                    ;;
                --help)
                    echo "Uso: ./install_skills.sh [opções]"
                    echo ""
                    echo "Opções:"
                    echo "  --agent {claude-code|cursor|codex|gemini}  Agente alvo"
                    echo "  --scope {project|user}                      Escopo de instalação"
                    echo "  --project                                    Instalar no projeto (padrão)"
                    echo "  --user                                       Instalar para o usuário"
                    echo "  --delay SEGUNDOS                             Delay entre instalações (padrão: 3)"
                    echo "  --skip-prompts                              Sem prompts interativos"
                    echo "  --help                                      Mostrar ajuda"
                    echo ""
                    echo "Exemplos:"
                    echo "  ./install_skills.sh                                         # Menu interativo"
                    echo "  ./install_skills.sh --agent cursor --project               # Cursor project"
                    echo "  ./install_skills.sh --agent claude-code --user             # Claude Code user"
                    echo "  ./install_skills.sh --agent codex --scope user --delay 5"
                    exit 0
                    ;;
                *)
                    print_error "Argumento desconhecido: $1"
                    exit 1
                    ;;
            esac
        done
    fi
    
    check_dependencies
    check_authentication
    show_config
    confirm_installation
    install_all_skills
    post_install_check
    
    echo ""
    print_header "🎉 INSTALAÇÃO CONCLUÍDA!"
    echo ""
    echo -e "${GREEN}Suas skills estão prontas para usar!${NC}"
    echo ""
    echo "Próximos passos:"
    echo "  1. Reinicie seu editor (Claude Code, Cursor, etc)"
    echo "  2. As skills serão descobertas automaticamente"
    echo "  3. Use-as mencionando o nome da skill em seus prompts"
    echo ""
}

# Executar main
main "$@"