#!/usr/bin/env bash

# ═══════════════════════════════════════════════════════════════════════════════
# SKILL INSTALLER
# Bulk-installs a bundle of agent skills from a GitHub repo, via `gh skill install`.
# The source repo and the list of skills come from a config file (see --config),
# so this script works with any skill bundle, not just the bundled example.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="$SCRIPT_DIR/../config/skills.example.conf"

# Defaults (overridable via flags, or via the config file for SOURCE_REPO)
AGENT_TARGET="claude-code"   # Options: cursor, claude-code, codex, gemini
INSTALL_SCOPE="project"      # Options: project, user
SKIP_PROMPTS=true            # true = no confirmation prompt, false = ask before installing
DELAY_BETWEEN_INSTALLS=3     # Seconds to wait between installs (avoids API rate limits)
CONFIG_FILE="$DEFAULT_CONFIG"
REPO_OVERRIDE=""
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }
print_info()    { echo -e "${YELLOW}ℹ️  $1${NC}"; }
print_warning() { echo -e "${CYAN}⚠️  $1${NC}"; }

usage() {
    cat << 'EOF'
Usage: ./install_skills.sh [options]

Options:
  --config PATH                                Path to a skills config file (default: ../config/skills.example.conf)
  --repo OWNER/REPO                            Source repo to install skills from (overrides SOURCE_REPO in the config file)
  --agent {claude-code|cursor|codex|gemini}    Target agent (default: claude-code)
  --scope {project|user}                       Install scope (default: project)
  --project                                     Shorthand for --scope project
  --user                                        Shorthand for --scope user
  --delay SECONDS                               Delay between installs, in seconds (default: 3)
  --skip-prompts                                Skip the confirmation prompt (default: on)
  --confirm                                     Ask for confirmation before installing (disables --skip-prompts)
  --dry-run                                     Print what would be installed without calling `gh`
  --help                                        Show this help

Examples:
  ./install_skills.sh                                              # Install the example bundle, interactive menu if no other flags
  ./install_skills.sh --config my-skills.conf --agent cursor       # Use your own bundle, target Cursor
  ./install_skills.sh --repo my-org/my-skills --user               # Override the source repo, install for the whole user
  ./install_skills.sh --dry-run                                    # See what would be installed, without installing anything
EOF
}

check_dependencies() {
    print_header "🔍 Checking dependencies"

    if ! command -v gh &> /dev/null; then
        print_error "GitHub CLI (gh) not found"
        echo "Install it from: https://cli.github.com/"
        exit 1
    fi
    print_success "GitHub CLI found"

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        echo "Install it with your system's package manager, e.g.: sudo apt install python3"
        exit 1
    fi
    print_success "Python 3 found"
}

check_authentication() {
    print_header "🔐 Checking GitHub authentication"

    if gh auth status &> /dev/null; then
        print_success "You are authenticated with GitHub"
        return 0
    fi

    print_warning "You are NOT authenticated with GitHub"
    echo ""
    echo "Without authentication, you'll hit severe API rate limits."
    echo "Let's log in now..."
    echo ""

    gh auth login || {
        print_error "Login failed"
        echo "Try logging in manually with: gh auth login"
        exit 1
    }

    print_success "Login successful!"
}

# Parses CONFIG_FILE into SOURCE_REPO and the SKILLS array.
# Lines starting with "#" are comments; a SOURCE_REPO="..." line sets the repo;
# every other non-blank line is treated as a skill name.
load_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi

    SOURCE_REPO=""
    SKILLS=()

    while IFS= read -r line || [[ -n "$line" ]]; do
        line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

        [[ -z "$line" ]] && continue
        [[ "$line" == \#* ]] && continue

        if [[ "$line" =~ ^SOURCE_REPO= ]]; then
            SOURCE_REPO="${line#SOURCE_REPO=}"
            SOURCE_REPO="${SOURCE_REPO%\"}"
            SOURCE_REPO="${SOURCE_REPO#\"}"
            continue
        fi

        SKILLS+=("$line")
    done < "$CONFIG_FILE"

    if [[ -n "$REPO_OVERRIDE" ]]; then
        SOURCE_REPO="$REPO_OVERRIDE"
    fi

    if [[ -z "$SOURCE_REPO" ]]; then
        print_error "No source repo configured. Set SOURCE_REPO in $CONFIG_FILE or pass --repo <owner/repo>."
        exit 1
    fi

    if [[ ${#SKILLS[@]} -eq 0 ]]; then
        print_error "No skills found in config file: $CONFIG_FILE"
        exit 1
    fi
}

show_config() {
    print_header "⚙️  INSTALLATION SETTINGS"
    echo ""
    echo -e "  ${BLUE}Source repo:${NC}      ${GREEN}${SOURCE_REPO}${NC}"
    echo -e "  ${BLUE}Config file:${NC}      ${GREEN}${CONFIG_FILE}${NC}"
    echo -e "  ${BLUE}Skills to install:${NC} ${GREEN}${#SKILLS[@]}${NC}"
    echo -e "  ${BLUE}Target agent:${NC}     ${GREEN}${AGENT_TARGET}${NC}"
    echo -e "  ${BLUE}Scope:${NC}            ${GREEN}${INSTALL_SCOPE}${NC}"
    echo -e "  ${BLUE}Delay between installs:${NC} ${GREEN}${DELAY_BETWEEN_INSTALLS}s${NC}"
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "  ${BLUE}Mode:${NC}             ${YELLOW}dry-run (nothing will actually be installed)${NC}"
    fi
    echo ""
}

confirm_installation() {
    if [[ "$SKIP_PROMPTS" == true ]] || [[ "$DRY_RUN" == true ]]; then
        return 0
    fi

    read -r -p "Continue with installation? (y/n) " -n 1
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    fi

    print_error "Installation cancelled"
    exit 0
}

install_skill() {
    local skill_name=$1
    local flags="--agent $AGENT_TARGET --scope $INSTALL_SCOPE"

    print_info "Installing: ${GREEN}${skill_name}${NC}"

    if [[ "$DRY_RUN" == true ]]; then
        print_info "[dry-run] gh skill install $SOURCE_REPO $skill_name $flags"
        return 0
    fi

    if gh skill install "$SOURCE_REPO" "$skill_name" $flags; then
        print_success "$skill_name installed successfully"
    else
        print_error "Failed to install $skill_name (rate limit or connection error)"
    fi

    print_info "Waiting ${DELAY_BETWEEN_INSTALLS}s before the next install..."
    sleep "$DELAY_BETWEEN_INSTALLS"
}

install_all_skills() {
    print_header "📦 STARTING SKILL INSTALLATION"
    echo ""

    local total=${#SKILLS[@]}
    local current=0

    for skill in "${SKILLS[@]}"; do
        current=$((current + 1))
        echo "[$current/$total] $skill"
        install_skill "$skill"
    done
}

post_install_check() {
    print_header "✅ POST-INSTALL CHECK"
    echo ""

    print_info "Listing installed skills..."
    gh skill list 2>/dev/null || echo "  (No skills listed - check manually)"
}

interactive_config() {
    print_header "⚙️  INTERACTIVE CONFIGURATION"
    echo ""

    echo "Choose the target agent:"
    echo "  1) Claude Code (default)"
    echo "  2) Cursor"
    echo "  3) Codex"
    echo "  4) Gemini"
    read -r -p "Option (1-4): " agent_choice

    case $agent_choice in
        1) AGENT_TARGET="claude-code" ;;
        2) AGENT_TARGET="cursor" ;;
        3) AGENT_TARGET="codex" ;;
        4) AGENT_TARGET="gemini" ;;
        *) AGENT_TARGET="claude-code" ;;
    esac

    echo ""
    echo "Choose the install scope:"
    echo "  1) Project (this project only)"
    echo "  2) User (all of your projects)"
    read -r -p "Option (1-2): " scope_choice

    case $scope_choice in
        1) INSTALL_SCOPE="project" ;;
        2) INSTALL_SCOPE="user" ;;
        *) INSTALL_SCOPE="project" ;;
    esac

    echo ""
    echo "Delay between installs? (default: 3s)"
    read -r -p "Seconds (Enter for 3): " delay_choice
    DELAY_BETWEEN_INSTALLS="${delay_choice:-3}"

    echo ""
    SKIP_PROMPTS=true
}

show_menu() {
    print_header "🚀 SKILL INSTALLER"
    echo ""
    echo "  1) Install with default settings (Claude Code, Project, 3s delay)"
    echo "  2) Configure manually"
    echo "  3) Exit"
    echo ""
    read -r -p "Choose an option (1-3): " menu_choice

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
            print_info "Bye!"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            exit 1
            ;;
    esac
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --agent) AGENT_TARGET="$2"; shift 2 ;;
            --scope) INSTALL_SCOPE="$2"; shift 2 ;;
            --project) INSTALL_SCOPE="project"; shift ;;
            --user) INSTALL_SCOPE="user"; shift ;;
            --delay) DELAY_BETWEEN_INSTALLS="$2"; shift 2 ;;
            --skip-prompts) SKIP_PROMPTS=true; shift ;;
            --confirm) SKIP_PROMPTS=false; shift ;;
            --config) CONFIG_FILE="$2"; shift 2 ;;
            --repo) REPO_OVERRIDE="$2"; shift 2 ;;
            --dry-run) DRY_RUN=true; shift ;;
            --help) usage; exit 0 ;;
            *)
                print_error "Unknown argument: $1"
                usage
                exit 1
                ;;
        esac
    done
}

main() {
    # With no arguments at all, fall back to the interactive menu (matches
    # historical behavior); any flag skips straight to flag-driven execution.
    if [[ $# -eq 0 ]]; then
        show_menu
    else
        parse_args "$@"
    fi

    load_config
    check_dependencies
    if [[ "$DRY_RUN" != true ]]; then
        check_authentication
    fi
    show_config
    confirm_installation
    install_all_skills
    if [[ "$DRY_RUN" != true ]]; then
        post_install_check
    fi

    echo ""
    print_header "🎉 INSTALLATION COMPLETE!"
    echo ""
    echo -e "${GREEN}Your skills are ready to use!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Restart your editor/agent (Claude Code, Cursor, etc.)"
    echo "  2. Skills will be discovered automatically"
    echo "  3. Use them by mentioning the skill's name in your prompts"
    echo ""
}

main "$@"
