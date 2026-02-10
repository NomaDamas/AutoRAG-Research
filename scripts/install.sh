#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# AutoRAG-Research  ·  End-User Installation Script
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/NomaDamas/AutoRAG-Research/main/scripts/install.sh -o install.sh
#   bash install.sh
#
# Non-interactive (CI):
#   AUTORAG_RESEARCH_NONINTERACTIVE=1 AUTORAG_RESEARCH_PG_SETUP=skip bash install.sh
#
# Optional dependencies (non-interactive):
#   AUTORAG_RESEARCH_OPTIONAL_DEPS=all      # Install all extras (default)
#   AUTORAG_RESEARCH_OPTIONAL_DEPS=none     # Base package only
#   AUTORAG_RESEARCH_OPTIONAL_DEPS=gpu,reranker  # Specific groups
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Constants ────────────────────────────────────────────────────────
GITHUB_REPO="NomaDamas/AutoRAG-Research"
GITHUB_BRANCH="main"
GITHUB_RAW="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}"
SCHEMA_URL="${GITHUB_RAW}/postgresql/db/init/001-schema.sql"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

# ── Non-interactive env vars ─────────────────────────────────────────
NONINTERACTIVE="${AUTORAG_RESEARCH_NONINTERACTIVE:-0}"
NI_PYTHON_ENV="${AUTORAG_RESEARCH_PYTHON_ENV:-uv}"
NI_PG_SETUP="${AUTORAG_RESEARCH_PG_SETUP:-docker}"
NI_PG_HOST="${AUTORAG_RESEARCH_PG_HOST:-localhost}"
NI_PG_PORT="${AUTORAG_RESEARCH_PG_PORT:-5432}"
NI_PG_USER="${AUTORAG_RESEARCH_PG_USER:-postgres}"
NI_PG_PASSWORD="${AUTORAG_RESEARCH_PG_PASSWORD:-postgres}"
NI_PG_DB="${AUTORAG_RESEARCH_PG_DB:-autorag}"
NI_CONTAINER="${AUTORAG_RESEARCH_CONTAINER_NAME:-autorag_postgres}"
NI_OPTIONAL_DEPS="${AUTORAG_RESEARCH_OPTIONAL_DEPS:-all}"

# ── Saved state across phases ────────────────────────────────────────
DB_YAML_CONTENT=""
VENV_PATH=""
USED_UV=0
SELECTED_EXTRAS=""

# ── Color helpers ────────────────────────────────────────────────────
setup_colors() {
    if [[ -t 1 ]]; then
        BOLD='\033[1m'
        DIM='\033[2m'
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        YELLOW='\033[0;33m'
        BLUE='\033[0;34m'
        CYAN='\033[0;36m'
        RESET='\033[0m'
    else
        BOLD='' DIM='' RED='' GREEN='' YELLOW='' BLUE='' CYAN='' RESET=''
    fi
}

info()    { echo -e "${BLUE}::${RESET} $*"; }
success() { echo -e "${GREEN}::${RESET} $*"; }
warn()    { echo -e "${YELLOW}:: WARNING:${RESET} $*"; }
error()   { echo -e "${RED}:: ERROR:${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}▸ $*${RESET}\n"; }

# ── Interactive helpers ──────────────────────────────────────────────
# ask VARNAME "prompt" "default"
ask() {
    local varname="$1" prompt="$2" default="${3:-}"
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        eval "$varname=\"$default\""
        return
    fi
    local input
    if [[ -n "$default" ]]; then
        read -rp "  ${prompt} [${default}]: " input
        eval "$varname=\"${input:-$default}\""
    else
        read -rp "  ${prompt}: " input
        eval "$varname=\"$input\""
    fi
}

# select_option VARNAME "prompt" "opt1" "opt2" ...
# Sets VARNAME to the 1-based index of the chosen option.
select_option() {
    local varname="$1" prompt="$2"
    shift 2
    local options=("$@")
    echo -e "  ${BOLD}${prompt}${RESET}"
    local i=1
    for opt in "${options[@]}"; do
        echo -e "    ${CYAN}${i})${RESET} ${opt}"
        ((i++))
    done
    local choice
    while true; do
        read -rp "  Enter choice [1-${#options[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
            eval "$varname=$choice"
            return
        fi
        echo "  Please enter a number between 1 and ${#options[@]}"
    done
}

confirm() {
    local prompt="$1" default="${2:-y}"
    if [[ "$NONINTERACTIVE" == "1" ]]; then return 0; fi
    local yn
    read -rp "  ${prompt} [Y/n]: " yn
    yn="${yn:-$default}"
    [[ "$yn" =~ ^[Yy] ]]
}

# ── Optional dependency selection ──────────────────────────────────────
VALID_GROUPS="gpu reporting reranker"

select_optional_deps() {
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        # Non-interactive: read from env var
        local deps="$NI_OPTIONAL_DEPS"
        if [[ "$deps" == "none" ]]; then
            SELECTED_EXTRAS=""
            info "Skipping optional dependencies (base package only)"
            return
        fi
        if [[ "$deps" == "all" ]]; then
            SELECTED_EXTRAS="[all]"
            info "Will install all optional dependencies"
            return
        fi
        # Validate comma-separated group names
        local validated="" invalid=""
        IFS=',' read -ra groups <<< "$deps"
        for g in "${groups[@]}"; do
            g=$(echo "$g" | xargs)  # trim whitespace
            if [[ " $VALID_GROUPS " == *" $g "* ]]; then
                if [[ -n "$validated" ]]; then
                    validated="${validated},${g}"
                else
                    validated="$g"
                fi
            else
                if [[ -n "$invalid" ]]; then
                    invalid="${invalid}, ${g}"
                else
                    invalid="$g"
                fi
            fi
        done
        if [[ -n "$invalid" ]]; then
            warn "Unknown optional dependency group(s): ${invalid}"
            warn "Valid groups are: ${VALID_GROUPS}"
        fi
        if [[ -n "$validated" ]]; then
            SELECTED_EXTRAS="[${validated}]"
            info "Will install optional dependencies: ${validated}"
        else
            SELECTED_EXTRAS=""
            warn "No valid groups specified, installing base package only"
        fi
        return
    fi

    # Interactive mode
    echo ""
    local choice
    select_option choice \
        "Which optional dependencies would you like to install?" \
        "Install all optional dependencies (Recommended)" \
        "Select specific groups to install" \
        "Skip optional dependencies (base package only)"

    case "$choice" in
        1)
            SELECTED_EXTRAS="[all]"
            ;;
        2)
            echo ""
            echo -e "  ${BOLD}Select which optional dependency groups to install:${RESET}"
            echo ""
            local selected=""
            if confirm "gpu - GPU-accelerated embeddings (colpali-engine, torch, transformers)?"; then
                selected="gpu"
            fi
            if confirm "reporting - Reporting & dashboards (duckdb, gradio)?"; then
                if [[ -n "$selected" ]]; then
                    selected="${selected},reporting"
                else
                    selected="reporting"
                fi
            fi
            if confirm "reranker - Reranking models (cohere, voyageai)?"; then
                if [[ -n "$selected" ]]; then
                    selected="${selected},reranker"
                else
                    selected="reranker"
                fi
            fi
            if [[ -n "$selected" ]]; then
                SELECTED_EXTRAS="[${selected}]"
                echo ""
                info "Will install optional dependencies: ${selected}"
            else
                SELECTED_EXTRAS=""
                echo ""
                info "No groups selected, installing base package only"
            fi
            ;;
        3)
            SELECTED_EXTRAS=""
            ;;
    esac
}

# ── Detection helpers ────────────────────────────────────────────────
has_command() { command -v "$1" &>/dev/null; }

detect_os() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux)
            if has_command apt-get; then echo "debian"
            elif has_command dnf; then echo "fedora"
            else echo "linux"
            fi ;;
        *) echo "unknown" ;;
    esac
}

# python_version CMD -> "3.12" or empty
python_version() {
    local cmd="$1"
    if has_command "$cmd"; then
        "$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true
    fi
}

# version_gte "3.12" "3.10" -> true
version_gte() {
    local major1="${1%%.*}" minor1="${1#*.}"
    local major2="${2%%.*}" minor2="${2#*.}"
    (( major1 > major2 || (major1 == major2 && minor1 >= minor2) ))
}

detect_venv() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then echo "venv"; return; fi
    if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then echo "conda"; return; fi
    if [[ -n "${PIPENV_ACTIVE:-}" ]]; then echo "pipenv"; return; fi
    if [[ -n "${PYENV_VIRTUAL_ENV:-}" ]]; then echo "pyenv"; return; fi
    echo "none"
}

# ── Pipe detection ───────────────────────────────────────────────────
check_not_piped() {
    if [[ ! -t 0 ]]; then
        echo ""
        echo "This script requires an interactive terminal."
        echo "Please download and run it instead of piping:"
        echo ""
        echo "  curl -LsSf ${GITHUB_RAW}/scripts/install.sh -o install.sh"
        echo "  bash install.sh"
        echo ""
        exit 1
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  Phase 1: Python Environment
# ══════════════════════════════════════════════════════════════════════
phase_python_env() {
    header "Phase 1/3: Python Environment"

    local current_venv
    current_venv=$(detect_venv)

    # Already in a virtual environment
    if [[ "$current_venv" != "none" ]]; then
        local py_ver
        py_ver=$(python_version python3)
        if [[ -z "$py_ver" ]]; then
            py_ver=$(python_version python)
        fi
        if [[ -n "$py_ver" ]] && version_gte "$py_ver" "${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}"; then
            success "Using active ${current_venv} environment (Python ${py_ver})"
            VENV_PATH="${VIRTUAL_ENV:-${CONDA_PREFIX:-}}"
            # Check if uv is available
            if has_command uv; then USED_UV=1; fi
            return
        else
            error "Active environment has Python ${py_ver:-unknown}, but >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} is required."
            exit 1
        fi
    fi

    # Not in a venv — offer choices
    local choice
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        case "$NI_PYTHON_ENV" in
            uv)   choice=1 ;;
            venv) choice=2 ;;
            skip) choice=3 ;;
            *)    choice=1 ;;
        esac
    else
        select_option choice \
            "How would you like to set up your Python environment?" \
            "Create a new virtual environment with uv (recommended)" \
            "Create a new virtual environment with python -m venv" \
            "I will set up my own environment (print instructions and exit)"
    fi

    case "$choice" in
        1) setup_env_uv ;;
        2) setup_env_venv ;;
        3) print_manual_env_instructions; exit 0 ;;
    esac
}

setup_env_uv() {
    USED_UV=1
    if ! has_command uv; then
        info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Source the env so uv is available
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        if ! has_command uv; then
            error "uv installation failed. Please install it manually: https://docs.astral.sh/uv/"
            exit 1
        fi
        success "uv installed"
    else
        success "uv is already installed"
    fi

    info "Creating virtual environment with uv..."
    uv venv .venv --python ">=${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}"

    # shellcheck disable=SC1091
    source .venv/bin/activate
    VENV_PATH="$(pwd)/.venv"
    success "Virtual environment activated at ${VENV_PATH}"
}

setup_env_venv() {
    # Find the best available Python
    local best_cmd="" best_ver=""
    for minor in 13 12 11 10; do
        local cmd="python3.${minor}"
        if has_command "$cmd"; then
            best_cmd="$cmd"
            best_ver="3.${minor}"
            break
        fi
    done

    # Fallback to python3
    if [[ -z "$best_cmd" ]]; then
        local ver
        ver=$(python_version python3)
        if [[ -n "$ver" ]] && version_gte "$ver" "${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}"; then
            best_cmd="python3"
            best_ver="$ver"
        fi
    fi

    if [[ -z "$best_cmd" ]]; then
        error "No Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} found."
        error "Please install Python 3.10+ and try again."
        exit 1
    fi

    info "Using ${best_cmd} (Python ${best_ver})"

    info "Creating virtual environment..."
    "$best_cmd" -m venv .venv

    # shellcheck disable=SC1091
    source .venv/bin/activate
    VENV_PATH="$(pwd)/.venv"

    # Check if uv is available for install phase
    if has_command uv; then USED_UV=1; fi
    success "Virtual environment activated at ${VENV_PATH} (Python ${best_ver})"
}

print_manual_env_instructions() {
    echo ""
    echo -e "${BOLD}Manual Environment Setup Instructions${RESET}"
    echo ""
    echo "  Set up a Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} environment using your preferred tool:"
    echo ""
    echo -e "  ${CYAN}conda:${RESET}"
    echo "    conda create -n autorag python=3.12 -y"
    echo "    conda activate autorag"
    echo ""
    echo -e "  ${CYAN}pyenv + virtualenv:${RESET}"
    echo "    pyenv install 3.12"
    echo "    pyenv virtualenv 3.12 autorag"
    echo "    pyenv activate autorag"
    echo ""
    echo -e "  ${CYAN}venv:${RESET}"
    echo "    python3 -m venv .venv"
    echo "    source .venv/bin/activate"
    echo ""
    echo -e "  ${CYAN}uv:${RESET}"
    echo "    uv venv .venv --python '>=3.10'"
    echo "    source .venv/bin/activate"
    echo ""
    echo "  Then re-run this script inside the activated environment."
    echo ""
}

# ══════════════════════════════════════════════════════════════════════
#  Phase 2: Install AutoRAG-Research Package
# ══════════════════════════════════════════════════════════════════════
phase_install_package() {
    header "Phase 2/3: Install AutoRAG-Research"

    # Check if already installed
    if python3 -c "import autorag_research" &>/dev/null; then
        local current_ver
        current_ver=$(python3 -c "from importlib.metadata import version; print(version('autorag-research'))" 2>/dev/null || echo "unknown")
        info "autorag-research ${current_ver} is already installed"
        if [[ "$NONINTERACTIVE" != "1" ]]; then
            if confirm "Upgrade to latest version?"; then
                select_optional_deps
                do_install --upgrade
                return
            else
                success "Keeping current installation"
                return
            fi
        else
            select_optional_deps
            do_install --upgrade
            return
        fi
    fi

    select_optional_deps
    do_install
}

do_install() {
    local upgrade_flag="${1:-}"
    local pkg_spec="autorag-research${SELECTED_EXTRAS}"

    if [[ "$USED_UV" == "1" ]]; then
        # Try uv add first (works in uv-managed projects with pyproject.toml)
        if [[ -f "pyproject.toml" ]]; then
            info "Installing with uv add..."
            if uv add "${pkg_spec}" 2>/dev/null; then
                success "${pkg_spec} installed via uv add"
                return
            fi
            info "uv add failed, falling back to uv pip install..."
        fi

        info "Installing with uv pip install..."
        if [[ "$upgrade_flag" == "--upgrade" ]]; then
            uv pip install --upgrade "${pkg_spec}"
        else
            uv pip install "${pkg_spec}"
        fi
    else
        info "Installing with pip..."
        if [[ "$upgrade_flag" == "--upgrade" ]]; then
            pip install --upgrade "${pkg_spec}"
        else
            pip install "${pkg_spec}"
        fi
    fi

    success "${pkg_spec} installed"
}

# ══════════════════════════════════════════════════════════════════════
#  Phase 3: PostgreSQL Setup
# ══════════════════════════════════════════════════════════════════════
phase_postgresql() {
    header "Phase 3/3: PostgreSQL Setup"

    local choice
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        case "$NI_PG_SETUP" in
            docker)   choice=1 ;;
            existing) choice=2 ;;
            skip)     choice=3 ;;
            *)        choice=3 ;;
        esac
    else
        select_option choice \
            "How would you like to set up PostgreSQL?" \
            "Docker (recommended) - automatic setup with VectorChord" \
            "Use an existing PostgreSQL server" \
            "Skip - I will set up PostgreSQL later"
    fi

    case "$choice" in
        1) setup_pg_docker ;;
        2) setup_pg_existing ;;
        3) skip_pg_setup ;;
    esac
}

install_docker() {
    local os
    os=$(detect_os)
    case "$os" in
        macos)
            if has_command brew; then
                info "Installing Docker Desktop via Homebrew..."
                brew install --cask docker
                echo ""
                info "Please start Docker Desktop from Applications, then press Enter to continue."
                read -rp "  Press Enter when Docker is running..."
            else
                error "Please install Docker Desktop from https://www.docker.com/products/docker-desktop/"
                error "Then re-run this script."
                exit 1
            fi
            ;;
        debian)
            info "Installing Docker via official convenience script..."
            curl -fsSL https://get.docker.com | sh
            info "Adding current user to docker group..."
            sudo usermod -aG docker "$USER"
            warn "You may need to log out and back in for group changes to take effect."
            ;;
        fedora)
            info "Installing Docker via official convenience script..."
            curl -fsSL https://get.docker.com | sh
            sudo systemctl enable --now docker
            sudo usermod -aG docker "$USER"
            warn "You may need to log out and back in for group changes to take effect."
            ;;
        *)
            error "Could not auto-install Docker on this system."
            error "Please install Docker manually: https://docs.docker.com/get-docker/"
            exit 1
            ;;
    esac
}

setup_pg_docker() {
    # Check / install Docker
    if ! has_command docker; then
        warn "Docker is not installed."
        if [[ "$NONINTERACTIVE" == "1" ]]; then
            install_docker
        else
            if confirm "Install Docker now?"; then
                install_docker
            else
                error "Docker is required for this option. Please install Docker and try again."
                exit 1
            fi
        fi
    fi

    # Verify Docker is running
    if ! docker info &>/dev/null; then
        error "Docker is installed but not running. Please start Docker and try again."
        exit 1
    fi

    # Gather settings
    local pg_password pg_port container_name pg_db
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        pg_password="$NI_PG_PASSWORD"
        pg_port="$NI_PG_PORT"
        container_name="$NI_CONTAINER"
        pg_db="$NI_PG_DB"
    else
        ask pg_password "PostgreSQL password" "postgres"
        ask pg_port "Host port for PostgreSQL" "5432"
        ask container_name "Docker container name" "autorag_postgres"
        ask pg_db "Database name" "autorag"
    fi

    # Check for existing container
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
            # Verify the running container's port matches what the user expects
            local actual_port
            actual_port=$(docker port "$container_name" 5432 2>/dev/null | grep -oE '[0-9]+$' || true)
            if [[ -n "$actual_port" && "$actual_port" != "$pg_port" ]]; then
                warn "Container '${container_name}' is running on port ${actual_port}, not ${pg_port}. Using actual port."
                pg_port="$actual_port"
            fi
            success "Container '${container_name}' is already running"
            write_db_yaml "localhost" "$pg_port" "postgres" "$pg_password"
            return
        else
            info "Container '${container_name}' exists but is stopped"
            if [[ "$NONINTERACTIVE" == "1" ]] || confirm "Start existing container?"; then
                docker start "$container_name"
                wait_for_pg "localhost" "$pg_port" "postgres" "$pg_password"
                success "Container '${container_name}' started"
                write_db_yaml "localhost" "$pg_port" "postgres" "$pg_password"
                return
            fi
        fi
    fi

    # Create postgresql directory structure
    local pg_dir="./postgresql"
    mkdir -p "${pg_dir}/db/init"

    # Write .env file
    cat > "${pg_dir}/.env" <<EOF
POSTGRES_DB=${pg_db}
POSTGRES_USER=postgres
POSTGRES_PASSWORD=${pg_password}
POSTGRES_PORT=${pg_port}
CONTAINER_NAME=${container_name}
EOF

    # Write docker-compose.yml (inline — script is standalone)
    cat > "${pg_dir}/docker-compose.yml" <<'COMPOSE'
services:
  db:
    # TensorChord VectorChord Suite image (includes VectorChord, VectorChord-BM25, pg_tokenizer)
    image: tensorchord/vchord-suite:pg18-latest
    container_name: ${CONTAINER_NAME:-autorag_postgres}
    env_file:
      - .env
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-autorag}
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - ./pgdata:/var/lib/postgresql/18/docker
      - ./db/init:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
COMPOSE

    # Download schema SQL
    info "Downloading database schema..."
    if curl -fsSL "$SCHEMA_URL" -o "${pg_dir}/db/init/001-schema.sql"; then
        success "Schema downloaded"
    else
        warn "Failed to download schema. You can download it manually later:"
        warn "  curl -fsSL ${SCHEMA_URL} -o ${pg_dir}/db/init/001-schema.sql"
    fi

    # Start container
    info "Starting PostgreSQL container..."
    docker compose -f "${pg_dir}/docker-compose.yml" up -d

    wait_for_pg "localhost" "$pg_port" "postgres" "$pg_password"
    success "PostgreSQL is ready (port ${pg_port})"

    write_db_yaml "localhost" "$pg_port" "postgres" "$pg_password"
}

wait_for_pg() {
    local host="$1" port="$2" user="$3" password="$4"
    local retries=30
    info "Waiting for PostgreSQL to be ready..."
    for ((i=1; i<=retries; i++)); do
        if has_command pg_isready; then
            if PGPASSWORD="$password" pg_isready -h "$host" -p "$port" -U "$user" &>/dev/null; then
                return 0
            fi
        else
            # Fallback: try a TCP connection
            if (echo > "/dev/tcp/${host}/${port}") &>/dev/null; then
                # Give it a moment after TCP is available (PG may accept TCP before it's query-ready)
                sleep 10
                return 0
            fi
        fi
        sleep 1
    done
    warn "PostgreSQL did not become ready within ${retries}s — it may still be starting."
}

setup_pg_existing() {
    echo ""
    echo -e "  ${YELLOW}Note:${RESET} The schema initialization script (001-schema.sql) will install"
    echo "  VectorChord extensions (vector search + BM25) on your PostgreSQL server."
    echo "  Your server must support these extensions (PostgreSQL 18 with VectorChord suite)."
    echo ""

    local pg_host pg_port pg_user pg_password pg_db
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        pg_host="$NI_PG_HOST"
        pg_port="$NI_PG_PORT"
        pg_user="$NI_PG_USER"
        pg_password="$NI_PG_PASSWORD"
        pg_db="$NI_PG_DB"
    else
        ask pg_host "PostgreSQL host" "localhost"
        ask pg_port "PostgreSQL port" "5432"
        ask pg_user "PostgreSQL user" "postgres"
        ask pg_password "PostgreSQL password" "postgres"
        ask pg_db "Database name" "autorag"
    fi

    # Test connection if psql is available
    if has_command psql; then
        info "Testing connection..."
        if PGPASSWORD="$pg_password" psql -h "$pg_host" -p "$pg_port" -U "$pg_user" -d "$pg_db" -c "SELECT 1;" &>/dev/null; then
            success "Connection successful"
        else
            warn "Could not connect. Please verify your settings."
            if [[ "$NONINTERACTIVE" != "1" ]]; then
                if ! confirm "Continue anyway?"; then
                    exit 1
                fi
            fi
        fi

        # Offer to run schema SQL
        info "Downloading schema initialization script..."
        local schema_file
        schema_file=$(mktemp)
        if curl -fsSL "$SCHEMA_URL" -o "$schema_file"; then
            if [[ "$NONINTERACTIVE" == "1" ]] || confirm "Run schema initialization on the database?"; then
                info "Applying schema..."
                if PGPASSWORD="$pg_password" psql -h "$pg_host" -p "$pg_port" -U "$pg_user" -d "$pg_db" -f "$schema_file" &>/dev/null; then
                    success "Schema applied"
                else
                    warn "Schema application had issues. You may need to run it manually."
                fi
            fi
        else
            warn "Could not download schema. Run it manually later:"
            warn "  curl -fsSL ${SCHEMA_URL} | psql -h ${pg_host} -p ${pg_port} -U ${pg_user} -d ${pg_db}"
        fi
        rm -f "$schema_file"
    else
        warn "psql not found — cannot test connection or apply schema."
        warn "After setup, apply the schema manually:"
        warn "  curl -fsSL ${SCHEMA_URL} | psql -h ${pg_host} -p ${pg_port} -U ${pg_user} -d ${pg_db}"
    fi

    write_db_yaml "$pg_host" "$pg_port" "$pg_user" "$pg_password"
}

skip_pg_setup() {
    echo ""
    info "Skipping PostgreSQL setup."
    echo ""
    echo "  To set up PostgreSQL later, you have two options:"
    echo ""
    echo -e "  ${CYAN}Option A: Docker (recommended)${RESET}"
    echo "    Re-run this script and choose the Docker option, or see:"
    echo "    https://github.com/${GITHUB_REPO}#postgresql-setup"
    echo ""
    echo -e "  ${CYAN}Option B: Existing server${RESET}"
    echo "    Ensure your server runs PostgreSQL 18 with VectorChord suite."
    echo "    Apply the schema: curl -fsSL ${SCHEMA_URL} | psql -h HOST -U USER -d DB"
    echo "    Then edit configs/db.yaml with your connection details."
    echo ""

    # Save default db.yaml content
    write_db_yaml "localhost" "5432" "postgres" "postgres"
}

write_db_yaml() {
    local host="$1" port="$2" user="$3" password="$4"
    # Store for Phase 4 — we write the file after autorag-research init
    DB_YAML_CONTENT="# Database connection configuration
host: ${host}
port: \${oc.env:POSTGRES_PORT,${port}}
user: ${user}
password: \${oc.env:POSTGRES_PASSWORD,${password}} # Recommend not changing and use POSTGRES_PASSWORD environment variables"
}

# ══════════════════════════════════════════════════════════════════════
#  Phase 4: Initialize Configs & Write db.yaml
# ══════════════════════════════════════════════════════════════════════
phase_init_and_configure() {
    header "Finalizing: Initialize Configs"

    info "Running autorag-research init..."
    autorag-research init

    # Overwrite db.yaml with user's actual settings
    if [[ -n "$DB_YAML_CONTENT" ]]; then
        local config_dir="./configs"
        mkdir -p "$config_dir"
        echo "$DB_YAML_CONTENT" > "${config_dir}/db.yaml"
        success "configs/db.yaml updated with your database settings"
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════
print_summary() {
    local workdir
    workdir="$(pwd)"

    echo ""
    echo -e "${BOLD}========================================"
    echo "  AutoRAG-Research Setup Complete!"
    echo -e "========================================${RESET}"
    echo ""
    echo -e "  Working directory:    ${workdir}"
    echo -e "  Configuration:        ${workdir}/configs/"
    if [[ -n "$VENV_PATH" ]]; then
        echo -e "  Virtual environment:  ${VENV_PATH}"
    fi
    if [[ -n "$SELECTED_EXTRAS" ]]; then
        echo -e "  Optional deps:        ${SELECTED_EXTRAS}"
    else
        echo -e "  Optional deps:        none (base only)"
    fi
    echo ""
    echo -e "  ${BOLD}Quick Start:${RESET}"
    if [[ -n "$VENV_PATH" && -d ".venv" ]]; then
        echo "    source .venv/bin/activate"
    fi
    echo "    autorag-research ingest --name=beir --extra dataset-name=scifact"
    echo "    autorag-research run --db-name=beir_scifact_test"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
main() {
    setup_colors

    # Pipe detection (non-interactive allowed via env var)
    if [[ "$NONINTERACTIVE" != "1" ]]; then
        check_not_piped
    fi

    echo ""
    echo -e "${BOLD}${CYAN}AutoRAG-Research Installer${RESET}"
    echo -e "${DIM}Automated setup for RAG research workflows${RESET}"
    echo ""

    phase_python_env
    phase_install_package
    phase_postgresql
    phase_init_and_configure
    print_summary
}

main "$@"
