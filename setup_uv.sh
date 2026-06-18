#!/bin/bash
# UV Environment Setup Script for lmms-eval
# This script sets up a Python virtual environment using uv and installs dependencies

export UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venv"
PYTHON_VERSION="3.12"

# UV paths on shared storage (CPFS) so that .venv symlinks work across cluster nodes
export UV_PYTHON_INSTALL_DIR="/mnt/cpfs/yangyicun/uv/python"
export UV_CACHE_DIR="/mnt/cpfs/yangyicun/uv/cache"
mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}"

# Print functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install uv first."
        echo "Install options:"
        echo "  - curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  - pip install uv"
        exit 1
    fi
    print_success "uv is installed ($(uv --version))"
}

# Check Python version
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python ${PYTHON_VERSION} or higher."
        exit 1
    fi
    
    local python_version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Found Python ${python_version}"
}

# Ensure uv-managed Python is installed on shared storage
ensure_uv_python() {
    # If currently inside a virtual environment, deactivate first so that
    # uv picks the real system/uv-managed interpreter instead of a venv symlink.
    if [[ "${VIRTUAL_ENV:-}" != "" ]]; then
        print_warning "Detected active virtual environment (${VIRTUAL_ENV}), deactivating..."
        deactivate 2>/dev/null || true
    fi

    print_info "Ensuring CPython ${PYTHON_VERSION} is installed in ${UV_PYTHON_INSTALL_DIR}..."
    uv python install "${PYTHON_VERSION}"
}

# Create virtual environment
create_venv() {
    if [ -d "$VENV_DIR" ]; then
        print_info "Virtual environment already exists at $VENV_DIR, skipping creation."
        return 0
    fi

    print_info "Creating virtual environment with uv..."
    uv venv "$VENV_DIR" --python "${PYTHON_VERSION}"
    print_success "Virtual environment created at $VENV_DIR"
}

# Install dependencies (all optional dependencies included by default)
install_deps() {
    print_info "Installing all dependencies (including optional ones)..."
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    uv pip install -e ".[all]"
    
    # Install latex2sympy2_extended which is required by math_verify but may be missing
    uv pip install latex2sympy2_extended
    
    uv pip install vllm

    uv pip install rouge_score
    
    uv pip install bert_score

    uv pip install rdkit

    uv pip install rdchiral

    uv pip install "modelscope>=1.18.1"

    # Download NLTK wordnet data required by SmolInstruct molecule captioning metric
    # Place it under .venv so it is shared across cluster nodes via CPFS.
    NLTK_DATA_DIR="$(pwd)/.venv/share/nltk_data"
    print_info "Downloading NLTK wordnet data to ${NLTK_DATA_DIR}..."
    if ! python -c "
import ssl, nltk, os
os.makedirs('${NLTK_DATA_DIR}', exist_ok=True)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('wordnet', download_dir='${NLTK_DATA_DIR}', quiet=True)
" 2>/dev/null; then
        print_warning "NLTK download failed (likely SSL/proxy issue). Falling back to manual wget..."
        mkdir -p "${NLTK_DATA_DIR}/corpora"
        wget -c -t 0 --timeout=300 \
            https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip \
            -O "${NLTK_DATA_DIR}/corpora/wordnet.zip"
        (cd "${NLTK_DATA_DIR}/corpora" && unzip -q wordnet.zip && rm wordnet.zip)
    fi
    
    print_success "All dependencies installed"
}

# Install development dependencies
install_dev_deps() {
    print_info "Installing development dependencies..."
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    uv pip install -e ".[dev]"
    
    print_success "Development dependencies installed"
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dev       Install development dependencies"
    echo "  -c, --clean     Clean existing virtual environment before setup"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Setup with all dependencies"
    echo "  $0 -d                 # Setup with all + dev dependencies"
}

# Main function
main() {
    local install_dev=false
    local clean=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dev)
                install_dev=true
                shift
                ;;
            -c|--clean)
                clean=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_info "Starting uv environment setup for lmms-eval..."
    
    # Checks
    check_uv
    check_python

    # Ensure uv-managed Python is on shared storage before creating venv
    ensure_uv_python

    # Clean if requested
    if [ "$clean" = true ] && [ -d "$VENV_DIR" ]; then
        print_info "Cleaning existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi

    # Create virtual environment (skip if already exists)
    create_venv
    
    # Install dependencies only when the virtual environment is newly created
    if [ ! -f "$VENV_DIR/.setup_complete" ]; then
        install_deps
        if [ "$install_dev" = true ]; then
            install_dev_deps
        fi
        touch "$VENV_DIR/.setup_complete"
    else
        print_info "Dependencies already installed, skipping installation."
    fi
    
    # Activate virtual environment in current shell
    print_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    # Print summary
    echo ""
    print_success "Setup complete! 🎉"
    echo ""
    print_success "Virtual environment is now ACTIVE!"
    echo ""
    echo "Python: $(which python)"
    echo "Version: $(python --version)"
    echo ""
    echo "Available commands:"
    echo "  lmms-eval          - Main evaluation CLI"
    echo "  lmms-eval-mcp      - MCP server CLI"
    echo "  lmms-eval-ui       - TUI interface"
    echo ""
    echo "To use uv run (recommended):"
    echo "  uv run python -m lmms_eval --help"
}

# Run main function
main "$@"
