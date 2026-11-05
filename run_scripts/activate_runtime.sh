#!/usr/bin/env bash
# Accept either a virtualenv or the shared Conda lmm prefix.
activate_eval_runtime() {
    local prefix="$1"
    if [[ ! -x "${prefix}/bin/python" ]]; then
        echo "[ERROR] Evaluation Python is missing: ${prefix}/bin/python" >&2
        return 1
    fi
    if [[ -f "${prefix}/bin/activate" ]]; then
        source "${prefix}/bin/activate"
    elif [[ -d "${prefix}/conda-meta" ]]; then
        # This prefix was relocated from the old mount. Its conda.sh and
        # console-script shebangs still contain that old absolute path.
        export PATH="${prefix}/bin:${PATH}"
        export CONDA_PREFIX="${prefix}"
        export CONDA_DEFAULT_ENV="${prefix##*/}"
        unset PYTHONHOME
    else
        echo "[ERROR] Expected a virtualenv or Conda environment: ${prefix}" >&2
        return 1
    fi
}
