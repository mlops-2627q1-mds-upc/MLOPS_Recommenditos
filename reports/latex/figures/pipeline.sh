#!/bin/sh
# Regenerate pipeline.pdf, the DVC pipeline figure of the report, from dvc.yaml.
# Run from the repository root after a change to the stages:
#
#   reports/latex/figures/pipeline.sh
#
# Needs the project environment (for `dvc`) and Graphviz (`dot`). The figure is
# committed, so building the report needs neither.
set -eu
cd "$(dirname "$0")/../../.."
uv run dvc dag --dot | dot -Tpdf \
	-Grankdir=LR -Gnodesep=0.06 -Granksep=0.22 -Gmargin=0 -Gpad=0.02 \
	-Nshape=box -Nstyle=rounded -Nfontname=Helvetica -Nfontsize=12 \
	-Nheight=0.24 -Nmargin=0.05,0.02 -Earrowsize=0.5 -Epenwidth=0.7 \
	-o reports/latex/figures/pipeline.pdf
