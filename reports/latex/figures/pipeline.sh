#!/bin/sh
# Regenerate pipeline.pdf, the DVC pipeline figure of the report, from dvc.yaml.
# Run from anywhere after a change to the stages:
#
#   reports/latex/figures/pipeline.sh
#
# Needs the project environment (for `dvc`), Graphviz (`dot`) and the DejaVu Sans
# font. The figure is committed with the font embedded, so building the report
# needs none of them.
set -eu
cd "$(dirname "$0")/../../.."

# Graphviz silently substitutes a font it cannot find, and a name such as
# Helvetica resolves to a different face on every system, so the figure would
# change with the machine that drew it. The figure is pinned to DejaVu Sans, a
# free font and the fontconfig default on most Linux systems, and this refuses
# to draw it with anything else.
font="DejaVu Sans"
if [ "$(fc-match -f '%{family}' "$font")" != "$font" ]; then
	echo "pipeline.sh: the font '$font' is not installed (fc-match resolves it to" \
		"'$(fc-match -f '%{family}' "$font")'); install it, e.g. fonts-dejavu-core." >&2
	exit 1
fi

uv run dvc dag --dot | dot -Tpdf \
	-Grankdir=LR -Gnodesep=0.06 -Granksep=0.22 -Gmargin=0 -Gpad=0.02 \
	-Nshape=box -Nstyle=rounded -Nfontname="$font" -Nfontsize=12 \
	-Nheight=0.24 -Nmargin=0.05,0.02 -Earrowsize=0.5 -Epenwidth=0.7 \
	-o reports/latex/figures/pipeline.pdf
