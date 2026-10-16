#!/bin/sh
# Regenerate pipeline.pdf, the DVC pipeline figure of the report, from dvc.yaml.
# Run from anywhere after a change to the stages:
#
#   reports/latex/figures/pipeline.sh
#
# Needs the project environment (for `dvc`), Graphviz (`dot`, `tred`) and the
# DejaVu Sans font. The figure is committed with the font embedded, so building
# the report needs none of them.
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

# The two gates get a heavy border: `validate-data` stops the pipeline on bad
# data and `evaluate` writes the release verdict. A border, not a colour, so the
# figure reads the same printed in greyscale.
gates='validate-data|evaluate'

# `tred` removes every edge that a longer path already implies. `dvc dag` draws
# each declared dependency, 29 edges for 15 stages, and most of them only repeat
# the order the others give: `split` feeds every `train` stage directly and
# through its `features` stage. What is left is what runs after what.
#
# One step per command and not one pipeline: POSIX sh has no `pipefail`, so in
# a pipeline a failing `dvc dag` or a missing `tred` would go unnoticed and the
# old figure would stay in place as if it had been redrawn.
graph=$(uv run dvc dag --dot)
graph=$(printf '%s\n' "$graph" | sed -E "s/^\"($gates)\";\$/\"\\1\" [penwidth=1.8];/")
graph=$(printf '%s\n' "$graph" | tred)
printf '%s\n' "$graph" | dot -Tpdf \
	-Grankdir=LR -Gnodesep=0.06 -Granksep=0.14 -Gmargin=0 -Gpad=0.02 \
	-Nshape=box -Nstyle=rounded -Nfontname="$font" -Nfontsize=12 \
	-Nheight=0.24 -Nwidth=0 -Nmargin=0.03,0.02 -Earrowsize=0.5 -Epenwidth=0.7 \
	-o reports/latex/figures/pipeline.pdf
