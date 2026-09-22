#!/bin/sh
# Build the report and the EDN into build/. Used by the Makefile, the Docker
# image and CI, so all three produce the same PDFs.
#
#   ./build.sh [draft]   drafts with guidance: report.pdf, report-final.pdf, edn.pdf
#   ./build.sh initial   submission: MLOps_Recommenditos_Initial_report.pdf + EDN
#   ./build.sh final     submission: MLOps_Recommenditos_Final_report.pdf + EDN
set -eu

cd "$(dirname "$0")"

latexmk_run() {
	latexmk -interaction=nonstopmode -halt-on-error "$@"
}

# Print page count and remaining \tbd placeholders of a finished build.
summary() {
	pages=$(sed -n 's/^Output written on .*(\([0-9]*\) pages\{0,1\}.*/\1/p' "build/$1.log")
	tbd=$(grep 'Unfilled placeholder' "build/$1.log" | sort -u | wc -l | tr -d ' ')
	echo ">>> reports/latex/build/$1.pdf: $pages pages, $tbd distinct unfilled placeholders"
}

# Intermediate files from another TeX Live version (e.g. Docker vs. a local
# install) break biber, so start clean whenever the toolchain changes.
toolchain="$(pdflatex --version | head -n 1) / $(biber --version | head -n 1)"
if [ "$(cat build/.toolchain 2>/dev/null)" != "$toolchain" ]; then
	rm -rf build
	mkdir build
	echo "$toolchain" > build/.toolchain
fi

deliverable=${1:-draft}
case "$deliverable" in
	draft)
		latexmk_run report.tex edn.tex
		latexmk_run -usepretex='\def\ReportFinal{}' -jobname=report-final report.tex
		jobs="report report-final edn"
		;;
	initial | final)
		if [ "$deliverable" = initial ]; then
			name=Initial limit=15 pretex='\def\ReportSubmission{}'
		else
			name=Final limit=30 pretex='\def\ReportSubmission{}\def\ReportFinal{}'
		fi
		latexmk_run -usepretex="$pretex" -jobname="MLOps_Recommenditos_${name}_report" report.tex
		latexmk_run -usepretex='\def\ReportSubmission{}' -jobname=MLOps_Recommenditos_EDN edn.tex
		jobs="MLOps_Recommenditos_${name}_report MLOps_Recommenditos_EDN"
		;;
	*)
		echo "Usage: $0 [draft|initial|final]" >&2
		exit 2
		;;
esac

for job in $jobs; do summary "$job"; done
if [ "$deliverable" != draft ]; then
	echo ">>> Page limit for the $deliverable report: $limit pages"
fi

# In the Docker image we run as root; hand the output back to whoever owns
# the mounted folder so it can be edited and deleted without sudo.
if [ "$(id -u)" = 0 ]; then
	chown -R "$(stat -c %u:%g .)" build
fi
