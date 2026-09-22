# Minimal TeX Live toolchain for the report and the EDN.
# The sources are mounted at run time, so the image only changes when the
# LaTeX package list does. Usage: see README.md in this folder.

# TeX Live 2025 in its frozen final state, so every build installs exactly
# the same package versions.
ARG TL_REPO=https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final

FROM debian:trixie-slim AS texlive
ARG TL_REPO
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl perl xz-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp/install-tl
RUN curl -fsSL "$TL_REPO/install-tl-unx.tar.gz" | tar -xz --strip-components=1
RUN printf '%s\n' \
        'selected_scheme scheme-infraonly' \
        'TEXDIR /opt/texlive' \
        'TEXMFLOCAL /opt/texlive/texmf-local' \
        'TEXMFSYSCONFIG /opt/texlive/texmf-config' \
        'TEXMFSYSVAR /opt/texlive/texmf-var' \
        'instopt_adjustpath 0' \
        'tlpdbopt_autobackup 0' \
        'tlpdbopt_install_docfiles 0' \
        'tlpdbopt_install_srcfiles 0' \
        > texlive.profile \
    && ./install-tl --profile texlive.profile --repository "$TL_REPO" \
    && ln -s "$(ls -d /opt/texlive/bin/*-linux)" /opt/texlive/bin/linux
ENV PATH=/opt/texlive/bin/linux:$PATH
# Packages used by preamble.tex and edn.tex. Add new ones here.
RUN tlmgr install \
        collection-latex latexmk biber biblatex biblatex-ieee \
        babel-english lm microtype xcolor booktabs tools enumitem etoolbox \
        fancyhdr listings tcolorbox pgf environ tikzfill pdfcol listingsutf8 \
        csquotes cleveref url
# We only compile with pdfLaTeX. Drop what it never reads to keep the image
# slim: LuaTeX binaries and formats, font sources and metrics pdfTeX does not
# use, and the tlmgr database (packages are installed above, never at run time).
RUN cd /opt/texlive \
    && rm -rf tlpkg install-tl* texmf-var/web2c/*.log \
        bin/linux/luatex bin/linux/luahbtex texmf-var/web2c/luatex texmf-var/web2c/luahbtex \
        texmf-dist/fonts/afm texmf-dist/fonts/opentype texmf-dist/fonts/source

FROM debian:trixie-slim
# latexmk needs Perl modules beyond the perl-base shipped in the slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends perl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=texlive /opt/texlive /opt/texlive
ENV PATH=/opt/texlive/bin/linux:$PATH \
    TEXMFVAR=/tmp/texmf-var
WORKDIR /report
# Run through sh so a checkout without the executable bit still works.
ENTRYPOINT ["sh", "/report/build.sh"]
CMD ["draft"]
