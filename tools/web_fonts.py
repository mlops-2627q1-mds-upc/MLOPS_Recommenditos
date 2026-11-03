"""Build the web page's text face, Schibsted Grotesk, from its upstream variable TTF.

The page (docs/docs/web-ui.md) sets everything in Schibsted Grotesk; its axis
ticks and IDs use JetBrains Mono, the static Latin-only subset the first design
shipped, which this tool does not build. `schibsted-grotesk.woff2` under
`recommenditos/api/web/fonts/` is a subset of the variable font in google/fonts: the characters of
`UNICODES` (Latin, Latin-1, Latin Extended-A for names such as Škoda or Łódź, and
the page's punctuation, euro and minus signs), the weight axis cut to the weights
the page uses, and only the OpenType features it needs.

One change to the font itself: Schibsted Grotesk's `tnum` feature makes the
comma, full stop, colon and percent sign as wide as a figure, which opens a gap
in "62,000" and "€24,593". `--tabular-digits-only` keeps the feature for the
figures alone. The licence (SIL OFL 1.1, no Reserved Font Name) allows both.

    uv run python -m tools.web_fonts "SchibstedGrotesk[wght].ttf" \\
        recommenditos/api/web/fonts/schibsted-grotesk.woff2 --weight 400:700 --tabular-digits-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

#: Every character the page and the listing names need; the woff2 test checks it.
UNICODES = (
    "U+0020-007E,U+00A0-017F,U+2013-2014,U+2018-201A,U+201C-201E,"
    "U+2022,U+2026,U+2033,U+20AC,U+2190,U+2212"
)

#: OpenType features kept: kerning and shaping, figure styles, case forms.
FEATURES = ["kern", "liga", "calt", "ccmp", "locl", "mark", "mkmk", "tnum", "pnum", "lnum", "case"]


def tabular_digits_only(font: TTFont) -> int:
    """Drop every glyph but the figures from the `tnum` substitutions; returns how many."""
    figures = {glyph for code, glyph in font.getBestCmap().items() if chr(code).isdigit()}
    gsub = font["GSUB"].table
    dropped = 0
    for record in gsub.FeatureList.FeatureRecord:
        if record.FeatureTag != "tnum":
            continue
        for index in record.Feature.LookupListIndex:
            for wrapped in gsub.LookupList.Lookup[index].SubTable:
                mapping = getattr(getattr(wrapped, "ExtSubTable", wrapped), "mapping", None)
                if mapping is None:
                    continue
                for glyph in [g for g in mapping if g not in figures]:
                    del mapping[glyph]
                    dropped += 1
    return dropped


def build(
    source: Path, target: Path, weight: tuple[float, float] | None, digits_only: bool
) -> None:
    font = TTFont(source)
    options = subset.Options()
    options.flavor = "woff2"
    options.layout_features = FEATURES
    options.name_IDs = ["*"]
    options.notdef_outline = True
    subsetter = subset.Subsetter(options)
    subsetter.populate(unicodes=subset.parse_unicodes(UNICODES))
    subsetter.subset(font)
    if weight is not None and "fvar" in font:
        font = instancer.instantiateVariableFont(font, {"wght": weight})
    if digits_only:
        tabular_digits_only(font)
    font.flavor = "woff2"
    font.save(target)
    print(f"{target}: {target.stat().st_size / 1000:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the upstream TTF")
    parser.add_argument("target", type=Path, help="the woff2 to write")
    parser.add_argument("--weight", help="weight axis range to keep, such as 400:700")
    parser.add_argument(
        "--tabular-digits-only", action="store_true", help="keep tnum for the figures alone"
    )
    args = parser.parse_args()
    weight = tuple(float(w) for w in args.weight.split(":")) if args.weight else None
    build(args.source, args.target, weight, args.tabular_digits_only)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
