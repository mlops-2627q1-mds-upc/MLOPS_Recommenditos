"""Build the web page's self-hosted fonts from their upstream files.

The page (docs/docs/web-ui.md) sets its text in BDO Grotesk and its uppercase
labels, table heads, axis ticks and IDs in a subset of Basically A Mono. Each
woff2 under `recommenditos/api/web/fonts/` is a subset of the upstream font:

- BDO Grotesk, the variable `Fonts/Variable/BDOGrotesk-VF.ttf` of
  github.com/LCTipografi/BDO-Grotesk, with the characters of `UNICODES` (Latin,
  Latin-1, Latin Extended-A for names such as Škoda or Łódź, and the page's
  punctuation, euro and minus signs) and the weight axis cut to 400-500.
- Basically A Mono, the Regular and Medium TTFs of
  experi.mentallydesigned.com/type/bam, with the characters of `MONO_UNICODES`:
  the mono never sets listing text.

Both are under the SIL OFL 1.1, which allows subsets. Two changes go beyond one:

- BDO Grotesk's `tnum` feature makes the comma, full stop and percent sign as
  wide as a figure, which opens a gap in "62,000". `--tabular-digits-only` keeps
  the feature for the figures alone.
- "Basically A Mono" is a Reserved Font Name: a modified version, and a subset
  is one, must not carry it. `--rename "Recommenditos Mono"` renames the family
  in the name table; the copyright and licence records stay.

    uv run python -m tools.web_fonts BDOGrotesk-VF.ttf \\
        recommenditos/api/web/fonts/bdo-grotesk.woff2 --weight 400:500 --tabular-digits-only
    uv run python -m tools.web_fonts BasicallyAMono-Regular.ttf \\
        recommenditos/api/web/fonts/recommenditos-mono-400.woff2 \\
        --mono --rename "Recommenditos Mono"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

#: Every character the page and the listing names need; the woff2 test checks it.
UNICODES = (
    "U+0020-007E,U+00A0-017F,U+2013-2014,U+2018-201A,U+201C-201E,"
    "U+2022,U+2026,U+2033,U+20AC,U+2190,U+2212"
)

#: What the mono sets: uppercase labels, table heads, axis ticks and IDs.
MONO_UNICODES = "U+0020-007E,U+00B7,U+2013,U+2022,U+2026,U+20AC,U+2212"

#: OpenType features kept: kerning and shaping, figure styles, case forms.
FEATURES = ["kern", "liga", "calt", "ccmp", "locl", "mark", "mkmk", "tnum", "pnum", "lnum", "case"]

#: Name records that keep their text under `--rename`: copyright, licence, licence URL.
KEPT_NAME_IDS = {0, 13, 14}


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


def rename(font: TTFont, family: str) -> None:
    """Give the font another family name everywhere but in its copyright and licence."""
    names = font["name"]
    old = names.getDebugName(16) or names.getDebugName(1)
    if not old:
        raise ValueError("the font names no family")
    for record in names.names:
        if record.nameID in KEPT_NAME_IDS:
            continue
        text = record.toUnicode()
        renamed = text.replace(old, family).replace(old.replace(" ", ""), family.replace(" ", ""))
        if renamed != text:
            record.string = renamed


@dataclass(frozen=True)
class Recipe:
    """What a build keeps and changes."""

    weight: tuple[float, float] | None = None
    digits_only: bool = False
    unicodes: str = UNICODES
    family: str | None = None


def build(source: Path, target: Path, recipe: Recipe) -> None:
    font = TTFont(source)
    options = subset.Options()
    options.flavor = "woff2"
    options.layout_features = FEATURES
    options.name_IDs = ["*"]
    options.notdef_outline = True
    subsetter = subset.Subsetter(options)
    subsetter.populate(unicodes=subset.parse_unicodes(recipe.unicodes))
    subsetter.subset(font)
    if recipe.weight is not None and "fvar" in font:
        font = instancer.instantiateVariableFont(font, {"wght": recipe.weight})
    if recipe.digits_only:
        tabular_digits_only(font)
    if recipe.family:
        rename(font, recipe.family)
    font.flavor = "woff2"
    font.save(target)
    print(f"{target}: {target.stat().st_size / 1000:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the upstream TTF")
    parser.add_argument("target", type=Path, help="the woff2 to write")
    parser.add_argument("--weight", help="weight axis range to keep, such as 400:500")
    parser.add_argument("--tabular-digits-only", action="store_true", help="tnum for figures only")
    parser.add_argument("--mono", action="store_true", help="the mono's characters")
    parser.add_argument(
        "--rename", metavar="FAMILY", help="a new family name (Reserved Font Name)"
    )
    args = parser.parse_args()
    weight = None
    if args.weight:
        low, high = args.weight.split(":")
        weight = (float(low), float(high))
    recipe = Recipe(
        weight=weight,
        digits_only=args.tabular_digits_only,
        unicodes=MONO_UNICODES if args.mono else UNICODES,
        family=args.rename,
    )
    build(args.source, args.target, recipe)


if __name__ == "__main__":
    main()
