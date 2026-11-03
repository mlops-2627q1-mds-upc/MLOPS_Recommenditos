"""The web page's fonts (#269): their characters, figures, names, licences and weight.

They need fontTools, from the `webfonts` dependency group; like the Node tests of
test_web_script.py, they are skipped on a machine without it but fail in CI.
"""

import os

import pytest

try:
    from fontTools import subset
    from fontTools.ttLib import TTFont
except ImportError:
    if os.environ.get("CI"):
        raise
    pytest.skip("fontTools is not installed (the webfonts group)", allow_module_level=True)

from tools.web_fonts import KEPT_NAME_IDS, UNICODES

from recommenditos.api.webui import WEB_DIR


def test_the_font_licences_ship_with_the_fonts():
    fonts = {path.stem for path in (WEB_DIR / "fonts").glob("*.woff2")}
    licences = {path.stem for path in (WEB_DIR / "fonts").glob("OFL-*.txt")}
    assert fonts == {"bdo-grotesk", "recommenditos-mono-400", "recommenditos-mono-500"}
    assert licences == {"OFL-bdo-grotesk", "OFL-basically-a-mono"}


def test_the_mono_subset_does_not_carry_its_reserved_font_name():
    """The name "Basically A Mono" is reserved: a subset, a Modified Version, must not use it."""
    licence = (WEB_DIR / "fonts" / "OFL-basically-a-mono.txt").read_text(encoding="utf-8")
    assert "Reserved Font Name 'Basically A Mono'" in licence
    for path in (WEB_DIR / "fonts").glob("recommenditos-mono-*.woff2"):
        names = TTFont(path)["name"]
        assert names.getDebugName(1).startswith("Recommenditos Mono")
        carrying = [
            record.nameID
            for record in names.names
            if record.nameID not in KEPT_NAME_IDS and "Basically" in record.toUnicode()
        ]
        assert not carrying, path.name


def test_the_text_face_has_every_character_of_the_page_and_the_listing_names():
    """Listing names carry Škoda, Citroën, Łódź; a missing letter falls back to another face."""
    cmap = TTFont(WEB_DIR / "fonts" / "bdo-grotesk.woff2").getBestCmap()
    # What BDO Grotesk lacks: the Esperanto and breve letters (Ĉ Ĝ Ĥ Ŝ Ŭ Ĕ Ĭ Ŏ), Ĳ, ĸ, ŉ, ſ,
    # µ, the soft hyphen and the double prime. Of these the listing names hold only two
    # double primes (2 of 70,172 names, 2026-11-03), which take the fallback face.
    lacking = set(
        subset.parse_unicodes("U+00AD,U+00B5,U+0108-0109,U+0114-0115,U+011C-011D,U+0124-0125")
    )
    lacking |= set(
        subset.parse_unicodes("U+012C-012D,U+0132-0133,U+0138,U+0149,U+014E-014F,U+015C-015D")
    )
    lacking |= set(subset.parse_unicodes("U+016C-016D,U+017F,U+2033"))
    wanted = set(subset.parse_unicodes(UNICODES)) - lacking
    assert not sorted(chr(code) for code in wanted - cmap.keys())
    # And every character the page's own files print, whatever the tool's list says.
    used = {
        ord(char)
        for name in ("index.html", "app.js", "app.css")
        for char in (WEB_DIR / name).read_text(encoding="utf-8")
        if char.isprintable() and not char.isspace()
    }
    assert not sorted(chr(code) for code in used - cmap.keys())


def test_the_text_face_lines_up_its_figures_without_widening_the_comma():
    """`tnum` is on wherever numbers line up; it must not open a gap in "62,000".

    Upstream, the feature widens the comma too; tools/web_fonts.py limits it to the figures.
    """
    font = TTFont(WEB_DIR / "fonts" / "bdo-grotesk.woff2")
    cmap = font.getBestCmap()
    figures = {cmap[ord(digit)] for digit in "0123456789"}
    gsub = font["GSUB"].table
    substituted: set[str] = set()
    for record in gsub.FeatureList.FeatureRecord:
        if record.FeatureTag == "tnum":
            for index in record.Feature.LookupListIndex:
                for table in gsub.LookupList.Lookup[index].SubTable:
                    substituted |= set(getattr(table, "ExtSubTable", table).mapping)
    assert substituted == figures


def test_the_fonts_weigh_no_more_than_those_of_the_first_design():
    """The first design's three faces were 79.7 KB, sent on every first visit."""
    assert sum(path.stat().st_size for path in (WEB_DIR / "fonts").glob("*.woff2")) <= 79_700
