"""`features` stage: turn the split frames into one feature matrix per set.

The stage adds what the split frames do not carry: `age_years` derived from the
reference date, the four equipment lists as multi-hot columns, `weight_kg`
parsed out of its text form (`'1,945 kg'`) and a normalised `model_version`.
This stage has no equivalent in the course demo, whose text model needs no
feature engineering; it is our addition and the report says so.

Every frame is first restricted to the makes the API serves. `split` records
them in `supported_makes.json` and leaves every row in its frames (EDN-48);
this stage is the one place the list is applied (EDN-67), to all five frames -
the `ES` holdout included - and before anything is decided from the rows,
because that is the only point at which the restriction still shapes the
feature space. Applied any later, it would leave the model encoded in make
levels, equipment columns and trims that listings the API answers with a 422
helped choose. `train` and `evaluate` check the property with
`check_served_makes_only` rather than filtering a second time, so the rule has
one owner and its consumers verify it instead of repairing what they are given.

Everything data-dependent about a matrix is then decided by the **served
training rows alone** and written beside the matrices as `feature_space.json`:
which equipment items cleared the frequency threshold, and which levels each
categorical has. Deciding it over all the frames would let validation, test and
the `ES` holdout choose the feature space, which is a leak. Deciding it per
frame would be worse: every split would get different columns and different
category codes, so a model fitted on one could not be applied to another at
all. The API loads the same artefact and calls the functions below, so a served
listing is encoded exactly as a training row was.

Two rules hold throughout, both from EDN-02 and EDN-15.

Categoricals stay categorical: they leave here as `category` columns over the
levels the training rows fixed, never as one-hot columns, because LightGBM and
CatBoost handle them natively.

Missing values are never imputed, because the missingness itself carries
signal. A level the training rows never saw is missing for the same reason: it
carries no fitted signal, and giving it a code of its own would shift the
meaning of every other code between training and serving.
"""

import ast
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata

from loguru import logger
import numpy as np
import pandas as pd
import typer

from recommenditos.config import PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.split_data import SPLIT_NAMES
from recommenditos.pipeline import load_params, read_frame, write_frame
from recommenditos.schema import (
    FEATURE_SPACE_FILE,
    PROCESSED_SCHEMA,
    Column,
    FeatureSpaceError,
    Schema,
    feature_schema,
    load_feature_schema,
    read_feature_space,
)

#: The `ES` holdout gets features too: M6 replays it against the API.
FEATURE_INPUTS: tuple[str, ...] = (*SPLIT_NAMES, "holdout_es")

#: The one split that decides the feature space. Every other frame - including
#: the holdout, and including a request at serving time - is encoded against
#: what this one produced, never against its own contents.
VOCABULARY_SPLIT = "train"

#: The make list `split` writes beside its frames and the API reads out of the
#: model bundle (FR-04).
SUPPORTED_MAKES_FILE = "supported_makes.json"

_DAYS_PER_YEAR = 365.25

#: The trim column, named because three functions have to agree on it: it is
#: the one categorical the stage normalises before counting its levels.
_MODEL_VERSION = "model_version"

#: How many offending values a parse failure names before it stops.
_MAX_REPORTED_VALUES = 5

app = typer.Typer()


def age_years(registration_date: pd.Series, reference_date: str) -> pd.Series:
    """Age in years at `reference_date`, null where the registration date is.

    The one place age is computed. The API calls it with the request date, so
    the feature cannot drift between training and serving (problem-spec 4).
    """
    reference = pd.Timestamp(reference_date)
    return (reference - pd.to_datetime(registration_date)).dt.days / _DAYS_PER_YEAR


#: `weight_kg` as the published file writes it: '894 kg', '1,945 kg'. The comma
#: has to be an English thousands separator in exactly the right place, and there
#: is no decimal branch, because the alternative is worse than a hard failure: on
#: a multilingual German site the likely upstream change is a locale flip, and a
#: looser pattern accepts '194,5 kg' and reads it as 1945 kg, or '1.945 kg' and
#: reads it as 1.945 kg. A value that does not match fails the stage instead,
#: which is the only outcome a reader can tell from a correct parse. All 84,051
#: non-null values of the scoped snapshot match, and none of them needs either
#: branch: no value carries a decimal point, and none has four digits without a
#: separator (measured 2026-09-30).
_WEIGHT_KG = re.compile(r"^\s*([0-9]{1,3}(?:,[0-9]{3})*)\s*kg\s*$")


def weight_kg(values: pd.Series) -> pd.Series:
    """Kerb weight as a number: `'1,945 kg'` becomes `1945.0`.

    The one place the text form is parsed, so a request that carries the
    listing's text rather than a number reaches the model as the same value.
    """
    text = values.astype("str")
    digits = text.str.extract(_WEIGHT_KG, expand=False)
    unparsed = sorted(set(text[values.notna() & digits.isna()]))
    if unparsed:
        shown = ", ".join(repr(value) for value in unparsed[:_MAX_REPORTED_VALUES])
        more = "" if len(unparsed) <= _MAX_REPORTED_VALUES else ", ..."
        raise ValueError(
            f"weight_kg holds {len(unparsed)} value(s) that are not a weight in kg: {shown}{more}"
        )
    return pd.to_numeric(digits.str.replace(",", "", regex=False)).astype("float64")


#: A dot that is not a decimal point inside a number. '2.0 TDI' keeps its dot,
#: because the engine size is the informative half of a trim; the dots in
#: '5p.ti' or in a truncated leading '.2' are punctuation.
_LONE_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")

#: Everything else that separates words in a trim: spaces, '|', '*', '/', '-'.
_TRIM_SEPARATOR = re.compile(r"[^0-9a-z.]+")

_NOT_ALPHANUMERIC = re.compile(r"[^0-9a-z]+")


def model_version(values: pd.Series, *, tokens: int) -> pd.Series:
    """`model_version` cut down to the first `tokens` tokens of its normalised form.

    The raw column is free text, truncated at 50 characters, multilingual and
    stuffed with marketing: 80,332 distinct values over the 105,405 scoped
    listings, which no level set can use. Folding case and accents and
    collapsing the separators leaves 78,738 of them, so normalising the text is
    not enough on its own. Keeping the leading token leaves 2,882, and which of
    those becomes a level is then a frequency question the vocabulary answers.

    What the leading token is worth depends on the make, and the honest summary
    is that it is a coarse starting point rather than the trim (measured
    2026-09-30 on the training split, `reports/analysis/model_version.py`). For
    Audi and Porsche it is usually the body style, so 'avant' or 'coupe' says
    something. For the German premium engine codes, which are the bulk of the
    data, it is the engine letter: 'd', '911' and 'e' are the three most frequent
    levels overall, 12.4 % of the rows lead with an engine letter, and that
    largely duplicates `fuel_category` while discarding the part a buyer cares
    about ('M Sport', 'Touring'). Inside one make and model the merging is
    heavy - Porsche 992 level '911' covers 743 distinct raw trims, Audi A6
    'avant' 473, BMW 320 'd' 331 - and 329 pairs of levels survive where one is a
    prefix of the other, so ('20d', '320d') are different levels for the same
    car. `features.model_version_tokens` and
    `features.model_version_min_frequency` are parameters precisely so the ladder
    can measure whether that coarseness costs anything; the EDN entry records the
    two finer alternatives that were measured and rejected.

    The API calls this too, so a request's trim is normalised exactly as a
    training row's was.
    """
    normalised = values.map(lambda text: _normalise_trim(text, tokens), na_action="ignore")
    # A value of nothing but punctuation normalises to the empty string, which
    # is an absent trim rather than a level of its own.
    return normalised.astype("str").mask(normalised == "", None)


def _normalise_trim(text: str, tokens: int) -> str:
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()
    return " ".join(_TRIM_SEPARATOR.sub(" ", _LONE_DOT.sub(" ", plain)).split()[:tokens])


def equipment_feature_name(source: str, item: str) -> str:
    """The multi-hot column one equipment item becomes.

    Slugged rather than used verbatim because LightGBM rejects a feature name
    holding a JSON character, and `'Alloy wheels (18")'` holds two. Two items
    that slugged to the same name would be a silently merged column, so
    `Schema.extend` refuses the clash and names it; no pair in the published
    snapshot does.
    """
    plain = unicodedata.normalize("NFKD", item).encode("ascii", "ignore").decode()
    return f"{source}_{_NOT_ALPHANUMERIC.sub('_', plain.casefold()).strip('_')}"


#: Text columns this stage parses into a number, with the dtype it leaves them
#: as. Listed once, because the frame and its contract have to agree on both.
_PARSERS: dict[str, tuple[Callable[[pd.Series], pd.Series], str]] = {
    "weight_kg": (weight_kg, "float64"),
}


@dataclass(frozen=True)
class Vocabulary:
    """The part of the feature space the training rows decide.

    `equipment` maps each equipment column to the items that cleared
    `features.equipment_min_frequency`; `categories` maps each categorical
    feature to its levels, in the order their codes will use.
    `model_version_tokens` travels with them because the trim's levels only
    mean anything against the normalisation that produced them.

    The thresholds are deliberately absent: they are parameters, and `dvc.lock`
    is where a run's parameter values are recorded.
    """

    equipment: dict[str, tuple[str, ...]]
    categories: dict[str, tuple[str, ...]]
    model_version_tokens: int
    train_rows: int

    @property
    def n_equipment_features(self) -> int:
        return sum(len(items) for items in self.equipment.values())

    def to_dict(self) -> dict:
        return {
            "train_rows": self.train_rows,
            "model_version_tokens": self.model_version_tokens,
            "equipment": {source: list(items) for source, items in self.equipment.items()},
            "categories": {column: list(levels) for column, levels in self.categories.items()},
        }

    @classmethod
    def from_dict(cls, entry: dict) -> "Vocabulary":
        """A vocabulary from what `to_dict` wrote.

        A field this code does not know is refused rather than ignored, for the
        same reason `Schema.from_dicts` refuses one: both read the same artefact,
        and an unexpected field there means the file was written by a version of
        this stage that recorded something extra, so reading only the fields we
        recognise would silently drop part of the feature space.
        """
        unknown = sorted(set(entry) - _VOCABULARY_FIELDS)
        if unknown:
            raise ValueError(
                f"the vocabulary carries field(s) this code does not know: {', '.join(unknown)}"
            )
        return cls(
            equipment={source: tuple(items) for source, items in entry["equipment"].items()},
            categories={column: tuple(levels) for column, levels in entry["categories"].items()},
            model_version_tokens=entry["model_version_tokens"],
            train_rows=entry["train_rows"],
        )


#: What `Vocabulary.to_dict` writes, which is what `from_dict` accepts.
_VOCABULARY_FIELDS: frozenset[str] = frozenset(
    {"train_rows", "model_version_tokens", "equipment", "categories"}
)


def build_vocabulary(
    frame: pd.DataFrame, feature_columns: "list[str]", feature_params: dict
) -> Vocabulary:
    """The feature space `frame` decides. `frame` is the training split, never another."""
    equipment = {
        column: _frequent_items(frame[column], feature_params["equipment_min_frequency"])
        for column in feature_params["equipment_columns"]
        if column in feature_columns
    }
    tokens = feature_params["model_version_tokens"]
    categories: dict[str, tuple[str, ...]] = {}
    for column in _categorical_columns(feature_columns, encoded=set(equipment)):
        if column == _MODEL_VERSION:
            categories[column] = _frequent_levels(
                model_version(frame[column], tokens=tokens),
                feature_params["model_version_min_frequency"],
            )
        else:
            categories[column] = tuple(sorted(frame[column].dropna().unique()))
    _check_levels(categories, equipment, train_rows=len(frame))
    return Vocabulary(equipment, categories, tokens, len(frame))


def _check_levels(
    categories: dict[str, tuple[str, ...]],
    equipment: dict[str, tuple[str, ...]],
    *,
    train_rows: int,
) -> None:
    """Refuse a categorical the training rows never observed; warn about equipment.

    A categorical with no level is an all-missing column: there is nothing in it
    to fit, and Parquet cannot even carry it - an empty level set comes back as
    text, which then breaks the stage that reads the matrix rather than this one.
    Naming the column here says what to do about it, which is to drop it from
    `features.sets` or to find out why the data lost it.

    An equipment list with no item is milder: the feature set still named it, the
    threshold simply kept nothing, and the matrix is fine without those columns.
    """
    unobserved = sorted(column for column, levels in categories.items() if not levels)
    if unobserved:
        raise ValueError(
            f"no level in the {train_rows:,} {VOCABULARY_SPLIT} rows for: "
            f"{', '.join(unobserved)}. A feature the training rows never observe cannot "
            f"be built; drop it from features.sets or check the data."
        )
    empty = sorted(source for source, items in equipment.items() if not items)
    if empty:
        logger.warning(
            f"no equipment item cleared features.equipment_min_frequency in "
            f"{', '.join(empty)}, so those lists contribute no column."
        )


def _frequent_items(values: pd.Series, min_frequency: float) -> tuple[str, ...]:
    """The items above `min_frequency` of `values`, alphabetically.

    Three properties, each of which a test holds, because each one is a silent
    wrong answer if it breaks rather than a failure.

    The denominator is every row, null lists included, so the threshold is a
    share of the training rows rather than of the rows that happen to list
    something. A repeated item inside one listing counts once, because the
    threshold is a share of listings and not of mentions. And the order is
    alphabetical rather than by frequency, so that two runs on the same data
    produce a byte-identical artefact and identically ordered multi-hot columns
    (NFR-06).
    """
    counts: Counter[str] = Counter()
    for value in values.dropna():
        # A repeated item in one listing is still one listing.
        counts.update(set(_items(value)))
    threshold = min_frequency * len(values)
    return tuple(sorted(item for item, count in counts.items() if count > threshold))


def _frequent_levels(values: pd.Series, min_frequency: float) -> tuple[str, ...]:
    """The values above `min_frequency` of `values`, alphabetically."""
    counts = values.value_counts()
    return tuple(sorted(counts[counts > min_frequency * len(values)].index))


def _items(value: str) -> list[str]:
    """One equipment cell as its items.

    The column is the repr of a Python list, so `literal_eval` reads it exactly.
    A JSON parse would choke on the single quotes, and splitting on ', ' would
    break on an item that contains a comma.
    """
    return ast.literal_eval(value)


def _selection(feature_columns: "list[str]", *, name: str) -> tuple[Schema, tuple[Column, ...]]:
    """The catalogue's view of a feature set, split into features and target.

    Splitting the target off keeps it at the end of the contract while the
    feature block in front of it is reshaped, so a matrix reads left to right as
    features then target however many multi-hot columns appear.
    """
    catalogue = feature_schema(feature_columns, name=name)
    requested = set(feature_columns)
    return (
        catalogue.select(feature_columns, name=name),
        tuple(column for column in catalogue.columns if column.name not in requested),
    )


def _categorical_columns(feature_columns: "list[str]", *, encoded: "set[str]") -> tuple[str, ...]:
    """The selected features that enter the model as levels rather than as numbers.

    Every text column of the catalogue except the ones this stage turns into
    something else: `encoded` holds the equipment lists, which become multi-hot
    columns, and `_PARSERS` the ones that become numbers.
    """
    features, _ = _selection(feature_columns, name="selection")
    return tuple(
        column.name
        for column in features.columns
        if column.dtype == "str" and column.name not in _PARSERS and column.name not in encoded
    )


def matrix_schema(feature_columns: "list[str]", vocabulary: Vocabulary, *, name: str) -> Schema:
    """The contract of one feature matrix, data-dependent columns and levels included.

    Everything the training rows decided is in the returned contract, not only
    which columns exist: a categorical carries the levels its codes are positions
    in, so the stage that reads a matrix, and the API, get them from the artefact
    instead of inferring them from the rows they happen to hold.
    """
    features, targets = _selection(feature_columns, name=name)
    sources = [column for column in feature_columns if column in vocabulary.equipment]
    if sources:
        features = features.drop(sources, name=name).extend(
            _equipment_columns(vocabulary), name=name
        )
    for column, (_, dtype) in _PARSERS.items():
        if column in feature_columns:
            features = features.with_dtype(column, dtype, name=name)
    for column, levels in vocabulary.categories.items():
        features = features.as_category(column, levels, name=name)
    return features.extend(targets, name=name)


def _equipment_columns(vocabulary: Vocabulary) -> tuple[Column, ...]:
    return tuple(
        Column(
            equipment_feature_name(source, item),
            "boolean",
            True,
            f"True where the listing names {item!r} under {source}. Nullable because that "
            f"list is: an empty list is every item False, a null list is null throughout.",
        )
        for source, items in vocabulary.equipment.items()
        for item in items
    )


def encode_equipment(frame: pd.DataFrame, vocabulary: Vocabulary) -> pd.DataFrame:
    """The multi-hot columns of every equipment column in `vocabulary`.

    The API calls this with a one-row frame, so a served listing is encoded by
    this function against this vocabulary, exactly as a training row was.
    """
    encoded = {}
    for source, items in vocabulary.equipment.items():
        positions = {item: index for index, item in enumerate(items)}
        listed = np.zeros((len(frame), len(items)), dtype=bool)
        absent = frame[source].isna().to_numpy()
        for row, value in enumerate(frame[source]):
            if absent[row]:
                continue
            for item in _items(value):
                position = positions.get(item)
                # An item below the threshold, or one a later snapshot added,
                # simply has no column; it is not an error.
                if position is not None:
                    listed[row, position] = True
        for index, item in enumerate(items):
            column = pd.array(listed[:, index], dtype="boolean")
            column[absent] = pd.NA
            encoded[equipment_feature_name(source, item)] = column
    return pd.DataFrame(encoded, index=frame.index)


def build_matrix(
    frame: pd.DataFrame,
    schema: Schema,
    vocabulary: Vocabulary,
    *,
    reference_date: str,
) -> pd.DataFrame:
    """A `processed`-shaped frame as the feature matrix `schema` describes.

    The contract and the vocabulary come from the artefact rather than from
    params.yaml, which is what lets a stage or the API build a matrix without
    knowing which feature set it is working with. `schema.conform` finishes the
    job: it selects the contract's columns in its order, so a column the feature
    set does not name simply never reaches the result, and it casts each
    categorical to the levels the contract declares rather than to the levels
    this particular frame happens to hold.
    """
    matrix = frame.copy()
    matrix["age_years"] = age_years(frame["registration_date"], reference_date)
    for column, (parse, _) in _PARSERS.items():
        if column in schema.names:
            matrix[column] = parse(frame[column])
    if _MODEL_VERSION in vocabulary.categories:
        matrix[_MODEL_VERSION] = model_version(
            frame[_MODEL_VERSION], tokens=vocabulary.model_version_tokens
        )
    return schema.conform(pd.concat([matrix, encode_equipment(frame, vocabulary)], axis=1))


@dataclass(frozen=True)
class FeatureSpace:
    """One feature set's contract and vocabulary, as the `features` stage wrote them.

    The public entry point for everything downstream of this stage: `train`
    (#37), `evaluate` (#39) and the API all hold a directory of matrices and need
    either the contract they were written against or a frame of their own in that
    same shape. Both come from here, so none of them re-derives a level set from
    the rows in front of it - which is the one mistake that produces a model
    fitted on one numbering and scored on another, with no error anywhere.

    - `space.schema` is the contract. `space.schema.conform(frame)` casts any
      frame to it, including the categoricals' declared levels, so `.cat.codes`
      on the result means what it meant during training.
    - `space.schema.feature_names` and `.target_names` are the split a model
      needs; `space.schema.features()` is the contract of a frame that carries no
      label, which is what a request is.
    - `space.matrix(frame, reference_date=...)` turns a `processed`-shaped frame
      into that matrix in one call, which is what the masking sweep of SC-06 and
      a served request both want.
    """

    schema: Schema
    vocabulary: Vocabulary

    @classmethod
    def load(cls, directory: Path, *, name: str) -> "FeatureSpace":
        """Both halves of the artefact in `directory`, read once."""
        space = read_feature_space(directory)
        return cls(
            schema=load_feature_schema(directory, name=name),
            vocabulary=Vocabulary.from_dict(space["vocabulary"]),
        )

    def matrix(self, frame: pd.DataFrame, *, reference_date: str) -> pd.DataFrame:
        """`frame` as the matrix this feature space describes."""
        return build_matrix(frame, self.schema, self.vocabulary, reference_date=reference_date)


def load_vocabulary(directory: Path) -> Vocabulary:
    """The vocabulary `build_features` wrote for the matrices in `directory`.

    For a caller that needs the vocabulary alone - which equipment items got a
    column, which levels a categorical has. A caller that also needs the contract
    wants `FeatureSpace.load`, which reads the file once.
    """
    space = read_feature_space(directory)
    try:
        return Vocabulary.from_dict(space["vocabulary"])
    except (KeyError, TypeError, ValueError) as error:
        raise FeatureSpaceError(
            f"{directory / FEATURE_SPACE_FILE} carries a vocabulary this code cannot read "
            f"({error}). It is the artefact of the `features` stage, so rebuild it with "
            f"`dvc repro features` rather than editing it by hand."
        ) from error


def read_supported_makes(directory: Path) -> tuple[str, ...]:
    """The makes the API serves, as `split` wrote them beside its frames (FR-04).

    Read before the vocabulary is built, because that is the only point at which
    restricting the model to the supported makes still decides the feature
    space: `served_rows` applies the list to every frame first, so `make`'s
    levels are the supported makes the training rows hold, and the equipment and
    trim thresholds are shares of the served rows rather than of every row.
    `train` reads the same file to check its matrices against and to record the
    list in the model bundle, which is where the API takes it from.

    Each entry of the artefact is `{"make": ..., "listings": ...}`, carrying the
    count that admitted it so the file can be audited on its own, so the name has
    to be taken out of the entry. Returning the entries themselves would satisfy
    `len()` and then match nothing in `frame["make"].isin(supported)`, so every
    row of every frame would be removed as unserved.

    An empty list is refused rather than read as "no make is supported": `split`
    raises when no make reaches the threshold, so an empty list here means the
    artefact did not come from `split`.
    """
    path = directory / SUPPORTED_MAKES_FILE
    entries = json.loads(path.read_text(encoding="utf-8"))["supported_makes"]
    if not entries:
        raise FeatureSpaceError(
            f"{path} names no supported make. `split` refuses to write an empty list, so "
            f"either this file did not come from `split` or its threshold was applied to "
            f"an empty frame; a matrix built against it would restrict the model to no "
            f"make at all."
        )
    return tuple(str(entry["make"]) for entry in entries)


def served_rows(frame: pd.DataFrame, supported: "tuple[str, ...]", *, name: str) -> pd.DataFrame:
    """`frame` without the rows of a make the API refuses (FR-04, EDN-48, EDN-67).

    The one place the supported-make list is applied. Called on every frame this
    stage reads, the `ES` holdout included, and on the training split before
    `build_vocabulary`, so no listing the API answers with a 422 is in a fit, a
    metric, the conformal calibration or the M6 replay, and none decides which
    levels or columns the model is encoded in.

    The `ES` holdout is filtered for the same reason the others are rather than
    kept raw for the drift scenario: M6 replays it against the API, and the API
    refuses the unsupported makes before the model sees them, so a raw holdout
    would replay requests that never reach the model and an unfiltered matrix
    would describe traffic the monitored system does not receive. On the real
    snapshot that is 100 of its 6,079 rows, the count EDN-14 already measured
    NFR-11 on.

    A null make is removed too, because `isin` is False for it and because FR-01
    makes `make` a required field, so the API refuses that request as well.

    An empty result fails the stage, whichever frame it is. For the training
    split the vocabulary would otherwise fail on every categorical at once and
    blame the feature set; for any other frame the matrix could not even be read
    back, because Parquet keeps no levels for an empty categorical and the next
    stage's contract check would refuse the file over its levels. In both cases
    the likely cause is a make list from a different `split` run than the
    frames, so that is what the message says.

    `reset_index(drop=True)` so the frame is indexed 0..n-1 whatever was removed.
    Nothing here depends on the index, and leaving gaps in it would make a later
    positional assumption wrong in a way that is invisible until it is not.
    """
    kept = frame[frame["make"].isin(supported)].reset_index(drop=True)
    if kept.empty:
        raise ValueError(
            f"no {name} row is one of the {len(supported)} supported make(s) "
            f"({', '.join(supported)}), so there is nothing to build a matrix from. The make "
            f"list and the frames have to come from the same `split` run."
        )
    removed = len(frame) - len(kept)
    logger.info(
        f"{name}: {len(kept):,} of {len(frame):,} rows are a supported make "
        f"({removed:,} removed, {removed / len(frame):.1%})."
    )
    return kept


def check_served_makes_only(
    matrix: pd.DataFrame, supported: "list[str] | tuple[str, ...]", *, name: str
) -> None:
    """Refuse a feature matrix that holds, or was encoded over, a make the API refuses.

    What `train` and `evaluate` call instead of filtering a second time. On the
    matrices this stage writes a second filter would never remove a row; on any
    other matrices - an older run, or a make list rewritten without rebuilding
    them - it would quietly repair the rows while keeping a feature space decided
    over another population, and the fit or the metric would go ahead with no
    error anywhere. Checking turns that state into a failure that names it, so
    the rule keeps one owner, `served_rows`, and its consumers verify a property
    rather than trusting an intention.

    Two things are checked, because each catches what the other cannot see.

    - The `make` levels are the vocabulary. A level outside the list means the
      feature space was decided over rows the API refuses even when every row
      left is served, which is exactly what these matrices were before EDN-67,
      when `train` filtered its rows and the encoding kept 25 levels.
    - The rows are the population. A row whose make is not in the list is a
      request the API answers with a 422. A row with no make level at all is
      refused too: `make` is a required field, so after `served_rows` a missing
      level can only be an unserved make the vocabulary does not know, or a
      supported make that no training row has, and neither belongs in a fit or
      in a metric about the served population.
    """
    allowed = set(supported)
    make = matrix["make"]
    problems = []

    foreign_levels = sorted(set(map(str, make.cat.categories)) - allowed)
    if foreign_levels:
        problems.append(
            f"its make levels include {', '.join(foreign_levels)}, so its feature space was "
            f"decided over rows the API refuses"
        )
    outside = ~make.isin(allowed)
    if outside.any():
        named = sorted(set(make[outside].dropna().astype(str)))
        unlevelled = int(make[outside].isna().sum())
        detail = [f"makes {', '.join(named)}"] if named else []
        if unlevelled:
            detail.append(
                f"{unlevelled:,} with no make level, which is an unserved make the vocabulary "
                f"does not know or a supported make no training row has"
            )
        problems.append(
            f"{int(outside.sum()):,} of {len(matrix):,} row(s) are not one of the "
            f"{len(supported)} supported make(s) ({'; '.join(detail)})"
        )

    if problems:
        raise ValueError(
            f"{name}: " + "; ".join(problems) + f". The supported makes are "
            f"{', '.join(supported)}. `features` restricts every frame to them before it "
            f"builds anything (EDN-67), so this matrix and the make list come from different "
            f"runs; rebuild the matrices with `dvc repro features` rather than filtering here."
        )


def unobserved_columns(matrix: pd.DataFrame) -> set[str]:
    """The columns of `matrix` that hold no value at all."""
    return {column for column in matrix.columns if not matrix[column].notna().any()}


def _warn_about_unobserved_columns(split: str, columns: "list[str]") -> None:
    """Name the columns a written split fills in no row at all.

    One column does this on the real snapshot and it is intended: the `ES`
    holdout's `country_code` is empty, because `ES` is held out by construction
    (EDN-03), so it is not a training level, and a value outside the training
    levels becomes missing rather than a level of its own - which is exactly the
    treatment EDN-18 requires of an unseen value. The holdout is the new-market
    scenario, and a model that had seen that market would not be one.

    The warning exists because nothing else says so. A fully empty column in a
    matrix reads as a defect in this stage, a pull-request description does not
    survive the squash merge, and M6 replays this holdout months from now.
    """
    if columns:
        logger.warning(
            f"{split}: no value at all in {', '.join(columns)}, although the "
            f"{VOCABULARY_SPLIT} rows fill it. Expected where the split is defined by that "
            f"column - the `ES` holdout has no country_code, because `ES` is not a "
            f"{VOCABULARY_SPLIT} level (EDN-03, EDN-18) - and a defect anywhere else."
        )


def _warn_about_supported_makes_without_training_rows(
    vocabulary: Vocabulary, supported: "tuple[str, ...]"
) -> None:
    """Name a supported make the training rows do not hold, if there is one.

    `split` counts support over train, validation, calibration and test
    together, so a make can clear EDN-05's threshold with every one of its
    sellers outside the training split. The API would then accept that make
    (FR-04) while the model encodes it as missing, and `check_served_makes_only`
    refuses its validation and test rows as unlevelled. It does not happen on the
    real snapshot, where all 11 supported makes are training levels; the warning
    exists so that the state is named here rather than discovered there.
    """
    if "make" not in vocabulary.categories:
        return
    absent = sorted(set(supported) - set(vocabulary.categories["make"]))
    if absent:
        logger.warning(
            f"supported make(s) {', '.join(absent)} have no {VOCABULARY_SPLIT} row, so the API "
            f"accepts them while the model encodes them as missing; `train` and `evaluate` "
            f"will refuse their rows."
        )


def _write_feature_space(directory: Path, schema: Schema, vocabulary: Vocabulary) -> None:
    """The contract and the vocabulary of one feature set, as one artefact.

    It lives beside the matrices, inside the directory the `features` stage
    already declares as an output, so `dvc repro` versions it with them and no
    matrix can be restored without the vocabulary that explains it. `train`
    copies it into the model bundle, which is how it reaches the API - the same
    route the supported-make list takes.
    """
    directory.mkdir(parents=True, exist_ok=True)
    space = {"schema": schema.to_dicts(), "vocabulary": vocabulary.to_dict()}
    path = directory / FEATURE_SPACE_FILE
    path.write_text(json.dumps(space, indent=2) + "\n", encoding="utf-8")
    logger.success(f"Wrote {path}.")


@app.command()
def main(
    feature_set: str = typer.Argument(..., help="a key of features.sets in params.yaml"),
    input_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR / "features",
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    feature_params = params["features"]
    columns = feature_params["sets"][feature_set]
    destination = output_dir / feature_set

    supported = read_supported_makes(input_dir)
    # Restricted before the vocabulary, because that is the only point at which
    # the restriction still decides the feature space (EDN-48, EDN-67).
    training = served_rows(
        read_frame(input_dir / f"{VOCABULARY_SPLIT}.parquet", PROCESSED_SCHEMA),
        supported,
        name=VOCABULARY_SPLIT,
    )
    vocabulary = build_vocabulary(training, columns, feature_params)
    schema = matrix_schema(columns, vocabulary, name=f"features-{feature_set}")
    _write_feature_space(destination, schema, vocabulary)
    logger.info(
        f"{feature_set}: {len(schema.names)} columns, {vocabulary.n_equipment_features} of them "
        f"equipment, decided by the {vocabulary.train_rows:,} served {VOCABULARY_SPLIT} rows. "
        f"{len(vocabulary.categories.get('make', ()))} make level(s) against "
        f"{len(supported)} supported make(s)."
    )
    _warn_about_supported_makes_without_training_rows(vocabulary, supported)

    unobserved_in_training: set[str] = set()
    for name in FEATURE_INPUTS:
        frame = served_rows(
            read_frame(input_dir / f"{name}.parquet", PROCESSED_SCHEMA), supported, name=name
        )
        matrix = build_matrix(frame, schema, vocabulary, reference_date=params["reference_date"])
        write_frame(matrix, destination / f"{name}.parquet", schema)
        # Compared against the training split rather than reported outright: a
        # column no split fills is a question for the feature set, and it is
        # already refused for a categorical, while a column only *this* split
        # leaves empty is the interesting case (`country_code` in the holdout).
        empty = unobserved_columns(matrix)
        if name == VOCABULARY_SPLIT:
            unobserved_in_training = empty
        _warn_about_unobserved_columns(name, sorted(empty - unobserved_in_training))


if __name__ == "__main__":
    app()
