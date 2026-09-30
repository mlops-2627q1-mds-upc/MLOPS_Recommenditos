"""`features` stage: turn the split frames into one feature matrix per set.

The stage adds what the split frames do not carry: `age_years` derived from the
reference date, the four equipment lists as multi-hot columns, `weight_kg`
parsed out of its text form (`'1,945 kg'`) and a normalised `model_version`.
This stage has no equivalent in the course demo, whose text model needs no
feature engineering; it is our addition and the report says so.

Everything data-dependent about a matrix is decided by the **training rows
alone** and written beside the matrices as `feature_space.json`: which
equipment items cleared the frequency threshold, and which levels each
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
    Schema,
    feature_schema,
)

#: The `ES` holdout gets features too: M6 replays it against the API.
FEATURE_INPUTS: tuple[str, ...] = (*SPLIT_NAMES, "holdout_es")

#: The one split that decides the feature space. Every other frame - including
#: the holdout, and including a request at serving time - is encoded against
#: what this one produced, never against its own contents.
VOCABULARY_SPLIT = "train"

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


#: `weight_kg` as the published file writes it: '894 kg', '1,945 kg'. Every
#: non-null value of the snapshot matches, so one that does not is a change in
#: the upstream format and fails the stage instead of quietly becoming a
#: missing value that no fill-rate expectation would flag.
_WEIGHT_KG = re.compile(r"^\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*kg\s*$")


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
    stuffed with marketing: 82,203 distinct values over the 113,708 scoped
    listings, which no level set can use. Folding case and accents and
    collapsing the separators leaves 80,495 of them, so normalising the text is
    not enough on its own. The first token is the one that names the trim -
    'd', '911', '2.0', 'xdrive', 'avant' - and there are 2,922 of those. Which
    of them becomes a level is then a frequency question, and the vocabulary
    answers it.

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
        return cls(
            equipment={source: tuple(items) for source, items in entry["equipment"].items()},
            categories={column: tuple(levels) for column, levels in entry["categories"].items()},
            model_version_tokens=entry["model_version_tokens"],
            train_rows=entry["train_rows"],
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

    The denominator is every row, null lists included, so the threshold is a
    share of the training rows rather than of the rows that happen to list
    something. Alphabetical rather than by frequency so that two runs on the
    same data produce a byte-identical artefact (NFR-06).
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
    """The contract of one feature matrix, its data-dependent columns included."""
    features, targets = _selection(feature_columns, name=name)
    sources = [column for column in feature_columns if column in vocabulary.equipment]
    if sources:
        features = features.drop(sources, name=name).extend(
            _equipment_columns(vocabulary), name=name
        )
    for column, (_, dtype) in _PARSERS.items():
        if column in feature_columns:
            features = features.with_dtype(column, dtype, name=name)
    for column in vocabulary.categories:
        features = features.with_dtype(column, "category", name=name)
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
    feature_columns: "list[str]",
    vocabulary: Vocabulary,
    *,
    reference_date: str,
) -> pd.DataFrame:
    """One split's feature matrix, before `write_frame` conforms it to the contract.

    Columns the feature set does not name are left in place rather than dropped:
    the contract selects, so only what it names can reach the artefact.
    """
    matrix = frame.copy()
    matrix["age_years"] = age_years(frame["registration_date"], reference_date)
    for column, (parse, _) in _PARSERS.items():
        if column in feature_columns:
            matrix[column] = parse(frame[column])
    if _MODEL_VERSION in vocabulary.categories:
        matrix[_MODEL_VERSION] = model_version(
            frame[_MODEL_VERSION], tokens=vocabulary.model_version_tokens
        )
    for column, levels in vocabulary.categories.items():
        matrix[column] = _as_levels(matrix[column], levels)
    return pd.concat([matrix, encode_equipment(frame, vocabulary)], axis=1)


def _as_levels(values: pd.Series, levels: tuple[str, ...]) -> pd.Series:
    """`values` as a categorical over exactly `levels`, whichever of them occur.

    The levels are the training rows', for every split and for a request alike,
    so a code means the same car wherever the model meets it. A value outside
    them becomes missing: it carries no fitted signal, and the alternative -
    letting each frame add its own levels - renumbers the others and silently
    changes what the model is asked about.
    """
    return values.where(values.isin(levels)).astype(pd.CategoricalDtype(levels))


def load_vocabulary(directory: Path) -> Vocabulary:
    """The vocabulary `build_features` wrote for the matrices in `directory`.

    The API's entry point into this module: with this and the functions above it
    builds a one-row matrix with the same columns, in the same order, over the
    same levels as the matrix the model was fitted on.
    """
    space = json.loads((directory / FEATURE_SPACE_FILE).read_text(encoding="utf-8"))
    return Vocabulary.from_dict(space["vocabulary"])


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

    training = read_frame(input_dir / f"{VOCABULARY_SPLIT}.parquet", PROCESSED_SCHEMA)
    vocabulary = build_vocabulary(training, columns, feature_params)
    schema = matrix_schema(columns, vocabulary, name=f"features-{feature_set}")
    _write_feature_space(destination, schema, vocabulary)
    logger.info(
        f"{feature_set}: {len(schema.names)} columns, {vocabulary.n_equipment_features} of them "
        f"equipment, decided by the {vocabulary.train_rows:,} {VOCABULARY_SPLIT} rows."
    )

    for name in FEATURE_INPUTS:
        frame = read_frame(input_dir / f"{name}.parquet", PROCESSED_SCHEMA)
        matrix = build_matrix(frame, columns, vocabulary, reference_date=params["reference_date"])
        write_frame(matrix, destination / f"{name}.parquet", schema)


if __name__ == "__main__":
    app()
