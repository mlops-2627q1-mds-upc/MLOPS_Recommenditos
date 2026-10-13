"""What a trained model does with a car: the three types of behavioural test.

Each class below is one type, so a run names the type of every case it reports.

| Type                          | Class                        |
|-------------------------------|------------------------------|
| Minimum functionality (MFT)   | `TestMinimumFunctionality`   |
| Invariance (INV)              | `TestInvariance`             |
| Directional expectation (DIR) | `TestDirectionalExpectation` |

MFT: a car whose price is known is priced within a factor of 1.5 of it.
INV: a change that must not matter leaves the prediction where it was.
DIR: a change whose effect has a known sign moves the prediction that way.

A model under test comes from one of two sources. `fixture` is the variant
`tests/conftest.py` trains on the synthetic fixture, which every run has, CI
included. `released` is the bundle `dvc pull` restores into `models/`, asked about
the real validation rows; a case on it is skipped where the bundle is absent,
which includes CI. The fixture's price is a formula, so a case on `fixture` shows
that an estimator can learn the behaviour and the same case on `released` that the
model we ship did.

`b0` is in the invariance tests only. Its lookup reads make, model and a two-year
age band, so it has no mileage or power to react to, and what it answers for a
given car is checked by hand in `tests/test_model.py`.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import VARIANTS, project_params, train_variants

from recommenditos.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommenditos.data.synthetic import noise_free_price
from recommenditos.modeling.model import MODEL_FILE, Model, load_model

SOURCES = ("fixture", "released")

#: The variants fitted on the numbers of a car. `b0` is a lookup (see above).
LEARNED_VARIANTS = tuple(
    name
    for name, settings in train_variants().items()
    if settings["estimator"] != "median_baseline"
)


def _released_rows(variant: str) -> Path:
    feature_set = train_variants()[variant]["feature_set"]
    return PROCESSED_DATA_DIR / "features" / feature_set / "validation.parquet"


@cache
def _released(variant: str) -> tuple[Model, pd.DataFrame]:
    return load_model(MODELS_DIR / variant), pd.read_parquet(_released_rows(variant))


@pytest.fixture(scope="session")
def subject(trained: dict, test_frames: dict):
    """`subject(source, variant)`: the model under test and the rows it is asked about.

    The validation rows for `released` rather than the test rows, which only
    `evaluate` reads.
    """

    def _subject(source: str, variant: str) -> tuple[Model, pd.DataFrame]:
        if source == "fixture":
            model = trained["models"][variant]
            return model, test_frames[model.feature_set]
        if (
            not (MODELS_DIR / variant / MODEL_FILE).exists()
            or not _released_rows(variant).exists()
        ):
            pytest.skip("needs the real bundle and matrices, which `dvc pull` restores")
        return _released(variant)

    return _subject


# --------------------------------------------------------------------------
# Minimum functionality (MFT): a car whose price is known
# --------------------------------------------------------------------------

#: How far a prediction may be from a known price, as a factor either way. The
#: median error of these variants is 6 % to 10 %; this is the bound on being
#: plainly wrong about one plain car, which is what a wrong unit, a prediction
#: left in log space or a categorical encoded by the wrong codes produces. The
#: largest deviation measured on 2026-10-13 is a factor of 1.35 on the real
#: bundle (`lgbm-extended` on the BMW 116 below) and of 1.21 on the fixture,
#: over the committed draw and eight other generator seeds.
_KNOWN_PRICE_FACTOR = 1.5

#: The three numbers the fixture's price depends on, at the rounded quartiles of
#: its training rows: the quartile that makes a car dearer for `young`, the
#: medians for `typical` and the quartile that makes it cheaper for `old`.
_FIXTURE_CARS = {
    "young": {"age_years": 6.0, "mileage_km_raw": 90_000.0, "power_kw": 250.0},
    "typical": {"age_years": 12.0, "mileage_km_raw": 130_000.0, "power_kw": 190.0},
    "old": {"age_years": 18.0, "mileage_km_raw": 165_000.0, "power_kw": 145.0},
}

#: Common cars of the real snapshot, one per price range, as the fields FR-01
#: requires and nothing else. The known price is the median price of the
#: comparable listings in the training split: the same make and model, an age
#: within 1.5 years, a mileage within 40 % and a power within 15 %. Read off the
#: matrices of the committed run on 2026-10-13, over 88, 337 and 81 listings.
_REAL_CARS = {
    "bmw-116": (
        {
            "make": "BMW",
            "model": "116",
            "body_type": "Sedan",
            "fuel_category": "Gasoline",
            "transmission": "Manual",
            "country_code": "DE",
            "seller_type": "Dealer",
            "age_years": 9.0,
            "mileage_km_raw": 110_000.0,
            "power_kw": 80.0,
        },
        12_895.0,
    ),
    "bmw-320": (
        {
            "make": "BMW",
            "model": "320",
            "body_type": "Station wagon",
            "fuel_category": "Diesel",
            "transmission": "Automatic",
            "country_code": "DE",
            "seller_type": "Dealer",
            "age_years": 5.0,
            "mileage_km_raw": 90_000.0,
            "power_kw": 140.0,
        },
        27_880.0,
    ),
    "porsche-992": (
        {
            "make": "Porsche",
            "model": "992",
            "body_type": "Coupe",
            "fuel_category": "Gasoline",
            "transmission": "Automatic",
            "country_code": "DE",
            "seller_type": "Dealer",
            "age_years": 2.0,
            "mileage_km_raw": 15_000.0,
            "power_kw": 283.0,
        },
        136_980.0,
    ),
}


def _required_features() -> set[str]:
    """The features a request always carries: FR-01's fields, as the matrix names them."""
    evaluate = project_params()["evaluate"]
    renamed = evaluate["input_field_to_feature"]
    return {renamed.get(field, field) for field in evaluate["required_input_fields"]}


def _smallest_request(model: Model, row: pd.DataFrame) -> pd.DataFrame:
    """`row` with every optional field absent, through `Model.mask_absent`.

    That method is what the seam makes of a field the caller did not send, so
    this is the request FR-01 allows that says the least about a car.
    """
    required = _required_features()
    return model.mask_absent(row, [name for name in model.features if name not in required])


def _assert_priced_within_the_factor(predicted: pd.Series, known: float, case: str) -> None:
    ratio = predicted.iloc[0] / known
    assert 1 / _KNOWN_PRICE_FACTOR < ratio < _KNOWN_PRICE_FACTOR, (
        f"{case}: predicted {predicted.iloc[0]:,.0f} EUR, {ratio:.2f} times the known price of "
        f"{known:,.0f} EUR and outside a factor of {_KNOWN_PRICE_FACTOR}"
    )


class TestMinimumFunctionality:
    """A car whose price is known is priced within a factor of 1.5 of it."""

    @pytest.mark.parametrize("car", list(_FIXTURE_CARS))
    @pytest.mark.parametrize("variant", LEARNED_VARIANTS)
    def test_a_fixture_car_is_priced_near_the_price_the_generator_gives_it(
        self, variant: str, car: str, subject
    ):
        """The generator's own formula is the known price, so nothing here is fitted.

        One car, of the make and model the frame holds most of, because a model
        fitted on a few hundred rows answers a rare make by the draw: asked of
        every row of the frame instead, this test fails under another generator
        seed with no defect in the model.
        """
        model, frame = subject("fixture", variant)
        numbers = _FIXTURE_CARS[car]
        complete = frame.dropna(subset=[name for name in _required_features() if name in frame])
        make, model_name = complete.groupby(["make", "model"], observed=True).size().idxmax()
        row = complete[(complete["make"] == make) & (complete["model"] == model_name)].head(1)

        predicted = model.predict_eur(_smallest_request(model, row.assign(**numbers)))

        known = noise_free_price(
            numbers["age_years"], numbers["mileage_km_raw"], numbers["power_kw"]
        )
        _assert_priced_within_the_factor(predicted, known, f"{variant}, {car}")

    @pytest.mark.parametrize("car", list(_REAL_CARS))
    @pytest.mark.parametrize("variant", LEARNED_VARIANTS)
    def test_a_common_real_car_is_priced_near_what_comparable_listings_cost(
        self, variant: str, car: str, subject
    ):
        """A car the snapshot holds many of, against what such cars are listed for."""
        model, frame = subject("released", variant)
        given, known = _REAL_CARS[car]
        assert set(given) == _required_features()
        # A level the model does not know is encoded as missing (EDN-18), so a
        # misspelt one would price another car and say nothing.
        for name, value in given.items():
            levels = model.input_schema.column(name).levels
            assert levels is None or value in levels, f"{name}: {value!r} is not a level"

        predicted = model.predict_eur(_smallest_request(model, frame.head(1).assign(**given)))

        _assert_priced_within_the_factor(predicted, known, f"{variant}, {car}")


# --------------------------------------------------------------------------
# Invariance (INV): a change that must not matter
# --------------------------------------------------------------------------

#: A change too small to be worth money, as (column, amount, sources): a test
#: drive, and a first registration one day earlier. The day is asked of the
#: fixture only, because the real snapshot's registration dates are a month apart
#: and no real prediction can move. Power is not among them: one kilowatt is the
#: resolution engine variants differ by, and the models price it.
_NEGLIGIBLE_STEPS = {
    "10-km-more": ("mileage_km_raw", 10.0, SOURCES),
    "a-day-older": ("age_years", 1 / 365.25, ("fixture",)),
}

#: A prediction within this share of where it was counts as not moved, and at most
#: `_MOVED_SHARE` of the predictions may be outside it. Not zero, because a tree
#: is a step function: a car standing on a split crosses it with one kilometre,
#: so some predictions move whatever the fit. One percent is a seventh of the
#: main model's median error. Measured on 2026-10-13 over the 8,859 real
#: validation rows, ten kilometres move 2.8 % of the predictions of `lgbm-basic`
#: by more than that and 1.6 % of `lgbm-extended`'s.
_UNMOVED_WITHIN = 0.01
_MOVED_SHARE = 0.05


class TestInvariance:
    """A change that must not matter leaves the prediction where it was."""

    @pytest.mark.parametrize("field", ["vin", "seller_group_id", "price", "log_price"])
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_a_column_that_is_not_a_feature_cannot_move_the_prediction(
        self, variant: str, field: str, subject
    ):
        """An identifier or the label in the caller's frame changes no answer.

        `price` and `log_price` are in every matrix, so a model that read one would be
        scored on its own target. `vin` and `seller_group_id` are in none: the test
        adds them, as a caller's frame taken from the raw file or a split might.
        Bit-identical, and on `fixture` only: this is a property of how the seam
        selects its columns, which no fit can change.
        """
        model, frame = subject("fixture", variant)
        assert field not in model.features
        # Another value in every row: a label that is in the frame is doubled, an
        # identifier that is not is added, unique per row.
        values = (
            frame[field] * 2 if field in frame else [f"{field}-{row}" for row in range(len(frame))]
        )

        np.testing.assert_array_equal(
            model.predict_eur(frame.assign(**{field: values})).to_numpy(),
            model.predict_eur(frame).to_numpy(),
        )

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize(
        ("source", "step"),
        [(source, step) for step, case in _NEGLIGIBLE_STEPS.items() for source in case[2]],
    )
    def test_a_negligible_change_leaves_nearly_every_prediction_where_it_was(
        self, source: str, step: str, variant: str, subject
    ):
        """Applied to every row, `step` moves at most 5 % of the predictions by over 1 %."""
        model, frame = subject(source, variant)
        column, amount, _ = _NEGLIGIBLE_STEPS[step]

        before = model.predict_eur(frame).to_numpy()
        after = model.predict_eur(frame.assign(**{column: frame[column] + amount})).to_numpy()

        moved = np.mean(np.abs(after / before - 1) > _UNMOVED_WITHIN)
        assert moved <= _MOVED_SHARE, (
            f"{variant} ({source}): {step} moves {moved:.1%} of the {len(frame)} predictions "
            f"by more than {_UNMOVED_WITHIN:.0%}"
        )


# --------------------------------------------------------------------------
# Directional expectation (DIR): a change whose effect has a known sign
# --------------------------------------------------------------------------

#: A realistic change whose effect on a price has a known sign, as (column,
#: amount, sign): about a year and a half of driving, a first registration one
#: year earlier, and the next engine up.
_SIGNED_STEPS = {
    "20000-km-more": ("mileage_km_raw", 20_000.0, -1),
    "a-year-older": ("age_years", 1.0, -1),
    "20-kw-more": ("power_kw", 20.0, +1),
}


class TestDirectionalExpectation:
    """A change whose effect has a known sign moves the prediction that way."""

    @pytest.mark.parametrize("step", list(_SIGNED_STEPS))
    @pytest.mark.parametrize("variant", LEARNED_VARIANTS)
    @pytest.mark.parametrize("source", SOURCES)
    def test_a_signed_change_moves_more_predictions_its_way_than_the_other(
        self, source: str, variant: str, step: str, subject
    ):
        """Applied to every row, `step` moves more predictions its way than the other.

        On aggregate, because boosted trees are not monotone and a per-row assertion
        would fail on a model that is right. A row that does not move counts for
        neither side: a step that crosses no split of a tree changes nothing, and how
        many rows that is depends on the fixture's draw, so a statistic that counted
        them as not moving the right way, as a median of the ratios does, could fail
        on a model with no defect. A model that ignores the column still fails, on
        0 > 0.
        """
        model, frame = subject(source, variant)
        column, amount, sign = _SIGNED_STEPS[step]

        before = model.predict_eur(frame).to_numpy()
        after = model.predict_eur(frame.assign(**{column: frame[column] + amount})).to_numpy()

        change = np.sign(after - before)
        its_way, the_other = np.mean(change == sign), np.mean(change == -sign)
        assert its_way > the_other, (
            f"{variant} ({source}): {step} moves {its_way:.1%} of the {len(frame)} predictions "
            f"{'up' if sign > 0 else 'down'} and {the_other:.1%} the other way"
        )
