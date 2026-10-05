"""The energy measurement of a fit, at its own seam (issue #38, NFR-10).

`recommenditos.modeling.energy.measure` is what `train` wraps `fit_variant` in.
These tests measure a stand-in callable rather than a model, because what is
under test is the measurement: which settings it obeys, which it refuses, that
it makes no network call, and that a fit CodeCarbon failed to record fails
loudly instead of passing with no figure. The integration with `train` - that a
training run logs the figures to MLflow and that only the fit is inside the
tracker - is in `tests/test_model.py`, which has the trained fixture.

Nothing here asserts an energy value. On a laptop and on a CI runner alike the
figure is CodeCarbon's estimate from the CPU's TDP, which differs per machine by
design; what is asserted is the arithmetic the figure is made of and the record
it is written to.
"""

import socket
import time

from codecarbon.input import DataSource
import pandas as pd
import pytest

from recommenditos.modeling.energy import (
    RECORD_COLUMNS,
    SETTINGS_KEYS,
    EnergyError,
    EnergySettings,
    measure,
    read_record,
)

SETTINGS = {"tracking_mode": "process", "country_iso_code": "ESP", "measure_power_secs": 15}

#: The CPU power methods CodeCarbon 3 implements. `cpu_load` is the one a Linux
#: machine without root, and every CI runner, ends up with.
KNOWN_METHODS = {
    "cpu_load",
    "constant",
    "intel_rapl",
    "intel_power_gadget",
    "windows_emi",
    "apple_powermetrics",
}


@pytest.fixture
def settings() -> EnergySettings:
    return EnergySettings.from_params(SETTINGS)


@pytest.fixture
def no_network(monkeypatch) -> list:
    """Every connection and name lookup in the process fails, and is recorded."""
    attempts: list = []

    def connect(self, address):
        attempts.append(("connect", address))
        raise OSError("the test suite makes no network call")

    def getaddrinfo(host, *args, **kwargs):
        attempts.append(("getaddrinfo", host))
        raise OSError("the test suite makes no network call")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return attempts


def _busy(cpu_seconds: float) -> int:
    """Compute until this process has used `cpu_seconds` of CPU time.

    A CPU-time deadline rather than a wall-clock one, so that the work done does
    not depend on how many other processes the machine is running.
    """
    end, count = time.process_time() + cpu_seconds, 0
    while time.process_time() < end:
        count += 1
    return count


def test_measure_returns_the_result_and_a_record_of_exactly_that_call(settings, tmp_path):
    output = tmp_path / "emissions" / "a-variant.csv"

    result, measured = measure(
        lambda: "the model", name="a-variant", settings=settings, output_file=output
    )

    assert result == "the model"
    row = pd.read_csv(output, dtype={"run_id": "str"})
    assert len(row) == 1
    assert row["run_id"].iloc[0] == measured.run_id
    assert row["project_name"].iloc[0] == "a-variant"
    # What MLflow receives is the row as written, in CodeCarbon's units.
    assert measured.metrics()["energy_kwh"] == row["energy_consumed"].iloc[0]
    assert measured.metrics()["emissions_kg_co2eq"] == row["emissions"].iloc[0]


def test_the_two_times_bracket_the_call_and_tell_waiting_from_working(settings, tmp_path):
    # `fit_cpu_seconds` is the measurement the CPU half of CodeCarbon's estimate
    # is proportional to, so it has to see the difference between a call that
    # waits and one that computes, which wall time alone does not.
    _, waiting = measure(
        lambda: time.sleep(0.3), name="wait", settings=settings, output_file=tmp_path / "wait.csv"
    )
    _, working = measure(
        lambda: _busy(0.3), name="work", settings=settings, output_file=tmp_path / "work.csv"
    )

    assert waiting.fit_seconds >= 0.3
    assert waiting.fit_cpu_seconds < 0.1
    assert working.fit_cpu_seconds >= 0.3
    # CodeCarbon's own duration also covers its start and stop, never less.
    assert waiting.record["duration"] >= waiting.fit_seconds


def test_the_figure_is_energy_times_the_pinned_grid_factor(settings, tmp_path):
    # Emissions are energy times one carbon intensity, the pinned country's, on
    # every machine. The online tracker would pick the factor by geolocation.
    _, measured = measure(
        lambda: _busy(0.2), name="grid", settings=settings, output_file=tmp_path / "grid.csv"
    )

    factor = DataSource().get_global_energy_mix_data()["ESP"]["carbon_intensity"] / 1000
    metrics = measured.metrics()
    assert metrics["emissions_kg_co2eq"] == pytest.approx(metrics["energy_kwh"] * factor)
    assert metrics["energy_kwh"] == pytest.approx(
        metrics["cpu_energy_kwh"] + metrics["ram_energy_kwh"]
    )
    assert measured.record["country_iso_code"] == "ESP"


def test_the_method_behind_the_figure_is_recorded_with_it(settings, tmp_path):
    # An estimate from a TDP and a counter reading are not comparable, so a run
    # has to say which it is. A CodeCarbon release that moves the private
    # attribute this is read from fails here rather than logging `unknown`.
    _, measured = measure(
        lambda: None, name="method", settings=settings, output_file=tmp_path / "method.csv"
    )

    params = measured.params()
    assert params["codecarbon.cpu_power_method"].split(" ")[0] in KNOWN_METHODS
    assert measured.record["tracking_mode"] == "process"
    if params["codecarbon.cpu_power_method"].startswith("cpu_load"):
        assert params["codecarbon.cpu_tdp_w"] > 0


def test_a_contributors_codecarbon_configuration_cannot_change_the_measurement(
    settings, tmp_path, monkeypatch, no_network
):
    # CodeCarbon reads `.codecarbon.config` in the working directory and in the
    # home directory, and every `CODECARBON_*` variable, for each argument it is
    # not given. Each line below would change the figure or reach the network if
    # it were obeyed: the API and HTTP outputs and an Electricity Maps token are
    # network calls, the rest rescale the number.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".codecarbon.config").write_text(
        "[codecarbon]\nsave_to_api = True\npue = 3\n", encoding="utf-8"
    )
    for name, value in {
        "CODECARBON_OUTPUT_METHODS": "csv,api,prometheus",
        "CODECARBON_EMISSIONS_ENDPOINT": "http://127.0.0.1:1/",
        "CODECARBON_ELECTRICITYMAPS_API_TOKEN": "a-token",
        "CODECARBON_FORCE_CPU_POWER": "1000",
        "CODECARBON_FORCE_RAM_POWER": "1000",
        "CODECARBON_FORCE_CARBON_INTENSITY_G_CO2E_KWH": "9999",
        "CODECARBON_COUNTRY_ISO_CODE": "USA",
    }.items():
        monkeypatch.setenv(name, value)

    _, measured = measure(
        lambda: _busy(0.2),
        name="configured",
        settings=settings,
        output_file=tmp_path / "configured.csv",
    )

    assert no_network == []
    metrics = measured.metrics()
    factor = DataSource().get_global_energy_mix_data()["ESP"]["carbon_intensity"] / 1000
    assert metrics["emissions_kg_co2eq"] == pytest.approx(metrics["energy_kwh"] * factor)
    assert metrics["ram_power_w"] < 1000
    assert measured.record["country_iso_code"] == "ESP"
    assert not measured.params()["codecarbon.cpu_power_method"].startswith("User")
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ".codecarbon.config",
        "configured.csv",
    ]


def test_measuring_makes_no_network_call(settings, tmp_path, no_network):
    measure(lambda: None, name="offline", settings=settings, output_file=tmp_path / "o.csv")

    assert no_network == []


def test_appending_keeps_every_fit_and_each_is_found_by_its_run_id(settings, tmp_path):
    # Outside `dvc repro`, which deletes the file first, CodeCarbon appends. The
    # row of a fit is therefore found by its run id: the demo's "last row" would
    # be some other fit's as soon as two processes share the file.
    output = tmp_path / "appended.csv"
    _, first = measure(lambda: None, name="first", settings=settings, output_file=output)
    _, second = measure(lambda: None, name="second", settings=settings, output_file=output)

    assert len(pd.read_csv(output)) == 2
    assert first.run_id != second.run_id
    assert read_record(output, first.run_id)["project_name"] == "first"
    with pytest.raises(EnergyError, match="0 row"):
        read_record(output, "a-run-nobody-measured")


def test_a_fit_codecarbon_failed_to_record_fails_loudly(settings, tmp_path, monkeypatch):
    # CodeCarbon catches every exception in `stop` and logs a warning, so a full
    # disk would otherwise leave a trained model with no figure and a green stage.
    def full_disk(self, total, delta):
        raise OSError("No space left on device")

    monkeypatch.setattr("codecarbon.output_methods.file.FileOutput.out", full_disk)

    with pytest.raises(EnergyError, match="wrote no"):
        measure(lambda: None, name="lost", settings=settings, output_file=tmp_path / "lost.csv")


def test_a_record_in_a_format_this_module_does_not_know_is_refused(tmp_path):
    # What a CodeCarbon release that renamed a column would write. The demo reads
    # the row positionally and would log whatever moved into that position.
    output = tmp_path / "renamed.csv"
    frame = pd.DataFrame([{column: 1 for column in RECORD_COLUMNS if column != "cpu_energy"}])
    frame.rename(columns={"ram_energy": "ram_energy_kwh"}).to_csv(output, index=False)

    with pytest.raises(EnergyError, match="cpu_energy, ram_energy"):
        read_record(output, "1")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"country_iso_code": "XXX"}, "XXX"),
        ({"country_iso_code": "ES"}, "three-letter"),
        ({"tracking_mode": "everything"}, "everything"),
        ({"measure_power_secs": 0}, "positive"),
        ({"output_dir": "somewhere"}, "output_dir, which nothing reads"),
    ],
)
def test_a_setting_codecarbon_would_silently_misread_is_refused(change, message):
    # An unknown country is measured against the world average with one log line,
    # and an unknown keyword leaves a half-built tracker, because CodeCarbon
    # swallows its own exceptions. Both are refused before anything is fitted.
    with pytest.raises(EnergyError, match=message):
        EnergySettings.from_params({**SETTINGS, **change})


def test_a_missing_setting_is_named(params):
    block = dict(params["train"]["energy"])
    block.pop("country_iso_code")

    with pytest.raises(EnergyError, match="does not set country_iso_code"):
        EnergySettings.from_params(block)


def test_the_project_settings_are_valid_and_complete(params):
    # The real params.yaml, so a typo there fails here rather than in a run.
    assert set(params["train"]["energy"]) == set(SETTINGS_KEYS)
    EnergySettings.from_params(params["train"]["energy"])
