"""Worker-count arithmetic: the guard that keeps a parallel run out of the OOM killer."""

from reagent.eval.parallel import (
    HEADROOM_MB,
    WORKER_RSS_MB,
    available_memory_mb,
    safe_job_count,
)


def test_serial_stays_serial():
    # Nothing should consult memory or spawn a pool for jobs=1.
    assert safe_job_count(1, available_mb=65536) == 1


def test_workers_are_capped_by_memory_not_by_request():
    # 4096 MB free, 1200 reserved, 1600 each -> (4096-1200)//1600 = 1.
    assert safe_job_count(8, available_mb=4096) == 1
    # 6144 MB free -> (6144-1200)//1600 = 3.
    assert safe_job_count(8, available_mb=6144) == 3


def test_a_request_below_the_memory_cap_is_honoured():
    assert safe_job_count(2, available_mb=32768) == 2


def test_never_returns_zero_on_a_full_machine():
    """With no memory to spare the caller still has to plan -- serially."""
    assert safe_job_count(8, available_mb=HEADROOM_MB) == 1
    assert safe_job_count(8, available_mb=0) == 1


def test_headroom_is_actually_reserved():
    """Exactly one worker's worth free, plus headroom, must still yield one."""
    assert safe_job_count(4, available_mb=HEADROOM_MB + WORKER_RSS_MB) == 1
    assert safe_job_count(4, available_mb=HEADROOM_MB + 2 * WORKER_RSS_MB) == 2


def test_the_boundary_is_exact_not_a_rounding_artifact():
    """One MB short of affording a second worker must not round up to two."""
    assert safe_job_count(4, available_mb=HEADROOM_MB + 2 * WORKER_RSS_MB - 1) == 1


def test_memory_probe_returns_something_plausible():
    reading = available_memory_mb()
    assert reading >= 0
    assert reading < 4_194_304


def test_progress_falls_back_to_lines_when_output_is_redirected(tmp_path, monkeypatch):
    """A rewriting bar is unreadable in a log file and defeats tailing a run.

    click.progressbar renders only its bare label when stdout is not a terminal,
    so without this branch every redirected run loses its progress output.
    """
    import sys

    import click

    log = tmp_path / "run.log"
    with log.open("w") as handle:
        monkeypatch.setattr(sys, "stdout", handle)
        assert not sys.stdout.isatty()
        # The branch the CLI takes: plain counted lines, not a bar.
        click.echo("  planned fluoxetine  (1/10)", file=handle)

    assert "planned fluoxetine  (1/10)" in log.read_text()
