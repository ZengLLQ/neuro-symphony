"""Unit tests for NeuroSymphony command-line interface."""

from __future__ import annotations

import pytest

from neuro_symphony.cli import main


class TestCLI:
    """Test suite executing CLI entrypoint subcommands."""

    def test_cli_help(self, capsys: pytest.CaptureFixture) -> None:
        """neuro-symphony --help should display usage."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "NeuroSymphony" in captured.out

    def test_cli_no_args(self, capsys: pytest.CaptureFixture) -> None:
        """Invoking with no arguments should print help and return 0."""
        code = main([])
        assert code == 0
        captured = capsys.readouterr()
        assert "Available subcommands" in captured.out or "usage:" in captured.out

    def test_cli_simulate_hh(self, capsys: pytest.CaptureFixture) -> None:
        """simulate-hh subcommand should run without errors."""
        code = main(["simulate-hh", "--duration", "40.0", "--dt", "0.05", "--width", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "HODGKIN-HUXLEY BIOPHYSICAL NEURON" in captured.out

    def test_cli_bgtc_oscillations(self, capsys: pytest.CaptureFixture) -> None:
        """bgtc-oscillations subcommand should execute network simulations."""
        code = main(["bgtc-oscillations", "--duration", "60.0", "--dt", "0.1", "--width", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "BGTC NETWORK: HEALTHY VS PARKINSONIAN" in captured.out

    def test_cli_dbs_control(self, capsys: pytest.CaptureFixture) -> None:
        """dbs-control subcommand should benchmark open-loop vs closed-loop."""
        code = main(["dbs-control", "--duration", "80.0", "--dt", "0.1"])
        assert code == 0
        captured = capsys.readouterr()
        assert "DBS CONTROL BENCHMARK" in captured.out
        assert "Beta Suppression (%)" in captured.out

    def test_cli_demo(self, capsys: pytest.CaptureFixture) -> None:
        """demo subcommand should render cockpit dashboard."""
        code = main(["demo", "--duration", "60.0", "--dt", "0.1", "--dopamine", "0.2", "--width", "60"])
        assert code == 0
        captured = capsys.readouterr()
        assert "NEUROSYMPHONY INTERACTIVE DEMO COCKPIT" in captured.out
