"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
from l2metrics.__main__ import run

filepath = Path(__file__)


def test_invalid_log_dir():
    # Test the default arguments except for log directory
    args = argparse.Namespace()
    args.log_dir = "does_not_exist"
    args.perf_measure = "reward"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        with pytest.raises(FileNotFoundError):
            run()
        mock_args.assert_called


def test_invalid_perf_measure():
    # Test the default arguments except for log directory
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "reward"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        with pytest.raises(KeyError):
            run()
    mock_args.assert_called


def test_store_single_ste():
    # Test storing single STE run
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ste_logs" / "ste_task1_1_run1"
    )
    args.perf_measure = "reward"
    args.recursive = False
    args.ste_store_mode = "w"
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


def test_store_recursive_ste():
    # Test storing single STE run
    args = argparse.Namespace()
    args.log_dir = filepath.parent.resolve() / ".." / "examples" / "ste_logs"
    args.perf_measure = "reward"
    args.recursive = True
    args.ste_store_mode = "a"
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


def test_default_args():
    # Test the default arguments except for log directory and performace measure
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


def test_recursive():
    # Test the recursive mode for example log directory
    args = argparse.Namespace()
    args.log_dir = filepath.parent.resolve() / ".." / "examples"
    args.perf_measure = "performance"
    args.recursive = True
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("variant_mode", ["aware", "agnostic"])
def test_variant_mode(variant_mode):
    # Test the different arguments for variant mode
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = variant_mode
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("ste_averaging_method", ["metrics", "time"])
def test_ste_averaging_method(ste_averaging_method):
    # Test the different arguments for STE averaging method
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = ste_averaging_method
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("aggregation_method", ["mean", "median"])
def test_aggregation_method(aggregation_method):
    # Test the different arguments for aggregation method
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = aggregation_method
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("maintenance_method", ["mrlep", "mrtlp", "both"])
def test_maintenance_method(maintenance_method):
    # Test the different arguments for maintenance method
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = maintenance_method
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("transfer_method", ["ratio", "contrast", "both"])
def test_transfer_method(transfer_method):
    # Test the different arguments for transfer method
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = transfer_method
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("normalization_method", ["task", "run", "none"])
def test_normalization_method(normalization_method):
    # Test the different arguments for normalization method
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = normalization_method
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize(
    "smoothing_method", ["flat", "hanning", "hamming", "bartlett", "blackman", "none"]
)
def test_smoothing_method(smoothing_method):
    # Test the different arguments for smoothing method
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = smoothing_method
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("smooth_eval_data", [False, True])
def test_smooth_eval_data(smooth_eval_data):
    # Test the different arguments for smooth_eval_data
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = smooth_eval_data
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("window_length", [None, 20])
def test_window_length(window_length):
    # Test the different arguments for window length
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = window_length
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("clamp_outliers", [False, True])
def test_clamp_outliers(clamp_outliers):
    # Test the different arguments for clamping outliers
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = clamp_outliers
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize(
    "data_range_file",
    [None, filepath.parent.resolve() / ".." / "examples" / "data_range.json"],
)
def test_data_range_file(data_range_file):
    # Test the different arguments for data range file
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = data_range_file
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("noise", [[0, 0], [1, 1]])
def test_noise(noise):
    # Test the different arguments for noise
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = noise
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


def test_output():
    # Test output argument
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = "test_output"
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    assert (filepath.parent.resolve() / "results" / "test_output_metrics.json").exists


@pytest.mark.parametrize("unit", ["exp_num", "steps"])
def test_plotting_units(unit):
    # Test the different arguments for plotting units
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = unit
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        if unit == "exp_num":
            run()
        elif unit == "steps":
            with pytest.raises(KeyError):
                run()
    mock_args.assert_called


@pytest.mark.parametrize("show_eval_lines", [False, True])
def test_show_eval_lines(show_eval_lines):
    # Test the different arguments for showing evaluation lines in plots
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = show_eval_lines
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("do_plot", [True, False])
def test_do_plot(do_plot):
    # Test the different arguments for enabling/disabling plotting
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = do_plot
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("plot_type", ["all", "raw", "eb", "lb", "ste"])
def test_plot_types(plot_type):
    # Test the different arguments for plot types
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = plot_type
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("do_save", ["all", "raw", "eb", "lb", "ste"])
def test_do_save(do_save):
    # Test the different arguments for enabling/disabling save
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = do_save
    args.load_settings = None
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize(
    "load_settings",
    [None, filepath.parent.resolve() / ".." / "examples" / "settings.json"],
)
def test_load_settings(load_settings):
    # Test the different arguments for loading settings from JSON
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = load_settings
    args.do_save_settings = True

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called


@pytest.mark.parametrize("do_save_settings", [True, False])
def test_do_save_settings(do_save_settings):
    # Test the different arguments for enabling/disabling save settings
    args = argparse.Namespace()
    args.log_dir = (
        filepath.parent.resolve() / ".." / "examples" / "ll_logs" / "multi_task"
    )
    args.perf_measure = "performance"
    args.recursive = False
    args.ste_store_mode = None
    args.variant_mode = "aware"
    args.ste_averaging_method = "metrics"
    args.aggregation_method = "mean"
    args.maintenance_method = "mrlep"
    args.transfer_method = "ratio"
    args.normalization_method = "task"
    args.smoothing_method = "flat"
    args.smooth_eval_data = False
    args.window_length = None
    args.clamp_outliers = False
    args.data_range_file = None
    args.noise = [0, 0]
    args.output_dir = filepath.parent.resolve() / "results"
    args.output = None
    args.unit = "exp_num"
    args.show_eval_lines = False
    args.do_plot = True
    args.plot_types = "all"
    args.do_save = True
    args.load_settings = None
    args.do_save_settings = do_save_settings

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = args
        run()
    mock_args.assert_called
