# Changelog

All notable changes to this repository are documented here. We are using [Semantic Versioning for Documents](https://semverdoc.org/), in which a version number has the format `major.minor.patch`.

## 3.0.0 - 2022-01-28

- Enforced black formatting
- Created git workflows for linting, building, and running examples
- Updated README
- Updated license year to include 2022
- Removed program-specific evaluation package
- Configured logging for tqdm compatibility

## 2.15.0 - 2022-01-21

- Handled additional edge cases for storing STEs
- Filtered STEs by “complete” experiences before storing
- Modified normalization function to accept constant performance and normalize to 0
- Replaced majority of print statements with logging library
- Updated reference to lifelong learning metrics specification document on arxiv
- Added additional documentation on data preprocessing

## 2.14.0 - 2021-12-07

- Implemented supplemental metrics subpackage for ABC transfer
- Designed task coloring in plots to be consistent across runs when computing recursive metrics or evaluation
- Optimized code by limiting groupby calls
- Refactored episodes to experiences in variable names and plot labels
- Updated example evaluation results

## 2.13.0 - 2021-10-18

- Filtered out wake EBs in plotting methods
- Sanitized metrics JSON parser example notebook
- Minor STE plot appearance changes
- Updated README and copyright statement

## 2.12.0 - 2021-08-17

- Implemented class for parsing metrics JSON files (WIP)
- Added more data to the metrics JSON output including:
  - List of relative performance values with multiple runs of STE
  - List of intermediate results used to calculate sample efficiency
  - List of sample efficiency values with multiple runs of STE
- Added additional output file for regime-level metrics containing terminal performance for each regime

## 2.11.0 - 2021-08-09

- Implemented variant-aware and variant-agnostic metric calculations
- Changed default STE averaging method to metrics
- Changed order of pre-processing to smooth then clamp then normalize
- Included block subtype in regime metrics dataframe
- Handled exception occurring with tasks in STE data without normalization ranges
- Added dashed line on STE plots where performance curves are stitched together
- Added feature to plot all STE runs when averaging with metrics method
- Added feature to calculate runtime from log timestamps
- Updated README and example output files

## 2.10.0 - 2021-07-20

- Added option to evaluate a single agent configuration directory
- Added more data validation for normalization and loading STE data
- Added support for M15 evaluation directory validation
- Implemented flag for enabling/disabling evaluation block data smoothing
- Redirected console output to tqdm write when showing progress bars

## 2.9.0 - 2021-06-29

- Consolidated settings output into one JSON file
- Added normalization range in metrics JSON for each run
- Disabled smoothing for evaluation block data
- Fixed bug with regime and block number mismatch by filtering NaNs before filling regime number
- Implemented using input data range for clamping outliers
- Updated normalization range to 1-101
- Implemented new block plotting visualization
- Created utility function for initializing l2metrics parser
- Fixed bug with numpy array type when aggregating by median
- Removed deprecated command-line argument: --show-rate-data

## 2.8.1 - 2021-05-27

- Updated evaluation package for M12 protocol
- Replaced lifetime aggregation method calls with numpy version to ignore NaNs
- Fixed bug with time-series averaging for multiple STE runs
- Added checks for proper settings before creating output directories
- Updated evaluation block plotting method to show pre-processed data instead of just mean line

## 2.8.0 - 2021-05-11

- Applied recursive option for storing STE data
- Aggregated log data into single feather file for recursive flag
- Output plots for recursive flag
- Add output directory flag to l2metrics
- Updated documentation and parameter descriptions
- Updated metrics calculations for scenarios with wake/sleep blocks
- Fix bug with learning block length of 1 when computing recovery time

## 2.7.1 - 2021-04-26

- Updated reference to metrics specification document 0.68
- Updated default settings for aggregation (mean) and transfer (ratio)
- Fixed bug with setting normalization method to none
- Clamp outliers in STE data as well as LL data

## 2.7.0 - 2021-04-16

- Minor performance optimizations
- Implemented per-task outlier clamping
- Implemented mean method for aggregating lifetime metrics
- Implemented input settings file and save out of settings used to produce metrics
- Handled scenario type in evaluation
- Added validation script to look for required evaluation files
- Simplified command-line arguments
- Output data frames using feather instead of pickle
- Allowed evaluation block data in STE logs
- Stored intermediately processed data in report log data
- Standardized task names when parsing data range file for normalization
- Implemented feature to save settings as JSON file
- Increased verbosity of error message in data range validation
- Handled storing multiple STE runs with option to save in write or append mode
- Implemented different options for averaging STE runs (time or metrics)
- Modified command-line arguments for normalization and smoothing methods
- Implemented option to show lines between evaluation blocks
- Implemented command-line argument for smoothing window length
- Updated example logs
- Implemented recursive flag in l2metrics package

## 2.6.0 - 2021-03-01

- Throw warnings instead of raising exception when checking for alternating blocks
- Print valid application measures in error message when input is invalid
- Implement feature to add Gaussian noise to log data
- Added relative Single-Task Expert plotting
- Modified evaluation script to loop over every agent configuration in the given evaluation directory
- Implemented function for recursively unzipping logs in evaluation directory
- Added fields to output TSV and also report task-level metrics in JSON format
- Modified terminal performance calculation to average over all evaluation block data
- Added functionality for normalization and outlier removal
- Created input parameter for passing in task performance ranges for normalization
- Separated LL metrics into individual modules
- Implemented alternative methods for performance maintenance (MRTLP and MILER)
- Recombined forward and backward transfer into a single transfer metric module
- Created parallel evaluation script with multiprocessing
- Added functionality to plot raw data behind smoothed performance curve
- Updated README and example logs/files

## 2.5.1 - 2021-02-09

- Handled scenarios with tasks that are not all trained

## 2.5.0 - 2021-02-08

- Handled LL logs in evaluation directories not contained within top-level scenario log directories
- Fixed import error in example metric calculation script
- Separated regime metric calculations to class method and store as member variable
- Added Python notebook for calculating single lifetime metrics with additional summaries
- Sort task names by order in which they are trained to make interpreting transfer matrix easier
- Handle NaNs in calculating regime metrics
- Filter log data by completed experiences before filling in regime number to handle misnumbering
- Handle NaNs and empty lists in saturation and terminal performance calculations
- Update evaluation README to describe directory structure assumptions

## 2.4.0 - 2021-01-25

- Added explanation for transfer matrix output in README
- Removed task name simplification for greater compatibility with learnkit and custom task names
- Added computational cost parser in evaluation scripts
- Fixed minor bug when no STE data is stored

## 2.3.1 - 2021-01-11

- Updated reference to metrics specification document 0.66
- Fixed manual path generation bug between Windows and Linux when getting STE task names
- Removed ambiguous imports in l2metrics init
- Removed performance measure filter when loading log data to pass validation
- Fixed dropping of metric values when NaNs were present in the same column
- Converted all task names to lowercase for more robust name comparison
- Fixed handling of NaN task parameters

## 2.3.0 - 2020-12-21

- Added scripts for multi-lifetime evaluation
- Added validation for scenario info (complexity/difficulty)
- Added type hints
- Updated reference to metrics specification document 0.65
- Updated calculation for performance recovery to use negative slope
- Add docstrings
- Minor updates to example multi-task log

## 2.2.0 - 2020-11-19

- Added validation to performance maintenance
- Updated performance maintenance calculation, more specifically maintenance values
- Updated backward transfer to compute multiple values for each task pair
- Updated example logs and documentation with ideal learner scenario
- Implemented option to compute transfer matrix with both contrast and ratio methods

## 2.1.0 - 2020-10-21

- Added log validation
- Reformatted metrics report output
- Updated metrics calculations per metrics specification version 0.64
  - Switched to Theil-Sen slope instead of linear regression for performance recovery
  - Added parameter for selecting transfer calculation method (contrast vs ratio)
  - Use median instead of mean for aggregating lifetime metrics
- Updated log data parsing per log format version 1.0
- Added flag for disabling smoothing
- Added flag for disabling plotting
- Added flag for disabling saving

## 2.0.0 - 2020-09-23

- Updated initial metrics to include all major identified lifelong learning metrics
- Removed dependence on learnkit
- Refactored metrics calculations to integrate with different log format
- Updated workflow for storing single-task expert data
- Updated READMEs and example scripts
- Added new example logs along with their metrics outputs

## 0.3.0 - 2020-03-06

- Updated initial agent and classification metrics for both adapting to new tasks and continual learning
- Updated reporting format, plots for metrics
- Added getting started with custom metrics guide (`examples/README.md`), including example log files (`examples/syllabus_ANT_harder-1582671493-285338/`), and example output plot (`examples/syllabus_ANT_harder-1582671493-285338_example.png`)

## 0.2.0 - 2019-12-05

- Updated foundation code and initial metrics for continual learning
- Overview document of Metrics Framework (`docs/metrics_documentation.pdf`)

## 0.1.0 - 2019-10-20

- Utilities and foundation code for metrics
