# Changelog

All notable changes to this repository are documented here. We are using [Semantic Versioning for Documents](https://semverdoc.org/), in which a version number has the format `major.minor.patch`.

## 2.5.0 - 2021-02-07

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
