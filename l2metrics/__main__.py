# (c) 2019 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
# All Rights Reserved. This material may be only be used, modified, or reproduced
# by or for the U.S. Government pursuant to the license rights granted under the
# clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission,
# please contact the Office of Technology Transfer at JHU/APL.

# NO WARRANTY, NO LIABILITY. THIS MATERIAL IS PROVIDED “AS IS.” JHU/APL MAKES NO
# REPRESENTATION OR WARRANTY WITH RESPECT TO THE PERFORMANCE OF THE MATERIALS,
# INCLUDING THEIR SAFETY, EFFECTIVENESS, OR COMMERCIAL VIABILITY, AND DISCLAIMS
# ALL WARRANTIES IN THE MATERIAL, WHETHER EXPRESS OR IMPLIED, INCLUDING (BUT NOT
# LIMITED TO) ANY AND ALL IMPLIED WARRANTIES OF PERFORMANCE, MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF INTELLECTUAL PROPERTY
# OR OTHER THIRD PARTY RIGHTS. ANY USER OF THE MATERIAL ASSUMES THE ENTIRE RISK
# AND LIABILITY FOR USING THE MATERIAL. IN NO EVENT SHALL JHU/APL BE LIABLE TO ANY
# USER OF THE MATERIAL FOR ANY ACTUAL, INDIRECT, CONSEQUENTIAL, SPECIAL OR OTHER
# DAMAGES ARISING FROM THE USE OF, OR INABILITY TO USE, THE MATERIAL, INCLUDING,
# BUT NOT LIMITED TO, ANY DAMAGES FOR LOST PROFITS.

import argparse

import l2metrics


def run():
    parser = argparse.ArgumentParser(description='Run L2Metrics from the command line')

    # We assume that the logs are found in a subdirectory under $L2DATA/logs - this subdirectory must be passed as a
    # parameter in order to locate the logs which will be parsed by this code
    parser.add_argument('-l', '--log_dir', help='Subdirectory under $L2DATA/logs for the log files')

    # Choose application measure to use as performance column
    parser.add_argument('-p', '--perf_measure', default="reward",
                        help='Name of column to use for metrics calculations')

    args = parser.parse_args()

    if args.log_dir is None:
        raise Exception('Log directory must be specified!')

    # Initialize metrics report
    report = l2metrics.AgentMetricsReport(log_dir=args.log_dir, perf_measure=args.perf_measure)

    # Calculate metrics in order of their addition to the metrics list.
    report.calculate()

    # Plot metrics
    report.plot(save=True)

    # Print table of metrics and save values to file
    report.report(save=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f'Error: {e}')
