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
    parser.add_argument('-log_dir', default=None, help='Subdirectory under $L2DATA/logs for the log files')

    # Choose syllabus type "agent" for Agent-based environments, and "class" for Classification-based environments
    parser.add_argument('-syllabus_type', choices=["agent", "class"],  default="agent", help='Type of learner '
                                                                                             'used in the syllabus')

    # Syllabus_subtype refers to the structure of the syllabus and will determine the default list of metrics calculated
    # where CL = Continual Learning; ANT_A = Adapting to New Tasks, type A; ANT_B = Adapting to New Tasks, type B; etc.
    # Please refer to the documentation for more details on the distinction between these types.
    parser.add_argument('-syllabus_subtype', choices=["CL", "ANT_A", "ANT_B", "ANT_C"],  default="ANT_A",
                        help='Subtype of syllabus')

    args = parser.parse_args()

    if args.log_dir is None:
        raise Exception('Log directory must be specified!')
    
    if args.syllabus_type == "class":
        report = l2metrics.ClassificationMetricsReport(log_dir=args.log_dir)
    else:
        report = l2metrics.AgentMetricsReport(log_dir=args.log_dir, syllabus_subtype=args.syllabus_subtype)
    
    report.calculate()
    report.report()


if __name__ == "__main__":
    run()
