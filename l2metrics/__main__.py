"""
(c) 2019 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
All Rights Reserved. This material may be only be used, modified, or reproduced
by or for the U.S. Government pursuant to the license rights granted under the
clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission,
please contact the Office of Technology Transfer at JHU/APL.

NO WARRANTY, NO LIABILITY. THIS MATERIAL IS PROVIDED “AS IS.” JHU/APL MAKES NO
REPRESENTATION OR WARRANTY WITH RESPECT TO THE PERFORMANCE OF THE MATERIALS,
INCLUDING THEIR SAFETY, EFFECTIVENESS, OR COMMERCIAL VIABILITY, AND DISCLAIMS
ALL WARRANTIES IN THE MATERIAL, WHETHER EXPRESS OR IMPLIED, INCLUDING (BUT NOT
LIMITED TO) ANY AND ALL IMPLIED WARRANTIES OF PERFORMANCE, MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF INTELLECTUAL PROPERTY
OR OTHER THIRD PARTY RIGHTS. ANY USER OF THE MATERIAL ASSUMES THE ENTIRE RISK
AND LIABILITY FOR USING THE MATERIAL. IN NO EVENT SHALL JHU/APL BE LIABLE TO ANY
USER OF THE MATERIAL FOR ANY ACTUAL, INDIRECT, CONSEQUENTIAL, SPECIAL OR OTHER
DAMAGES ARISING FROM THE USE OF, OR INABILITY TO USE, THE MATERIAL, INCLUDING,
BUT NOT LIMITED TO, ANY DAMAGES FOR LOST PROFITS.
"""
import argparse
import l2metrics


def run():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('-log_dir', default=None, help='Subdirectory under the l2data root for the log files')
    parser.add_argument('-syllabus_type', choices=["agent", "class"],  default="agent", help='Type of syllabus')
    parser.add_argument('-syllabus_subtype', choices=["CL", "ANT_A", "ANT_B", "ANT_C"],  default="ANT_A",
                        help='Type of syllabus')
    args = parser.parse_args()
    
    # We assume that the logs are found as a subdirectory under the l2data root directory
    # This subdirectory must be passed as a parameter in order to locate the logs which will be parsed by this code
    
    if args.syllabus_type == "class":
        report = l2metrics.ClassificationMetricsReport(log_dir=args.log_dir)
    else:
        report = l2metrics.AgentMetricsReport(log_dir=args.log_dir, syllabus_subtype=args.syllabus_subtype)
    
    report.calculate()
    report.report()


if __name__ == "__main__":
    run()
