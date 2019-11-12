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
