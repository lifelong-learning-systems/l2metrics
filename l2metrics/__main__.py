import argparse
import l2metrics
import learnkit

def run():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-log_rootdir', default=None,  help='Root directory for logs')
    parser.add_argument('-type', choices=["agent", "class"],  default="agent", help='Type of syllabus')
    args = parser.parse_args()
    
    if args.log_rootdir is None:
        args.log_rootdir = learnkit.data_util.utils.get_l2data_root()
    
    if args.type == "agent":
        report = l2metrics.AgentMetricsReport(log_rootdir=args.log_rootdir)
    else:
        report = l2metrics.ClassificationMetricsReport(log_rootdir=args.log_rootdir)
    
    report.calculate()
    report.plot()


if __name__ == "__main__":
    run()
