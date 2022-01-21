"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    # Instantiate parser
    parser = argparse.ArgumentParser(description='Validate evaluation submission from the command line')

    # Evaluation directory be absolute or relative paths
    parser.add_argument('eval_dir', type=str,
                        help='Evaluation directory for certain month (e.g., ./m12_eval/)')

    # Parse arguments
    args = parser.parse_args()

    # Generate path from evaluation directory input
    eval_dir = Path(args.eval_dir)

    required_files = ()

    if 'm12' in str(eval_dir):
        required_files = (
            (eval_dir / 'docs/eval_protocol.docx', eval_dir / 'docs/eval_protocol.pdf'),
            (eval_dir / 'docs/system_architecture.pptx', eval_dir / 'docs/system_architecture.pdf'),
            eval_dir / 'docs/task_relationships.csv',
            eval_dir / 'agent_config/docs/computation_ste.csv',
            eval_dir / 'agent_config/docs/computation_scenario_2-intermediate_permuted.csv',
            eval_dir / 'agent_config/docs/computation_scenario_2-intermediate_alternating.csv',
            eval_dir / 'agent_config/ll_logs/2-intermediate_permuted_logs.zip',
            eval_dir / 'agent_config/ll_logs/2-intermediate_alternating_logs.zip',
            eval_dir / 'agent_config/ste_logs/ste_logs.zip',
        )
    elif 'm15' in str(eval_dir):
        required_files = (
            (eval_dir / 'docs/eval_protocol.docx', eval_dir / 'docs/eval_protocol.pdf'),
            (eval_dir / 'docs/system_architecture.pptx', eval_dir / 'docs/system_architecture.pdf'),
            eval_dir / 'docs/task_variant_relationships.csv',
            eval_dir / 'agent_config/docs/computation_ste.csv',
            eval_dir / 'agent_config/docs/computation_scenario_3-high_condensed.csv',
            eval_dir / 'agent_config/docs/computation_scenario_3-high_dispersed.csv',
            eval_dir / 'agent_config/ll_logs/3-high_condensed.zip',
            eval_dir / 'agent_config/ll_logs/3-high_dispersed.zip',
            eval_dir / 'agent_config/ste_logs/ste_logs.zip',
        )
    else:
        logger.error('Validation for directory not implemented!')
        return

    num_required_files = len(required_files)
    num_files_found = 0

    for required_file in required_files:
        if hasattr(required_file, '__iter__'):
            if any(filepath.exists() for filepath in required_file):
                num_files_found = num_files_found + 1
            else:
                logger.error(f'Missing file: {required_file[0].with_suffix("")}')
        elif not required_file.exists():
            logger.error(f'Missing file: {required_file}')
        else:
            num_files_found = num_files_found + 1

    if num_files_found == num_required_files:
        logger.info('Everything has been submitted!')
    else:
        logger.info(f'\nSubmitted {num_files_found} / {num_required_files} files')


if __name__ == '__main__':
    # Configure logger
    logging.basicConfig(level=logging.INFO)

    main()
