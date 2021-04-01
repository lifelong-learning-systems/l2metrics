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
import traceback
from pathlib import Path


def main() -> None:
    # Instantiate parser
    parser = argparse.ArgumentParser(description='Validate evaluation submission from the command line')

    # Evaluation directory be absolute or relative paths
    parser.add_argument('eval_dir', type=str,
                        help='Evaluation directory for certain month (e.g., ./m12_eval/)')

    # Parse arguments
    args = parser.parse_args()

    # Generate path from evaluation directory input
    eval_dir: Path = Path(args.eval_dir)

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

    num_required_files: int = len(required_files)
    num_files_found: int = 0

    for f in required_files:
        if hasattr(f, '__iter__'):
            if not (f[0].exists() or f[1].exists()):
                print(f'Missing file: {f[0].with_suffix("")}')
            else:
                num_files_found = num_files_found + 1
        elif not f.exists():
            print(f'Missing file: {f}')
        else:
            num_files_found = num_files_found + 1

    if num_files_found == num_required_files:
        print('Everything has been submitted!')
    else:
        print(f'\nSubmitted {num_files_found} / {num_required_files} files')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
