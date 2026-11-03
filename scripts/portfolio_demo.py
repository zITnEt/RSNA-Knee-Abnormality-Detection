"""Exercise the submission contract using synthetic scores; no model or medical data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_submission import validate

FINDINGS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
            'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('demo-output'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ids = [f'synthetic-study-{i:03d}' for i in range(8)]
    sample = pd.DataFrame({'StudyInstanceUID': ids, **{label: .5 for label in FINDINGS}})
    submission = sample.copy()
    submission[FINDINGS] = np.random.default_rng(42).uniform(.01, .99, (len(ids), 12))
    sample_path, submission_path = args.output / 'sample_submission.csv', args.output / 'submission.csv'
    sample.to_csv(sample_path, index=False)
    submission.to_csv(submission_path, index=False)
    report = validate(submission_path, sample_path)
    if not report.startswith('status: PASS'):
        raise RuntimeError(report)
    (args.output / 'validation.txt').write_text(report, encoding='utf-8')
    invalid = submission.copy()
    invalid.iloc[0, 1] = 1.1
    invalid_path = args.output / 'invalid_example.csv'
    invalid.to_csv(invalid_path, index=False)
    if not validate(invalid_path, sample_path).startswith('status: FAIL'):
        raise RuntimeError('Validator did not reject an out-of-range synthetic score')
    receipt = dict(synthetic=True, model_loaded=False, medical_data_used=False,
                   studies=len(ids), findings=len(FINDINGS), valid_submission='PASS',
                   out_of_range_rejection='PASS', output=str(args.output.resolve()))
    (args.output / 'demo_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
