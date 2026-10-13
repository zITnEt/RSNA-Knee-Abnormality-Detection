"""Do the official Gold58 labels follow the conventions the JEV author used for the 16 report-verified studies?
Prints, for the 58 officially labelled studies, the report wording around effusion / synovitis terms and the
official label, plus how often an official positive has no mention of the finding in the report.

    python scripts/check_gold_conventions.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT/'audit/training/labels/train.csv').dropna(subset=['ACL'])
TERMS = {
    'Effusion': r'effusion|derrame|izljev|vocht|hydrops|fluid|l[ií]quido|tekuć',
    'Synovitis': r'synovi|sinovi|hoffa|sinovij',
    'Contusion': r'contusion|bone bruise|edema|oedeem|oedema|edem|contusi',
    'Fracture': r'fractur|frattur|prijelom|breuk',
}
SEV = r'(small|mild|minimal|trace|slight|physiolog|discret|peque|leve|escas|mal[io]|blag|minim|gering|klein|moderate|moderad|umjeren|large|abundant|importante|significant|velik|marked|massive)'
for f, pat in TERMS.items():
    print(f'\n===== {f}: official positives {int(d[f].sum())}/58')
    nomention = 0
    for _, r in d.iterrows():
        text = str(r.Report)
        hits = [m.start() for m in re.finditer(pat, text, flags=re.I)]
        if not hits and r[f] == 1:
            nomention += 1
        snippets = []
        for h in hits[:3]:
            s = re.sub(r'\s+', ' ', text[max(0, h-70):h+60])
            snippets.append(s)
        if hits:
            sev = sorted({m.lower() for s in snippets for m in re.findall(SEV, s, flags=re.I)})
            print(f'label={int(r[f])} sev={sev} | ' + ' || '.join(snippets)[:330])
    print(f'-- official positives with NO mention of the finding terms: {nomention}')
