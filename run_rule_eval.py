import sys
import time
sys.path.insert(0, 'src')

from dataset import Talk2CarDataset
from evaluate import evaluate_ap50
from baselines import random_baseline, largest_box_baseline, class_match_baseline
from rule_based import rule_based_score, rule_class_only, rule_class_plus_spatial, rule_no_color, rule_no_confidence

val_ds = Talk2CarDataset(split='val', data_dir='data')

methods = [
    ("Random", random_baseline),
    ("Largest-box", largest_box_baseline),
    ("Class match (oracle)", class_match_baseline),
    ("Rule: class only", rule_class_only),
    ("Rule: class + spatial", rule_class_plus_spatial),
    ("Rule: no color", rule_no_color),
    ("Rule: no confidence", rule_no_confidence),
    ("Rule: full", rule_based_score),
]

print(f"{'Method':<25} {'AP50':>8} {'Time':>8}")
print("-" * 45)
for name, fn in methods:
    t0 = time.time()
    score = evaluate_ap50(fn, val_ds)
    print(f"{name:<25} {score:>8.4f} {time.time()-t0:>7.1f}s")
