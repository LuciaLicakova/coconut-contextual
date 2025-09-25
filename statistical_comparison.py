"""
Compare CoT and Coconut checkpoints using metrics and statistical tests.

"""

import re
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

# Load the logs
cot_log_file = "output_cot.log"       
coconut_log_file = "output_coconut.log"

def extract_correct_total(log_file):
    # Parse logs to extract correct / total counts per checkpoint
    checkpoints = []
    with open(log_file) as f:
        for line in f:
            match = re.search(r"Accuracy on validation set: (\d+) / (\d+)", line)
            if match:
                correct = int(match.group(1))
                total = int(match.group(2))
                checkpoints.append((correct, total))
    return checkpoints

cot_checkpoints = extract_correct_total(cot_log_file)
coconut_checkpoints = extract_correct_total(coconut_log_file)

# Compute accuracy and 95% confidence intervals
def compute_metrics(checkpoints):
    # Return list of tuples: (accuracy, CI_low, CI_high)
    metrics = []
    for correct, total in checkpoints:
        acc = correct / total
        ci_low, ci_high = proportion_confint(correct, total, method="wilson")
        metrics.append((acc, ci_low, ci_high))
    return metrics

cot_metrics = compute_metrics(cot_checkpoints)
coco_metrics = compute_metrics(coconut_checkpoints)

print("CoT Checkpoints:")
for i, (acc, ci_low, ci_high) in enumerate(cot_metrics, 1):
    print(f"  Checkpoint {i}: Accuracy={acc:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}]")

print("\nCoconut Checkpoints:")
for i, (acc, ci_low, ci_high) in enumerate(coco_metrics, 1):
    print(f"  Checkpoint {i}: Accuracy={acc:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}]")

# Select the best checkpoint (highest accuracy) for each
best_cot_idx = max(range(len(cot_metrics)), key=lambda i: cot_metrics[i][0])
best_coco_idx = max(range(len(coco_metrics)), key=lambda i: coco_metrics[i][0])

cot_correct, cot_total = cot_checkpoints[best_cot_idx]
coco_correct, coco_total = coconut_checkpoints[best_coco_idx]

# Two-proportion z-test between the best CoT and the best Coconut
count = [cot_correct, coco_correct]
nobs = [cot_total, coco_total]
stat, pval = proportions_ztest(count, nobs)

print("\nStatistical comparison of best checkpoints:")
print(f"  Best CoT checkpoint {best_cot_idx+1}: {cot_correct}/{cot_total} = {cot_correct/cot_total:.3f}")
print(f"  Best Coconut checkpoint {best_coco_idx+1}: {coco_correct}/{coco_total} = {coco_correct/coco_total:.3f}")
print(f"  Z-statistic={stat:.3f}, p-value={pval:.4f}")
# Check if differences are statistically significant
if pval < 0.05:
    print("  => Difference is statistically significant at p<0.05")
else:
    print("  => Difference is NOT statistically significant at p<0.05")
