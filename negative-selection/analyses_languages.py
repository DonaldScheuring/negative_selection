"""
This right now script includes:
 - loop over r=1..9 (fix n=10)
 - train on 'english.train'
 - test each 'english.test' line as normal (label=0) and each other language line as anomalous (label=1)
 - compute ROC and AUC; generate plots; store numerical results
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#config
JAR_FILE = "negsel2.jar" 
TRAIN_FILE = "english.train" 
TEST_ENGLISH = "english.test" 
TEST_TAGALOG = "tagalog.test"
LANG_DIR = "lang"
LANG_FILES = [
    "hiligaynon.txt",
    "middle-english.txt",
    "plautdietsch.txt",
    "xhosa.txt"
]
N_VALUE = 10
R_VALUES = range(1, 10)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_negsel(r_value, test_file):
    """
    run negsel2.jar with the specified r_value on test_file
    return list of floating anomaly scores (one per line in test_file)
    """
    cmd = (
        f"java -jar {JAR_FILE} "
        f"-self {TRAIN_FILE} "
        f"-n {N_VALUE} "
        f"-r {r_value} "
        f"-c -l "
        f"< {test_file}"
    )
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    scores = []
    for line in proc.stdout.strip().split("\n"):
        stripped = line.strip()
        if stripped:
            try:
                scores.append(float(stripped))
            except ValueError:
                pass
    return scores


def compute_roc_auc(scores_normal, scores_anomalous):
    y_true = np.array([0]*len(scores_normal) + [1]*len(scores_anomalous))
    scores = np.array(scores_normal + scores_anomalous)

    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    auc_value = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_value


def plot_roc(fpr, tpr, auc_value, title, out_path):
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    #random classifier baseline
    plt.plot([0,1],[0,1], 'r--', label="Random Classifier")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def analyze_toy_example():
    results = []
    for r in R_VALUES:
        print(f"negative selection for r={r}")
        scores_en = run_negsel(r, TEST_ENGLISH)
        scores_ta = run_negsel(r, TEST_TAGALOG)

        fpr, tpr, _, auc_val = compute_roc_auc(scores_en, scores_ta)
        results.append((r, auc_val))

        out_plot = os.path.join(OUTPUT_DIR, f"ROC_r_{r}_English_vs_Tagalog.png")
        title = f"ROC for r={r} (English vs Tagalog)"
        plot_roc(fpr, tpr, auc_val, title, out_plot)
        print(f"AUC={auc_val:.3f}, plot saved to: {out_plot}")

    summary_path = os.path.join(OUTPUT_DIR, "summary_r_vs_auc_toy.csv")
    with open(summary_path, "w") as f:
        f.write("r,AUC\n")
        for (r, auc_val) in results:
            f.write(f"{r},{auc_val:.6f}\n")
    print(f"summary for toy example: {summary_path}\n")


def analyze_multiple_languages():
    english_scores_by_r = {}
    for r in R_VALUES:
        english_scores_by_r[r] = run_negsel(r, TEST_ENGLISH)

    lang_results = {}

    for lang_file in LANG_FILES:
        lang_path = os.path.join(LANG_DIR, lang_file)
        if not os.path.isfile(lang_path):
            print(f"WARNING: {lang_path} not found, skipping.")
            continue

        r_auc_pairs = []
        for r in R_VALUES:
            scores_en = english_scores_by_r[r]
            scores_lang = run_negsel(r, lang_path)
            fpr, tpr, _, auc_val = compute_roc_auc(scores_en, scores_lang)
            r_auc_pairs.append((r, auc_val))

        lang_results[lang_file] = r_auc_pairs

    summary_path = os.path.join(OUTPUT_DIR, "summary_r_vs_auc_languages.csv")
    with open(summary_path, "w") as f:
        f.write("language,r,auc\n")
        for lang_file, r_auc_pairs in lang_results.items():
            for (r, auc_val) in r_auc_pairs:
                f.write(f"{lang_file},{r},{auc_val:.6f}\n")
    print(f"Per-language results saved to: {summary_path}")

    plt.figure(figsize=(6,4))
    for lang_file, r_auc_pairs in lang_results.items():
        rs = [x[0] for x in r_auc_pairs]
        aucs = [x[1] for x in r_auc_pairs]
        plt.plot(rs, aucs, marker='o', label=lang_file)
    plt.xlabel("r value")
    plt.ylabel("AUC")
    plt.title("English vs. Each Language: AUC vs. r")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    multiline_plot_path = os.path.join(OUTPUT_DIR, "AUC_comparison_all_languages.png")
    plt.savefig(multiline_plot_path)
    plt.close()
    print(f"multi-line AUC comparison saved to: {multiline_plot_path}")

    #each language: pick the best r, plot the ROC
    for lang_file, r_auc_pairs in lang_results.items():
        best_r, best_auc = max(r_auc_pairs, key=lambda tup: tup[1])
        #re-run to get fpr,tpr for the best r
        scores_en = english_scores_by_r[best_r]
        lang_path = os.path.join(LANG_DIR, lang_file)
        scores_lang = run_negsel(best_r, lang_path)
        fpr, tpr, _, _ = compute_roc_auc(scores_en, scores_lang)
        out_plot = os.path.join(OUTPUT_DIR, f"ROC_best_r_{best_r}_{lang_file}.png")
        title = f"{lang_file} vs English (Best r={best_r}, AUC={best_auc:.3f})"
        plot_roc(fpr, tpr, best_auc, title, out_plot)
        print(f"  Best r for {lang_file}: r={best_r}, AUC={best_auc:.3f}. ROC saved to {out_plot}")

   


def main():
    analyze_toy_example()
    analyze_multiple_languages()
    print("analysis finished")
    

if __name__ == "__main__":
    main()
