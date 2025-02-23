import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

NEGSEL_JAR = "negsel2.jar"

CHUNK_LENGTH = 10 
USE_OVERLAPPING = True 

#in theory we can add more r values, but takes too long >_<
R_VALUES = [4]  

DATASETS = [
    ("snd-cert", "snd-cert", ["snd-cert.1", "snd-cert.2", "snd-cert.3"]),
    ("snd-unm",  "snd-unm",  ["snd-unm.1",  "snd-unm.2",  "snd-unm.3"])
]

OUTPUT_DIR = "outputs_unix" 
os.makedirs(OUTPUT_DIR, exist_ok=True)



def load_sequences(file_path):
    """
    Return a list of strings (each line in file_path)
    """
    seqs = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                seqs.append(line)
    return seqs


def chunk_sequence(seq, chunk_len=10, overlapping=True):
    """
    break single sequence into chunks of length chunk_len
    If overlapping=True -> sliding window; otherwise: blocks
    """
    chunks = []
    L = len(seq)
    if overlapping:
        for start in range(0, L - chunk_len + 1):
            chunks.append(seq[start:start+chunk_len])
    else:
        for start in range(0, L, chunk_len):
            sub = seq[start:start+chunk_len]
            if len(sub) == chunk_len:
                chunks.append(sub)
    return chunks


def write_lines_to_file(lines, out_path):
    with open(out_path, "w") as f:
        for ln in lines:
            f.write(f"{ln}\n")


def run_negsel(alpha_file, train_chunks_path, test_chunks_path, n_val, r_val):
    """
    call to negsel2.jar on the entire test_chunks_path
    """
    cmd = (
        f"java -jar {NEGSEL_JAR} "
        f"-alphabet file://{alpha_file} "
        f"-self {train_chunks_path} "
        f"-n {n_val} "
        f"-r {r_val} "
        f"-c -l "
        f"< {test_chunks_path}"
    )
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    scores = []
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                scores.append(float(line))
            except ValueError:
                pass
    return scores


def compute_roc_auc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return fpr, tpr, auc(fpr, tpr)



def analyze_intrusion(folder_path, train_prefix, test_prefix, chunk_len, r_val):
    alpha_file = os.path.join(folder_path, f"{train_prefix}.alpha")
    train_file = os.path.join(folder_path, f"{train_prefix}.train")
    test_file  = os.path.join(folder_path, f"{test_prefix}.test")
    label_file = os.path.join(folder_path, f"{test_prefix}.labels")

    test_seqs = load_sequences(test_file)
    labels = []
    with open(label_file, "r") as lf:
        for line in lf:
            line=line.strip()
            if line:
                labels.append(int(line))

    all_test_chunks = []
    chunk_counts = [] 
    for seq in test_seqs:
        cseq = chunk_sequence(seq, chunk_len, overlapping=USE_OVERLAPPING)
        all_test_chunks.extend(cseq)
        chunk_counts.append(len(cseq))

    test_chunks_path = os.path.join(OUTPUT_DIR, f"{test_prefix}_allchunks.txt")
    write_lines_to_file(all_test_chunks, test_chunks_path)

    train_chunks_path = os.path.join(OUTPUT_DIR, f"{train_prefix}_trainchunks.txt")
    chunk_scores = run_negsel(alpha_file, train_chunks_path, test_chunks_path, chunk_len, r_val)

    final_scores = []
    idx = 0
    for ccount in chunk_counts:
        if ccount == 0:
            final_scores.append(0.0)
        else:
            seq_chunk_scores = chunk_scores[idx : idx+ccount]
            avg_score = float(np.mean(seq_chunk_scores))
            final_scores.append(avg_score)
        idx += ccount

    fpr, tpr, rocVal = compute_roc_auc(labels, final_scores)

    outplot = os.path.join(OUTPUT_DIR, f"ROC_{test_prefix}_r{r_val}.png")
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={rocVal:.3f}")
    plt.plot([0,1],[0,1],'r--')
    plt.title(f"ROC: {test_prefix}, r={r_val}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outplot)
    plt.close()

    return rocVal


def main():
    summary = []
    for (subfolder, trainpref, testprefixes) in DATASETS:
        folder_path = os.path.join("syscalls", subfolder)
        if not os.path.isdir(folder_path):
            print(f"no folder {folder_path}")
            continue

        train_chunks_path = os.path.join(OUTPUT_DIR, f"{trainpref}_trainchunks.txt")
        if not os.path.isfile(train_chunks_path):
            print(f"Building training chunks for {subfolder}/{trainpref}...")
            train_file = os.path.join(folder_path, f"{trainpref}.train")
            train_seqs = load_sequences(train_file)
            train_chunks = []
            for seq in train_seqs:
                cseq = chunk_sequence(seq, CHUNK_LENGTH, overlapping=USE_OVERLAPPING)
                train_chunks.extend(cseq)
            write_lines_to_file(train_chunks, train_chunks_path)
        else:
            print(f"Already have training chunks for {subfolder}/{trainpref} at {train_chunks_path}")

        for testpref in testprefixes:
            for r_val in R_VALUES:
                print(f"\n[+] Processing {subfolder}/{testpref}, r={r_val}, chunk_len={CHUNK_LENGTH} ...")
                rocVal = analyze_intrusion(
                    folder_path, trainpref, testpref,
                    chunk_len=CHUNK_LENGTH, r_val=r_val
                )
                summary.append((subfolder, testpref, r_val, rocVal))

    sumfile = os.path.join(OUTPUT_DIR, "intrusion_summary.csv")
    with open(sumfile, "w") as f:
        f.write("dataset,testprefix,r,auc\n")
        for (dset, tpref, rr, av) in summary:
            f.write(f"{dset},{tpref},{rr},{av:.4f}\n")

    print(f"\nsummary file: {sumfile}")


if __name__ == "__main__":
    main()
