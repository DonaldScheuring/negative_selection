import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#config
NEGSEL_JAR = "negsel2.jar"
CHUNK_LENGTH = 10
USE_OVERLAPPING = True
#we could use a range for r, but this takes way too long >_<
R_VALUES = [4]

DATASETS = [
    ("snd-cert", "snd-cert", ["snd-cert.1", "snd-cert.2", "snd-cert.3"]),
    ("snd-unm",  "snd-unm",  ["snd-unm.1",  "snd-unm.2",  "snd-unm.3"])
]

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def load_sequences(file_path):
    seqs = []
    with open(file_path, "r") as f:
        for line in f:
            line=line.strip()
            if line:
                seqs.append(line)
    return seqs

def chunk_sequence(seq, chunk_len=10, overlapping=True):
    chunks = []
    L = len(seq)
    if overlapping:
        for start in range(0, L - chunk_len + 1):
            chunks.append(seq[start : start + chunk_len])
    else:
        for start in range(0, L, chunk_len):
            chunk = seq[start : start + chunk_len]
            if len(chunk) == chunk_len:
                chunks.append(chunk)
    return chunks

def write_chunks_to_file(chunks, out_path):
    with open(out_path, "w") as f:
        for c in chunks:
            f.write(f"{c}\n")

def run_negsel(alpha_file, train_chunks_file, test_chunks_file, n_val, r_val):
    cmd = (
        f"java -jar {NEGSEL_JAR} "
        f"-alphabet file://{alpha_file} "
        f"-self {train_chunks_file} "
        f"-n {n_val} "
        f"-r {r_val} "
        f"-c -l "
        f"< {test_chunks_file}"
    )
    #print(cmd)
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    scores = []
    for line in proc.stdout.strip().split("\n"):
        if line.strip():
            try:
                scores.append(float(line.strip()))
            except ValueError:
                pass
    return scores

def compute_roc_auc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return fpr, tpr, auc(fpr, tpr)


def analyze_intrusion(dataset_folder, train_prefix, test_prefix, chunk_len, r_val):
    alpha_file = os.path.join(dataset_folder, f"{train_prefix}.alpha")
    train_file = os.path.join(dataset_folder, f"{train_prefix}.train")
    test_file  = os.path.join(dataset_folder, f"{test_prefix}.test")
    label_file = os.path.join(dataset_folder, f"{test_prefix}.labels")

    train_seqs = load_sequences(train_file)
    train_chunks = []
    for seq in train_seqs:
        train_chunks.extend(chunk_sequence(seq, chunk_len, overlapping=USE_OVERLAPPING))
    train_chunks_path = os.path.join(OUTPUT_DIR, f"{test_prefix}_train_chunks.txt")
    write_chunks_to_file(train_chunks, train_chunks_path)

    test_seqs = load_sequences(test_file)
    labels = [int(x.strip()) for x in open(label_file).readlines() if x.strip()]

    final_scores = []
    for seq in test_seqs:
        seq_chunks = chunk_sequence(seq, chunk_len, overlapping=USE_OVERLAPPING)
        temp_test_chunk_path = os.path.join(OUTPUT_DIR, f"{test_prefix}_testchunks_tmp.txt")
        write_chunks_to_file(seq_chunks, temp_test_chunk_path)

        #run negsel
        chunk_scores = run_negsel(alpha_file, train_chunks_path, temp_test_chunk_path, chunk_len, r_val)
        if len(chunk_scores) == 0:
            final_scores.append(0.0)
        else:
            final_scores.append(float(np.mean(chunk_scores)))

    #roc/auc comp
    fpr, tpr, auc_val = compute_roc_auc(labels, final_scores)

    outpath_roc = os.path.join(OUTPUT_DIR, f"ROC_{test_prefix}_r{r_val}.png")
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
    plt.plot([0,1],[0,1],'r--')
    plt.title(f"ROC: {test_prefix}, r={r_val}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(outpath_roc)
    plt.close()

    return auc_val

def main():
    summary = []
    n_val = CHUNK_LENGTH
    for (subfolder, trainpref, testprefixes) in DATASETS:
        folder_path = os.path.join("syscalls", subfolder)
        if not os.path.isdir(folder_path):
            print(f"no folder {folder_path}")
            continue
        for testpref in testprefixes:
            for r in R_VALUES:
                print(f"Processing {subfolder}/{testpref}: r={r}, chunk_len={CHUNK_LENGTH}")
                auc_val = analyze_intrusion(
                    dataset_folder=folder_path,
                    train_prefix=trainpref,
                    test_prefix=testpref,
                    chunk_len=CHUNK_LENGTH,
                    r_val=r
                )
                summary.append((subfolder, testpref, r, auc_val))

    sumfile = os.path.join(OUTPUT_DIR, "intrusion_summary.csv")
    with open(sumfile, "w") as f:
        f.write("dataset,testprefix,r,auc\n")
        for (dset, tpref, rr, av) in summary:
            f.write(f"{dset},{tpref},{rr},{av:.4f}\n")

    print(f"\n script done; summary file: {sumfile}\n")

if __name__ == "__main__":
    main()