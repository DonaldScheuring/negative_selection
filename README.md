# Negative Selection Assignment

This repository contains code and data for an anomaly detection assignment using a negative selection algorithm. It comprises two main parts:

1. **Language Classification (analyses_languages.py)**: Distinguish English text from various other languages, computing ROC curves and AUC metrics.
2. **Intrusion Detection (analyses_unix_intrusion.py)**: Apply negative selection to detect anomalous system-call traces on Unix processes.

## Repository Contents

- **`negsel2.jar`**  
  Java implementation of the negative selection algorithm.

- **`english.train`, `english.test`, `tagalog.test`, and `lang/`**  
  Datasets for the language classification task  
  - `english.train`: Used to train negative selection detectors on English.  
  - `english.test`: Test set of English lines (normal).  
  - `tagalog.test`: Test set of Tagalog lines (anomalous).  
  - `lang/`: Contains additional languages for further testing.

- **`syscalls/`**  
  Contains subfolders (`snd-cert`, `snd-unm`) with Unix system-call traces for the intrusion detection task. Each subfolder has `.train`, `.test`, `.labels`, and an `.alpha` file.

- **`analyses_languages.py`**  
  Python script for language classification. Computes ROC, AUC for different parameter values and languages.

- **`analyses_unix_intrusion.py`**  
  Python script for intrusion detection. Performs chunking of system-call data, runs negative selection, and calculates AUC.

- **`outputs/`**  
  Folders where the scripts save their plots (PNG) and summaries (CSV). 


## Requirements

- **Python 3.7+**
- Python packages:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
- **Java** (for `negsel2.jar`)

