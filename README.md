# Fake News Detection with RoBERTa

Hệ thống phát hiện tin giả sử dụng kiến trúc hybrid multi-stage pipeline:
**RoBERTa → MC Dropout → Temperature Scaling → LLM Verification**


## Demo

Watch Demo Video: https://youtu.be/LsJlCSz_lJo

## System Architecture

```
News Text
   │
   ▼
Text Preprocessing (normalize, tokenize, clean)
   │
   ▼
RoBERTa (fine-tuned on news domain)
   │
   ▼
Monte Carlo Dropout (10 forward passes → mean + variance)
   │
   ▼
Temperature Scaling (calibrated probability)
   │
   ├── High confidence (≥ 0.7) → Final Result
   │
   └── Low confidence (< 0.7)
           │
           ▼
       LLM Verification (Groq API) → Final Decision
```

## Project Structure

```
Fake_news_RoBERTa/
├── data/
│   ├── data.csv              # Raw dataset (~357MB)
│   └── processed/            # Train/Val/Test splits
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Data cleaning & splitting
│   ├── 03_baseline_lr_tfidf.ipynb  # Baseline: LR + TF-IDF
│   ├── 04_roberta_training.ipynb   # RoBERTa fine-tuning
│   └── 05_experiments.ipynb        # All 4 experiments
├── src/
│   ├── config.py             # Centralized configuration
│   ├── utils.py              # Utilities (seed, metrics, early stopping)
│   ├── dataset.py            # PyTorch Dataset for RoBERTa
│   ├── model.py              # Model loading + MC Dropout
│   ├── trainer.py            # Training loop
│   ├── evaluation.py         # Evaluation & visualizations
│   ├── calibration.py        # Temperature Scaling (ECE, Brier)
│   └── llm_verifier.py       # LLM fact-checking (Groq)
├── models/                   # Saved models
│   ├── baseline/             # LR + TF-IDF
│   └── roberta/              # Fine-tuned RoBERTa
├── results/                  # Experiment results & plots
├── requirements.txt
└── .gitignore
```

## Experiments

| # | Experiment | Comparison | Metrics |
|---|---|---|---|
| 1 | Baseline | LR+TF-IDF vs RoBERTa | Accuracy, F1 |
| 2 | Domain Adaptation | RoBERTa (general) vs RoBERTa (fine-tuned) | Accuracy, F1 |
| 3 | Calibration | RoBERTa vs RoBERTa + Temperature Scaling | ECE, Brier Score |
| 4 | Hybrid System | RoBERTa vs MC Dropout vs Hybrid+LLM | Accuracy, F1, Uncertainty |

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Notebooks In Order
1. **01_eda.ipynb** — Khám phá dữ liệu (chạy local)
2. **02_preprocessing.ipynb** — Tiền xử lý & chia data (chạy local)
3. **03_baseline_lr_tfidf.ipynb** — Huấn luyện baseline (chạy local)
4. **04_roberta_training.ipynb** — Fine-tune RoBERTa (cần GPU → Google Colab)
5. **05_experiments.ipynb** — Chạy thí nghiệm (cần GPU → Google Colab)

### 3. For Google Colab
Upload `src/` folder và `data/processed/` lên Colab, sau đó chạy notebook 04 và 05.

## Research Contributions

1. **Domain-adapted RoBERTa** cho fake news detection trên dữ liệu tin tức
2. **Monte Carlo Dropout** để ước lượng prediction uncertainty
3. **Temperature Scaling** cải thiện probability calibration
4. **Hybrid detection system** kết hợp RoBERTa classifier và LLM verification

## Tech Stack

- **Model**: RoBERTa-base (Hugging Face Transformers)
- **Baseline**: Logistic Regression + TF-IDF (scikit-learn)
- **Framework**: PyTorch
- **Calibration**: Temperature Scaling
- **LLM**: Groq API (Llama 3.3 70B)

---

Built for graduation thesis research on fake news detection.
