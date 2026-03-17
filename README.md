# 🌐 English → Hindi Neural Machine Translator

A sequence-to-sequence neural machine translation system using 
Encoder-Decoder architecture with Bahdanau Attention, built with PyTorch.

---

## Architecture
```
Input (English)
      ↓
 Embedding Layer (128-dim)
      ↓
 Bidirectional GRU Encoder
 [reads sentence forward + backward]
      ↓
 Encoder Outputs + Hidden State
      ↓
 Bahdanau Attention
 [alignment score = v·tanh(W1·hidden + W2·encoder_out)]
 [context = weighted sum of encoder outputs]
      ↓
 GRU Decoder + Teacher Forcing (50%)
      ↓
 Linear Layer → Softmax over Hindi Vocab
      ↓
 Output (Hindi)
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| EMBED_DIM | 128 | Word embedding dimension |
| HIDDEN_DIM | 256 | GRU hidden state size |
| N_LAYERS | 1 | Number of GRU layers |
| DROPOUT | 0.3 | Dropout rate for regularization |
| BATCH_SIZE | 64 | Sentences per gradient update |
| N_EPOCHS | 10 | Training epochs |
| LEARNING_RATE | 1e-3 | Initial Adam learning rate |
| GRAD_CLIP | 1.0 | Gradient clipping threshold |
| TEACHER_FORCING | 0.5 | 50% chance of using true target |
| OPTIMIZER | Adam | Adaptive learning rate optimizer |
| SCHEDULER | ReduceLROnPlateau | Halves LR after 2 bad epochs |
| LOSS | CrossEntropyLoss | Ignores PAD tokens (index=0) |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [HindiEnglish Corpora - Kaggle](https://www.kaggle.com/datasets/aiswaryaramachandran/hindienglish-corpora) |
| Total pairs | 122,885 |
| Subset used | 30,000 |
| Train split | 27,000 (90%) |
| Val split | 3,000 (10%) |
| Max length | 50 tokens |
| Min word freq | 2 |
| English vocab | ~40,738 words |
| Hindi vocab | ~39,794 words |
| Stopwords | NOT removed (as required) |

---

## Training Results

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 01 | 6.8891 | 6.6614 ✓ |
| 02 | 5.9784 | 6.4584 ✓ |
| 03 | 5.3067 | **6.3935** ✓ best |
| 04 | 4.6909 | 6.4412 |
| 05 | 4.2405 | 6.5079 |
| 06 | 3.9528 | 6.5861 |
| 07 | 3.6302 | 6.5897 |
| 08 | 3.4728 | 6.6304 |
| 09 | 3.3363 | 6.6797 |
| 10 | 3.2016 | 6.6922 |

**Best model saved at Epoch 3 — Val Loss: 6.3935**  
Training device: CUDA (Kaggle P100 GPU)

---

## Model Checkpoint

The trained model (`best_model.pt`, ~183MB) exceeds GitHub's 100MB 
file size limit and is stored externally.

**Download:** [Google Drive - best_model.pt](YOUR_GOOGLE_DRIVE_LINK)

After downloading, place at: `checkpoints/best_model.pt`

**Or retrain from scratch:**
```bash
python src/download_data.py
python src/preprocess.py
cd src && python train.py
```

---

## Setup & Run
```bash
# 1. Clone
git clone https://github.com/Maanasvee/Eng-Hindi-TranslatorV2.git
cd Eng-Hindi-TranslatorV2

# 2. Install
pip install -r requirements.txt

# 3. Download model from Google Drive link above
#    Place at: checkpoints/best_model.pt

# 4. Run frontend
cd frontend
python app.py

# 5. Open browser
# http://localhost:5000
```

---

## Project Structure
```
Eng-Hindi-Translator/
├── src/
│   ├── download_data.py   # Download via kagglehub
│   ├── preprocess.py      # Clean, tokenize, build vocab
│   ├── model.py           # Encoder + Attention + Decoder
│   ├── train.py           # Training loop (see for all params)
│   └── translate.py       # Inference utility
├── frontend/
│   ├── app.py             # Flask REST API
│   ├── templates/
│   │   └── index.html     # Translator UI
│   └── static/
│       └── style.css      # Styling
├── data/                  # Dataset files (gitignored)
├── checkpoints/           # Model weights (gitignored)
├── .gitignore
└── requirements.txt
```

---

## Author
**Maanasvee Khetan**  
6th Semester — ATML Lab