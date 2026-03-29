# Arabizi Offensive Speech Detection

Multilingual offensive speech detection on **Algerian Arabizi** — a code-switched dialect mixing Arabic, French, and Latin script with numeric substitutions (3=ع, 7=ح, 9=ق).

## Highlights

- **ArabiziNormalizer**: maps numeric substitutions (e.g. `7` → `h`, `3` → `aa`), reduces elongation (`looool` → `lol`), detects offensive intensity
- **ArabiziAugmenter**: char swap, keyboard typo injection, Arabizi synonym replacement
- **BERTWithDialect**: custom BERT head that concatenates [CLS] token with a learned dialect embedding (32-dim) before classification
- **Stacking meta-classifier**: LR/SVM/RF/BERT predictions combined via cross-val stacking
- **Error analysis by dialect**: FP/FN breakdown per dialect (Algerian/Moroccan)

## Project Structure

```
├── preprocessing.py       # ArabiziNormalizer, text cleaning
├── augmentation.py        # ArabiziAugmenter, char swap, typo injection
├── models.py              # Classical ML + BERTWithDialect
├── ensemble.py            # Stacking meta-classifier
├── error_analysis.py      # Confusion matrices, dialect-level error breakdown
├── requirements.txt
├── notebooks/
│   └── TP5.ipynb
└── results/
```

## Models

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | Classical | TF-IDF + handcrafted features |
| Linear SVM | Classical | Best classical baseline |
| Random Forest | Classical | With emoji/offensive intensity features |
| **BERT+Dialect** | Deep | AraBERT/DziriBERT + dialect embedding |
| **Meta-Classifier** | Stacking | LR over all model predictions (5-fold CV) |

## Dataset

Algerian/Moroccan Arabic social media corpus with offensive speech labels.  
Features: text, dialect tag (DZ/MA), offensive/non-offensive binary label.

## Key Design Decisions

- **Arabizi normalization before tokenization**: numeric characters (3,7,9…) must be mapped to phonetic equivalents before any TF-IDF or BERT tokenizer sees the text
- **Dialect as a feature, not a filter**: instead of training separate models per dialect, a 32-dim embedding captures dialect-specific patterns without dataset fragmentation
- **Stacking over voting**: meta-classifier uses calibrated probabilities from 5-fold CV to avoid train-set leakage

## License

MIT
