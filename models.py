#MODÈLES CLASSIQUES


print("MODÈLES CLASSIQUES")

# Préparation des données
X_text = df_augmented['text_clean']
X_features = df_augmented[['length', 'word_count', 'uppercase_ratio', 'exclamation_count',
                           'question_count', 'emoji_count', 'offensive_intensity']]
y = df_augmented['label']
dialects = df_augmented['Dialect']

# Split avec stratification
X_text_train, X_text_temp, X_feat_train, X_feat_temp, y_train, y_temp, d_train, d_temp = train_test_split(
    X_text, X_features, y, dialects, test_size=0.4, random_state=42, stratify=y
)

X_text_dev, X_text_test, X_feat_dev, X_feat_test, y_dev, y_test, d_dev, d_test = train_test_split(
    X_text_temp, X_feat_temp, y_temp, d_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTaille des ensembles:")
print(f"  Train: {len(X_text_train)} ({len(X_text_train)/len(df_augmented)*100:.1f}%)")
print(f"  Dev:   {len(X_text_dev)} ({len(X_text_dev)/len(df_augmented)*100:.1f}%)")
print(f"  Test:  {len(X_text_test)} ({len(X_text_test)/len(df_augmented)*100:.1f}%)")

# TF-IDF avec n-grammes mixtes (mots + caractères)
print("\n--- Vectorisation TF-IDF Avancée ---")

tfidf_word = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=3000,
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)

tfidf_char = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),
    max_features=3000,
    min_df=1
)

# Entraînement
X_train_word = tfidf_word.fit_transform(X_text_train)
X_dev_word = tfidf_word.transform(X_text_dev)
X_test_word = tfidf_word.transform(X_text_test)

X_train_char = tfidf_char.fit_transform(X_text_train)
X_dev_char = tfidf_char.transform(X_text_dev)
X_test_char = tfidf_char.transform(X_text_test)

# Combiner TF-IDF avec features manuelles
from scipy.sparse import hstack

X_train_combined = hstack([X_train_word, X_train_char, X_feat_train.values])
X_dev_combined = hstack([X_dev_word, X_dev_char, X_feat_dev.values])
X_test_combined = hstack([X_test_word, X_test_char, X_feat_test.values])

print(f"TF-IDF + Features): {X_train_combined.shape}")

# Modèles avec poids de classes
print("\n--- Entraînement des modèles classiques ---")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0),
    'Linear SVM': LinearSVC(max_iter=2000, class_weight='balanced', C=0.5),
    'Complement NB': ComplementNB(alpha=0.1),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                           max_depth=20, random_state=42)
}

def evaluate_model(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = np.mean(f1)

    results = {
        'Accuracy': acc,
        'Precision_0': p[0],
        'Recall_0': r[0],
        'F1_0': f1[0],
        'Precision_1': p[1],
        'Recall_1': r[1],
        'F1_1': f1[1],
        'Macro_F1': macro_f1,
    }

    if y_prob is not None:
        try:
            results['AUC'] = roc_auc_score(y_true, y_prob)
        except:
            results['AUC'] = 0.0

    return results

# Entraînement et évaluation
results_list = []
trained_models = {}

for name, model in models.items():
    print(f"\n {name}")

    model.fit(X_train_combined, y_train)
    y_pred = model.predict(X_test_combined)

    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_combined)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_test_combined)

    results = evaluate_model(y_test, y_pred, y_prob)
    results['Model'] = name
    results_list.append(results)

    trained_models[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    print(f"  F1 Offensive: {results['F1_1']:.4f}, Macro F1: {results['Macro_F1']:.4f}")

# Résultats
results_df = pd.DataFrame(results_list)
print("Résultats des modèles classiques")
print(results_df[['Model', 'Accuracy', 'F1_0', 'F1_1', 'Macro_F1', 'AUC']].to_string(index=False))

best_classic = results_df.loc[results_df['Macro_F1'].idxmax(), 'Model']
print(f"\nMeilleur modèle classique: {best_classic}")

#MODÈLE BERT AVEC DIALECT EMBEDDING

from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments
)

print("BERT AVEC INTÉGRATION DU DIALECTE")

class BERTWithDialect(nn.Module):
    def __init__(self, bert_model_name, num_labels=2, num_dialects=2, dropout=0.3):
        super().__init__()
        # Load base BERT model
        self.bert = AutoModel.from_pretrained(bert_model_name)

        # Dialect embedding
        self.dialect_embedding = nn.Embedding(num_dialects, 32)

        # Classification head
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + 32, num_labels)

    def forward(self, input_ids, attention_mask, dialect_ids, labels=None):
        # BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]

        # Dialect embedding
        dialect_emb = self.dialect_embedding(dialect_ids)

        # Concatenate
        combined = torch.cat([pooled, dialect_emb], dim=1)
        combined = self.dropout(combined)

        # Classification
        logits = self.classifier(combined)

        return {'logits': logits}

# Préparation données BERT
dialect_map = {d: i for i, d in enumerate(df_augmented['Dialect'].unique())}
df_augmented['dialect_id'] = df_augmented['Dialect'].map(dialect_map)

train_df = pd.DataFrame({
    'text': X_text_train.values,
    'label': y_train.values,
    'dialect_id': df_augmented.loc[X_text_train.index, 'dialect_id'].values
})

test_df = pd.DataFrame({
    'text': X_text_test.values,
    'label': y_test.values,
    'dialect_id': df_augmented.loc[X_text_test.index, 'dialect_id'].values
})

print(f"\n Données BERT:")
print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
print(f"  Dialectes: {list(dialect_map.keys())}")

# Tokenizer
model_checkpoint = "alger-ia/dziribert"
print(f"\nChargement: {model_checkpoint}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print("✓ Tokenizer chargé")
except Exception as e:
    print(f"DziriBERT non disponible ({e}), utilisation d'un modèle alternatif")
    model_checkpoint = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Custom Dataset
class ArabiziDataset(TorchDataset):
    def __init__(self, texts, labels, dialect_ids, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels.values, dtype=torch.long)
        self.dialect_ids = torch.tensor(dialect_ids.values, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'dialect_ids': self.dialect_ids[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# Créer datasets
train_dataset = ArabiziDataset(
    train_df['text'],
    train_df['label'],
    train_df['dialect_id'],
    tokenizer
)
test_dataset = ArabiziDataset(
    test_df['text'],
    test_df['label'],
    test_df['dialect_id'],
    tokenizer
)

# Initialiser le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_with_dialect = BERTWithDialect(
    bert_model_name=model_checkpoint,
    num_labels=2,
    num_dialects=len(dialect_map),
    dropout=0.3
).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_bert_dialect',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,
    report_to="none",
)

# Custom Trainer avec weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Extract logits from output dictionary
        logits = outputs['logits']

        # Loss avec poids pour gérer le déséquilibre
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.0], device=logits.device)
        )

        loss = loss_fct(logits, labels)

        # Return in the format expected by Trainer
        return (loss, outputs) if return_outputs else loss

# Fonction de métrique
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if len(predictions.shape) > 2:
        predictions = predictions.reshape(-1, predictions.shape[-1])

    preds = np.argmax(predictions, axis=1)

    min_len = min(len(labels), len(preds))
    labels = labels[:min_len]
    preds = preds[:min_len]

    f1 = f1_score(labels, preds, average='macro')
    return {'f1': f1}

trainer = WeightedTrainer(
    model=model_with_dialect,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Entraîner
print("\n Démarrage de l'entraînement")
trainer.train()

# Prédictions
print("\n Génération des prédictions")
predictions = trainer.predict(test_dataset)
y_pred_bert = np.argmax(predictions.predictions, axis=1)

# Évaluation
bert_results = evaluate_model(y_test, y_pred_bert)
bert_results['Model'] = 'BERT+Dialect'

print(f"\nBERT Results:")
print(f"  F1 Offensive: {bert_results['F1_1']:.4f}")
print(f"  Macro F1: {bert_results['Macro_F1']:.4f}")

