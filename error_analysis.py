#PARTIE 4: ANALYSE D'ERREURS
print("PARTIE 4: ANALYSE D'ERREURS")



# MATRICES DE CONFUSION POUR TOUS LES MODÈLES

print("\nMatrices de confusion\n")

models_to_analyze = {
    'Logistic Regression': trained_models['Logistic Regression']['y_pred'],
    'Linear SVM': trained_models['Linear SVM']['y_pred'],
    'Random Forest': trained_models['Random Forest']['y_pred'],
    'BERT+Dialect': y_pred_bert,
    'Meta-Classifier': y_pred_meta
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (model_name, y_pred) in enumerate(models_to_analyze.items()):
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Offensive', 'Offensive'],
                yticklabels=['Non-Offensive', 'Offensive'],
                ax=axes[idx])
    axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy_score(y_test, y_pred):.3f}')
    axes[idx].set_ylabel('Vraie classe')
    axes[idx].set_xlabel('Classe prédite')

# Cacher le dernier subplot
axes[-1].axis('off')
plt.tight_layout()
plt.show()



# ANALYSE PAR DIALECTE

print("\nAnalyse des erreurs par dialecte\n")

def analyze_errors_by_dialect(y_true, y_pred, dialects, model_name):
    """Calcule FP et FN par dialecte"""
    results = []

    for dialect in dialects.unique():
        mask = dialects == dialect
        y_true_d = y_true[mask]
        y_pred_d = y_pred[mask]

        # Calcul des métriques
        tn = ((y_true_d == 0) & (y_pred_d == 0)).sum()
        fp = ((y_true_d == 0) & (y_pred_d == 1)).sum()
        fn = ((y_true_d == 1) & (y_pred_d == 0)).sum()
        tp = ((y_true_d == 1) & (y_pred_d == 1)).sum()

        total_neg = tn + fp
        total_pos = fn + tp

        fp_rate = fp / total_neg if total_neg > 0 else 0
        fn_rate = fn / total_pos if total_pos > 0 else 0

        results.append({
            'Model': model_name,
            'Dialect': dialect,
            'FP': fp,
            'FN': fn,
            'FP_Rate': fp_rate * 100,
            'FN_Rate': fn_rate * 100,
            'Total_Samples': len(y_true_d)
        })

    return pd.DataFrame(results)

# Analyser tous les modèles
all_dialect_results = []

for model_name, y_pred in models_to_analyze.items():
    df_results = analyze_errors_by_dialect(y_test.values, y_pred, d_test, model_name)
    all_dialect_results.append(df_results)

dialect_analysis = pd.concat(all_dialect_results, ignore_index=True)

print("\nTaux de Faux Positifs et Faux Négatifs par Dialecte:")
print(dialect_analysis.to_string(index=False))

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Faux Positifs
fp_pivot = dialect_analysis.pivot(index='Model', columns='Dialect', values='FP_Rate')
fp_pivot.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Taux de Faux Positifs par Dialecte (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Taux (%)')
ax1.set_xlabel('Modèle')
ax1.legend(title='Dialecte')
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Faux Négatifs
fn_pivot = dialect_analysis.pivot(index='Model', columns='Dialect', values='FN_Rate')
fn_pivot.plot(kind='bar', ax=ax2, width=0.8)
ax2.set_title('Taux de Faux Négatifs par Dialecte (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Taux (%)')
ax2.set_xlabel('Modèle')
ax2.legend(title='Dialecte')
ax2.grid(axis='y', alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')



# SÉLECTION ET ANNOTATION DES EXEMPLES MAL CLASSÉS

print("\nÉchantillonnage des erreurs pour analyse qualitative\n")

def extract_misclassified_examples(y_true, y_pred, X_text, model_name, n_samples=30):
    """Extrait des exemples mal classés"""
    y_true_arr = y_true.values

    # Identifier les erreurs
    fn_mask = (y_true_arr == 1) & (y_pred == 0)  # Faux négatifs
    fp_mask = (y_true_arr == 0) & (y_pred == 1)  # Faux positifs

    fn_indices = np.where(fn_mask)[0]
    fp_indices = np.where(fp_mask)[0]

    # Échantillonner
    n_fn = min(20, len(fn_indices))
    n_fp = min(10, len(fp_indices))

    fn_sample = np.random.choice(fn_indices, n_fn, replace=False) if len(fn_indices) > 0 else []
    fp_sample = np.random.choice(fp_indices, n_fp, replace=False) if len(fp_indices) > 0 else []

    examples = []

    for idx in fn_sample:
        examples.append({
            'Model': model_name,
            'Text': X_text.iloc[idx],
            'True_Label': 'Offensive',
            'Pred_Label': 'Non-Offensive',
            'Error_Type': 'Faux Négatif',
            'Offense_Type': 'À annoter'  # Sera annoté manuellement
        })

    for idx in fp_sample:
        examples.append({
            'Model': model_name,
            'Text': X_text.iloc[idx],
            'True_Label': 'Non-Offensive',
            'Pred_Label': 'Offensive',
            'Error_Type': 'Faux Positif',
            'Offense_Type': 'N/A'
        })

    return examples

# Extraire exemples pour TOUS les modèles classiques + BERT
all_error_examples = []
for model_name, y_pred in models_to_analyze.items():
    examples = extract_misclassified_examples(y_test, y_pred, X_text_test, model_name)
    all_error_examples.extend(examples)

error_examples_df = pd.DataFrame(all_error_examples)

print(f"Nombre total d'exemples d'erreurs extraits: {len(error_examples_df)}")
print(f"  Faux négatifs: {(error_examples_df['Error_Type'] == 'Faux Négatif').sum()}")
print(f"  Faux positifs: {(error_examples_df['Error_Type'] == 'Faux Positif').sum()}")


# Afficher quelques exemples
print("\nExemples de Faux Négatifs (offensive → non-offensive):")
print("-" * 80)
fn_examples = error_examples_df[error_examples_df['Error_Type'] == 'Faux Négatif'].head(10)
for idx, row in fn_examples.iterrows():
    print(f"\n[{row['Model']}]")
    print(f"Texte: {row['Text']}")
    print(f"Vraie: {row['True_Label']} | Prédite: {row['Pred_Label']}")

#COMPARAISON: BERT vs TOUS LES MODÈLES CLASSIQUES

print("\nAnalyse comparative: BERT vs Tous les Modèles Classiques\n")

def compare_model_predictions(y_true, y_pred_1, y_pred_2, model_1_name, model_2_name):
    """Compare où deux modèles réussissent/échouent différemment"""
    comparison = []

    for i in range(len(y_true)):
        true_label = y_true.iloc[i]
        pred_1 = y_pred_1[i]
        pred_2 = y_pred_2[i]

        if pred_1 == true_label and pred_2 != true_label:
            comparison.append(f'{model_1_name} réussit, {model_2_name} échoue')
        elif pred_1 != true_label and pred_2 == true_label:
            comparison.append(f'{model_1_name} échoue, {model_2_name} réussit')
        elif pred_1 == true_label and pred_2 == true_label:
            comparison.append('Les deux réussissent')
        else:
            comparison.append('Les deux échouent')

    return comparison

# Comparer BERT vs TOUS les modèles classiques
classic_models = ['Logistic Regression', 'Linear SVM', 'Complement NB', 'Random Forest']

for classic_model in classic_models:
    print(f"\n{'='*80}")
    print(f"COMPARAISON: {classic_model} vs BERT+Dialect")
    print('='*80)

    comparison = compare_model_predictions(
        y_test,
        trained_models[classic_model]['y_pred'],
        y_pred_bert,
        classic_model,
        'BERT+Dialect'
    )

    comparison_counts = pd.Series(comparison).value_counts()
    print("\nRépartition des prédictions:")
    print(comparison_counts)

    # Cas spécifiques: où BERT réussit et le modèle classique échoue
    bert_wins_mask = [c == f'{classic_model} échoue, BERT+Dialect réussit' for c in comparison]
    classic_wins_mask = [c == f'{classic_model} réussit, BERT+Dialect échoue' for c in comparison]

    print(f"\nCas où BERT réussit et {classic_model} échoue: {sum(bert_wins_mask)}")
    if sum(bert_wins_mask) > 0:
        print("Exemples:")
        bert_win_texts = X_text_test[bert_wins_mask].head(5)
        bert_win_labels = y_test[bert_wins_mask].head(5)
        for text, label in zip(bert_win_texts, bert_win_labels):
            label_str = "Offensive" if label == 1 else "Non-Offensive"
            print(f"  [{label_str}] {text}")

    print(f"\nCas où {classic_model} réussit et BERT échoue: {sum(classic_wins_mask)}")
    if sum(classic_wins_mask) > 0:
        print("Exemples:")
        classic_win_texts = X_text_test[classic_wins_mask].head(5)
        classic_win_labels = y_test[classic_wins_mask].head(5)
        for text, label in zip(classic_win_texts, classic_win_labels):
            label_str = "Offensive" if label == 1 else "Non-Offensive"
            print(f"  [{label_str}] {text}")

# Visualisation comparative
print("\n--- Visualisation: BERT vs Modèles Classiques ---")

comparison_data = []
for classic_model in classic_models:
    comparison = compare_model_predictions(
        y_test,
        trained_models[classic_model]['y_pred'],
        y_pred_bert,
        classic_model,
        'BERT+Dialect'
    )

    counts = pd.Series(comparison).value_counts()
    comparison_data.append({
        'Modèle Classique': classic_model,
        'Classique réussit, BERT échoue': counts.get(f'{classic_model} réussit, BERT+Dialect échoue', 0),
        'BERT réussit, Classique échoue': counts.get(f'{classic_model} échoue, BERT+Dialect réussit', 0),
        'Les deux réussissent': counts.get('Les deux réussissent', 0),
        'Les deux échouent': counts.get('Les deux échouent', 0)
    })

comparison_df = pd.DataFrame(comparison_data)

# Graphique comparatif
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(classic_models))
width = 0.2

categories = ['Classique réussit, BERT échoue', 'BERT réussit, Classique échoue',
              'Les deux réussissent', 'Les deux échouent']
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D']

for i, (category, color) in enumerate(zip(categories, colors)):
    values = comparison_df[category].values
    ax.bar(x + i*width, values, width, label=category, color=color, alpha=0.8)

ax.set_xlabel('Modèles Classiques', fontsize=12, fontweight='bold')
ax.set_ylabel('Nombre de cas', fontsize=12, fontweight='bold')
ax.set_title('Comparaison détaillée: Modèles Classiques vs BERT+Dialect',
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(classic_models, rotation=15, ha='right')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Tableau récapitulatif
print("\nTableau récapitulatif des comparaisons:")
print(comparison_df.to_string(index=False))



# 6. COURBES DE PRÉCISION ET PERTE (BERT)

print("\nCourbes d'entraînement BERT\n")

# Récupérer l'historique d'entraînement
log_history = trainer.state.log_history

# Extraire les métriques
train_loss = []
eval_loss = []
eval_f1 = []
epochs = []

for entry in log_history:
    if 'loss' in entry and 'epoch' in entry:
        train_loss.append(entry['loss'])
    if 'eval_loss' in entry:
        eval_loss.append(entry['eval_loss'])
        eval_f1.append(entry.get('eval_f1', 0))
        epochs.append(entry['epoch'])

# Tracer les courbes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Courbes de perte
if train_loss:
    ax1.plot(train_loss, label='Train Loss', marker='o', linewidth=2)
if eval_loss:
    ax1.plot(epochs, eval_loss, label='Validation Loss', marker='s', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Courbes de perte - BERT+Dialect', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Courbe F1
if eval_f1:
    ax2.plot(epochs, eval_f1, label='Validation F1', marker='o', linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Score F1 sur Validation - BERT+Dialect', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bert_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Courbes d'entraînement sauvegardées: bert_training_curves.png")



# 7. RÉSUMÉ DE L'ANALYSE D'ERREURS

print("\n" + "="*80)
print("RÉSUMÉ DE L'ANALYSE D'ERREURS")
print("="*80)

summary_stats = []
for model_name, y_pred in models_to_analyze.items():
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    summary_stats.append({
        'Modèle': model_name,
        'Vrais Négatifs': tn,
        'Faux Positifs': fp,
        'Faux Négatifs': fn,
        'Vrais Positifs': tp,
        'Taux FP (%)': fp / (tn + fp) * 100 if (tn + fp) > 0 else 0,
        'Taux FN (%)': fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    })

summary_df = pd.DataFrame(summary_stats)
print("\n" + summary_df.to_string(index=False))

