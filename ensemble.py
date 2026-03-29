#ENSEMBLE & STACKING (META-CLASSIFIER)

print("META-CLASSIFIER (STACKING)")


meta_features_train = []
meta_features_test = []

# Prédictions sur train (avec cross-validation pour éviter overfitting)
from sklearn.model_selection import cross_val_predict

for name, model_dict in trained_models.items():
    model = model_dict['model']

    # Train predictions (CV)
    if hasattr(model, 'predict_proba'):
        train_probs = cross_val_predict(model, X_train_combined, y_train,
                                        cv=5, method='predict_proba')[:, 1]
    else:
        train_probs = cross_val_predict(model, X_train_combined, y_train,
                                       cv=5, method='decision_function')

    meta_features_train.append(train_probs)

    # Test predictions
    test_probs = model_dict['y_prob'] if model_dict['y_prob'] is not None else model_dict['y_pred']
    meta_features_test.append(test_probs)

# Ajouter BERT (simulation)
meta_features_train.append(np.random.rand(len(y_train)))
meta_features_test.append(np.random.rand(len(y_test)))

# Créer matrices
X_meta_train = np.column_stack(meta_features_train)
X_meta_test = np.column_stack(meta_features_test)

print(f"Meta-features shape: {X_meta_train.shape}")

# Meta-classifier
meta_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
meta_clf.fit(X_meta_train, y_train)

y_pred_meta = meta_clf.predict(X_meta_test)
y_prob_meta = meta_clf.predict_proba(X_meta_test)[:, 1]

meta_results = evaluate_model(y_test, y_pred_meta, y_prob_meta)
meta_results['Model'] = 'Meta-Classifier (Stacking)'

print(f"\nMeta-Classifier Results:")
print(f"  F1 Offensive: {meta_results['F1_1']:.4f}")
print(f"  Macro F1: {meta_results['Macro_F1']:.4f}")
print(f"  AUC: {meta_results['AUC']:.4f}")

# Importance des modèles dans le meta-classifier
feature_importance = np.abs(meta_clf.coef_[0])
model_names = list(trained_models.keys()) + ['BERT+Dialect']

print("\nImportance des modèles dans le meta-classifier:")
for name, imp in sorted(zip(model_names, feature_importance), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.4f}")



