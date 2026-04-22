import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, confusion_matrix, auc

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'lines.color': 'black',
    'lines.linewidth': 1.5,
    'font.size': 10
})

# Phase 1: Data Loading & Preprocessing
# download the csv file from the dataset link
df = pd.read_csv(data)

# Scale Time and Amount
scaler = StandardScaler()
df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df = df.drop(['Time', 'Amount'], axis=1)

# Define Features (X) and Target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Initial split to hold out a pure Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Phase 2: Cross-Validation & Modeling
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest with balanced class weight
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
scoring_metrics = ['average_precision', 'recall', 'precision']

cv_results = cross_validate(rf_model, X_train, y_train, cv=cv_strategy, scoring=scoring_metrics)

print(f"Mean PR-AUC:   {cv_results['test_average_precision'].mean():.4f} (+/- {cv_results['test_average_precision'].std():.4f})")
print(f"Mean Recall:   {cv_results['test_recall'].mean():.4f}")

# Phase 3: Final Training & Business Strategy
rf_model.fit(X_train, y_train)

# Get the probability of fraud for the test set
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Custom Business Threshold
custom_threshold = 0.35 
y_pred_custom = (y_prob_rf >= custom_threshold).astype(int)

print(f"\nClassification Report (Custom Threshold = {custom_threshold}):\n")
print(classification_report(y_test, y_pred_custom))


# Phase 4: Data Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_rf)
pr_auc_score = auc(recall, precision)

axes[0].plot(recall, precision, color='black', label=f'PR Curve (AUC = {pr_auc_score:.2f})')
axes[0].set_xlabel('Recall (Fraud Caught)')
axes[0].set_ylabel('Precision (True Fraud Ratio)')
axes[0].set_title('Precision-Recall Curve')
axes[0].grid(True, linestyle='--', alpha=0.5, color='gray')
axes[0].legend(loc='lower left')

# Plot 2: Confusion Matrix for the Custom Threshold
cm = confusion_matrix(y_test, y_pred_custom)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', cbar=False, ax=axes[1],
            linewidths=1, linecolor='black', 
            xticklabels=['Normal (0)', 'Fraud (1)'], 
            yticklabels=['Normal (0)', 'Fraud (1)'])

axes[1].set_ylabel('Actual Transaction')
axes[1].set_xlabel(f'Predicted Transaction (Threshold={custom_threshold})')
axes[1].set_title('Business Impact: Confusion Matrix')

plt.tight_layout()
plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
plt.show()
