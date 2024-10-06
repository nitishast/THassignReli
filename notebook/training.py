# %%
import pandas as pd
import numpy as np
import talib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Assume X and y are your feature matrix and labels from the previous step
# read x, y 
X = pd.read_csv('prepared_data.csv')
y = pd.read_csv('y.csv')

X.head()

# %%
tscv = TimeSeriesSplit(n_splits=5)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# %%
results = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append(report)

    print("Fold Results:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + "="*50 + "\n")

# %%
avg_performance = {
    'accuracy': np.mean([r['accuracy'] for r in results]),
    'macro avg': {
        'precision': np.mean([r['macro avg']['precision'] for r in results]),
        'recall': np.mean([r['macro avg']['recall'] for r in results]),
        'f1-score': np.mean([r['macro avg']['f1-score'] for r in results])
    }
}

print("Average Performance Across Folds:")
print(f"Accuracy: {avg_performance['accuracy']:.4f}")
print(f"Macro Avg Precision: {avg_performance['macro avg']['precision']:.4f}")
print(f"Macro Avg Recall: {avg_performance['macro avg']['recall']:.4f}")
print(f"Macro Avg F1-score: {avg_performance['macro avg']['f1-score']:.4f}")

# %%
def buy_and_hold(y):
    # Assume 'buy and hold' always predicts the majority class
    majority_class = y.mode().iloc[0]
    return [majority_class] * len(y)

bnh_results = []

for _, test_index in tscv.split(X):
    y_test = y.iloc[test_index]
    bnh_pred = buy_and_hold(y_test)
    bnh_report = classification_report(y_test, bnh_pred, output_dict=True)
    bnh_results.append(bnh_report)

bnh_avg_performance = {
    'accuracy': np.mean([r['accuracy'] for r in bnh_results]),
    'macro avg': {
        'f1-score': np.mean([r['macro avg']['f1-score'] for r in bnh_results])
    }
}

print("\nBuy-and-Hold Baseline Performance:")
print(f"Accuracy: {bnh_avg_performance['accuracy']:.4f}")
print(f"Macro Avg F1-score: {bnh_avg_performance['macro avg']['f1-score']:.4f}")

# %%
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# %% [markdown]
# Thank you for providing these results. Let's analyze them and consider our next steps:
# 
# 1. Model Performance:
#    The accuracy of 0.3170 (31.70%) across folds is relatively low, considering we have 8 classes. This suggests our model is struggling to consistently predict the correct label. The low precision, recall, and F1-scores further confirm this.
# 
# 2. Feature Importance:
#    - OBV (On-Balance Volume) is by far the most important feature, which suggests volume is a key indicator for stock movement.
#    - Percentage from 52-week high and low are also significant, indicating that a stock's position relative to its recent price range is important.
#    - ATR (Average True Range) and SMA_50 (50-day Simple Moving Average) round out the top 5, showing that volatility and medium-term trends are also key factors.
# 
# Given these results, here are some steps we can take to improve our analysis and better address the original task:
# 
# 1. Simplify Classification:
#    Our current 8-class system might be too granular. Consider reducing to 3-5 classes (e.g., Strong Up, Weak Up, Flat, Weak Down, Strong Down) to potentially improve model performance.
# 
# 2. Feature Engineering:
#    - Focus on creating more features based on volume and price extremes, given the importance of OBV and 52-week high/low metrics.
#    - Consider creating interaction features between top predictors.
# 
# 3. Try Different Models:
#    - Implement a Gradient Boosting Classifier (like XGBoost or LightGBM) which often performs well on financial data.
#    - Consider using a multi-layer perceptron (neural network) which might capture more complex patterns.
# 
# 4. Analyze Synchronous Movement:
#    Despite the model's low accuracy, we can still use it to group stocks and analyze their correlations. This directly addresses the original hypothesis about stocks moving in sync.
# 
# 5. Sector Analysis:
#    Incorporate sector information to see if certain sectors tend to move together more than others.
# 
# This code implements the suggested improvements and directly addresses the original hypothesis by analyzing synchronous movement within predicted groups. The results from this analysis will provide insights into whether stocks with similar predicted movements actually show correlated returns.
# 
# Would you like to run this analysis or focus on any specific part of it?

# %%
# from scipy.stats import pearsonr

# def analyze_sync_movement(df, predictions):
#     df['Predicted_Label'] = predictions
    
#     # Group stocks by predicted label
#     groups = df.groupby('Predicted_Label')
    
#     for label, group in groups:
#         print(f"Analyzing group: {label}")
        
#         # Calculate pairwise correlations of returns within the group
#         stocks = group['Ticker'].unique()
#         correlations = []
        
#         for i in range(len(stocks)):
#             for j in range(i+1, len(stocks)):
#                 stock1 = group[group['Ticker'] == stocks[i]]['Returns']
#                 stock2 = group[group['Ticker'] == stocks[j]]['Returns']
#                 corr, _ = pearsonr(stock1, stock2)
#                 correlations.append(corr)
        
#         avg_correlation = np.mean(correlations)
#         print(f"Average correlation within group: {avg_correlation:.4f}")
#         print("="*50)

# # Assuming 'predictions' are the labels predicted by our model
# analyze_sync_movement(df, predictions)

# %%
# load label encoder from pickle and convert lable_encoded column in y back to originla label
import pickle
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
y_original = le.inverse_transform(y)
y_original = y.apply(le.inverse_transform)


# %%
# 1. Simplify Classification
def simplify_labels(label):
    if label in ['Weak Up', 'Moderate Up']:
        return 'Weak Up'
    elif label in ['Weak Down', 'Moderate Down']:
        return 'Weak Down'
    elif label in ['Strong Up']:
        return 'Strong Up'
    elif label in ['Strong Down']:
        return 'Strong Down'
    else:
        return 'Flat'

y_simplified = y_original.map(simplify_labels)
y_simplified.Label_Encoded.value_counts(normalize=True)


# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_simplified_encoded = le.fit_transform(y_simplified['Label_Encoded'])
y = y_simplified_encoded

# %%
y = pd.DataFrame(y_simplified_encoded)
y

# %%
# 2. Feature Engineering
X['OBV_SMA_ratio'] = X['OBV'] / X['SMA_50']
X['High_Low_Range'] = X['High'] - X['Low']

# %%
# 3. Try Different Models
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

xgb_model = XGBClassifier(random_state=42)
mlp_model = MLPClassifier(random_state=42, max_iter=1000)

# Function to evaluate model
def evaluate_model(model, X, y):
    results = []
    for train_index, test_index in TimeSeriesSplit(n_splits=5).split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        results.append(report)
    
    avg_performance = {
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'macro avg': {
            'f1-score': np.mean([r['macro avg']['f1-score'] for r in results])
        }
    }
    return avg_performance

# Evaluate models
rf_performance = evaluate_model(RandomForestClassifier(random_state=42), X, y)
xgb_performance = evaluate_model(xgb_model, X, y)
mlp_performance = evaluate_model(mlp_model, X, y)

print("Random Forest Performance:", rf_performance)
print("XGBoost Performance:", xgb_performance)
print("Neural Network Performance:", mlp_performance)


# %%
# 4. Analyze Synchronous Movement (using the best performing model)
best_model = max([rf_model, xgb_model, mlp_model], key=lambda m: evaluate_model(m, X, y)['accuracy'])
predictions = best_model.predict(X)

# %%

pd.DataFrame(predictions).value_counts()



# %%
print(le.inverse_transform([0, 1, 2, 3, 4]))

# %%
# Assuming we have a LabelEncoder object called 'le' used earlier
label_names = le.inverse_transform([0, 1, 2, 3, 4])
prediction_counts = pd.Series(predictions).value_counts()
print("Prediction Distribution:")
for label, count in zip(label_names, prediction_counts):
    print(f"{label}: {count}")

# %%
X.head(3)

# %%
# df = pd.read_csv('final_df.csv')

# Add predictions to DataFrame
df['Predicted_Label'] = le.inverse_transform(predictions)

# %%
import pickle
with open('label_encoder.pkl', 'rb') as f:
    le_7 = pickle.load(f)

df['Label_Encoded_inverse'] = le_7.inverse_transform(df['Label_Encoded'])
# y_original = y.apply(le.inverse_transform)
df.head(2)

# %%
def simplify_labels(label):
    if label in ['Weak Up', 'Moderate Up']:
        return 'Weak Up'
    elif label in ['Weak Down', 'Moderate Down']:
        return 'Weak Down'
    elif label in ['Strong Up']:
        return 'Strong Up'
    elif label in ['Strong Down']:
        return 'Strong Down'
    else:
        return 'Flat'

df['Label_5'] = df['Label_Encoded_inverse'].map(simplify_labels)
df.head()

# %%
df.columns

# %%
df['Predicted_Label'].value_counts()


# %%
df['Label_5'].value_counts()

# %%
## compare accuracy between columns predicted and label_5 in df
def compare_accuracy(df):
    """
    Compares the accuracy between the predicted and label_5 columns in the given DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the predicted and label_5 columns.
        
    Returns:
        float: The accuracy score, calculated as the number of rows where predicted and label_5 match divided by the total number of rows.
    """
    correct = (df['Predicted_Label'] == df['Label_5']).sum()
    total = len(df)
    accuracy = correct / total
    return accuracy

print(compare_accuracy(df))

# %%


# %%
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Assuming df is your DataFrame with 'Predicted_Label' and 'Label_5' columns

# Confusion Matrix
conf_matrix = confusion_matrix(df['Label_5'], df['Predicted_Label'])
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(df['Label_5'], df['Predicted_Label'])
print("\nClassification Report:")
print(class_report)

# Accuracy
accuracy = accuracy_score(df['Label_5'], df['Predicted_Label'])
print(f"\nAccuracy: {accuracy:.4f}")

# Precision
precision = precision_score(df['Label_5'], df['Predicted_Label'], average='macro')
print(f"Precision: {precision:.4f}")

# Recall
recall = recall_score(df['Label_5'], df['Predicted_Label'], average='macro')
print(f"Recall: {recall:.4f}")

# F1 Score
f1 = f1_score(df['Label_5'], df['Predicted_Label'], average='macro')
print(f"F1 Score: {f1:.4f}")

# %%
# Add predictions to DataFrame
df['Predicted_Label'] = le.inverse_transform(predictions)

# Now we can proceed with clustering
# ... (clustering code as provided earlier) ...

# In the analyze_cluster_movement function, use 'Predicted_Label' instead of 'Label'
def analyze_cluster_movement(df):
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        print(f"\nAnalyzing Cluster {cluster}")
        
        # Calculate average correlation of returns within the cluster
        stocks = cluster_data['Ticker'].unique()
        correlations = []
        for i in range(len(stocks)):
            for j in range(i+1, len(stocks)):
                stock1 = cluster_data[cluster_data['Ticker'] == stocks[i]]['Returns']
                stock2 = cluster_data[cluster_data['Ticker'] == stocks[j]]['Returns']
                corr, _ = pearsonr(stock1, stock2)
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        print(f"Average correlation within cluster: {avg_correlation:.4f}")
        
        # Show distribution of predicted labels within this cluster
        label_dist = cluster_data['Predicted_Label'].value_counts(normalize=True)
        print("Predicted Label Distribution in Cluster:")
        print(label_dist)
        
        # Show top features for this cluster
        cluster_center = kmeans.cluster_centers_[cluster]
        feature_importance = pd.Series(cluster_center, index=X.columns).sort_values(ascending=False)
        print("Top features for this cluster:")
        print(feature_importance.head())
        print("="*50)

# Run the clustering analysis
analyze_cluster_movement(df)

# %%
# def analyze_sync_movement(df, predictions):
#     df['Predicted_Label'] = predictions
#     groups = df.groupby('Predicted_Label')

#     for label, group in groups:
#         print(f"Analyzing group: {label}")
#         stocks = group['Ticker'].unique()
#         correlations = []
        
#         for i in range(len(stocks)):
#             for j in range(i+1, len(stocks)):
#                 stock1 = group[group['Ticker'] == stocks[i]]['Returns']
#                 stock2 = group[group['Ticker'] == stocks[j]]['Returns']
#                 corr, _ = pearsonr(stock1, stock2)
#                 correlations.append(corr)
        
#         avg_correlation = np.mean(correlations)
#         print(f"Average correlation within group: {avg_correlation:.4f}")
#         print("="*50)

# analyze_sync_movement(df, predictions)

# %%
# Assuming 'model' is your trained model (e.g., RandomForestClassifier or XGBClassifier)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
print(feature_importance.head(15))

# Visualize feature importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
feature_importance.head(15).plot(x='feature', y='importance', kind='bar')
plt.title('Top 15 Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### model performacne overtime

# %%
# Assuming 'df' is your main dataframe with 'Date', 'Predicted_Label', and 'Label_5' columns
df['Date'] = pd.to_datetime(df['Date'])
df['YearMonth'] = df['Date'].dt.to_period('M')

temporal_performance = []

for period in df['YearMonth'].unique():
    period_df = df[df['YearMonth'] == period]
    
    accuracy = accuracy_score(period_df['Label_5'], period_df['Predicted_Label'])
    f1 = f1_score(period_df['Label_5'], period_df['Predicted_Label'], average='weighted')
    
    temporal_performance.append({
        'Period': period,
        'Accuracy': accuracy,
        'F1_Score': f1,
        'Sample_Size': len(period_df)
    })

temporal_df = pd.DataFrame(temporal_performance)

# Plot temporal performance
plt.figure(figsize=(12, 6))
plt.plot(temporal_df['Period'].astype(str), temporal_df['Accuracy'], label='Accuracy')
plt.plot(temporal_df['Period'].astype(str), temporal_df['F1_Score'], label='F1 Score')
plt.title('Model Performance Over Time')
plt.xlabel('Time Period (Year-Month)')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print summary statistics
print("Performance Statistics:")
print(temporal_df[['Accuracy', 'F1_Score']].describe())

# Identify periods with highest and lowest performance
print("\nBest Performing Period:")
print(temporal_df.loc[temporal_df['Accuracy'].idxmax()])
print("\nWorst Performing Period:")
print(temporal_df.loc[temporal_df['Accuracy'].idxmin()])

# %%
class_distribution = df.groupby('YearMonth')['Label_5'].value_counts(normalize=True).unstack()

plt.figure(figsize=(12, 6))
class_distribution.plot(kind='area', stacked=True)
plt.title('Class Distribution Over Time')
plt.xlabel('Time Period')
plt.ylabel('Proportion')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
class_distribution

# %% [markdown]
# ### Out-of-Time Validation:
# 

# %%
df.columns

# %%
last_month = df['YearMonth'].max()
train_df = df[df['YearMonth'] != last_month]
test_df = df[df['YearMonth'] == last_month]

# %%
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_50', 'EMA_20', 
            'RSI', 'RSI_14', 'RSI_28', 'BB_upper', 'BB_lower', 'Volume_MA_5', 
            'Close_Pct_Change', 'Volume_Pct_Change', 'MACD', 'MACD_Signal', 'BB_Pct', 
            'Returns_Rolling_Mean', 'Returns_Rolling_Std', 'ATR', 'OBV', 'Momentum', 
            'DayOfWeek', 'Month', 'Pct_From_52W_High', 'Pct_From_52W_Low'] + \
           [f'Close_Lag_{i}' for i in range(1, 6)] + [f'Volume_Lag_{i}' for i in range(1, 6)]

train_df[features]


# %%
train_df.head(3)

# %%
# Assuming 'le' is the LabelEncoder object used earlier
train_df['Label_5_Encoded'] = le.transform(train_df['Label_5'])
test_df['Label_5_Encoded'] = le.transform(test_df['Label_5'])


# %%
X_train = train_df[features]
y_train = train_df['Label_5_Encoded']
X_test = test_df[features]
y_test = test_df['Label_5_Encoded']

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("Out-of-Time Validation Results:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### Per class analysis

# %%
from sklearn.metrics import precision_recall_fscore_support

class_metrics = precision_recall_fscore_support(y_test, y_pred, average=None, labels=best_model.classes_)
per_class_df = pd.DataFrame({
    'Class': best_model.classes_,
    'Precision': class_metrics[0],
    'Recall': class_metrics[1],
    'F1-Score': class_metrics[2],
    'Support': class_metrics[3]
})

print("Per-Class Performance:")
print(per_class_df)

# Visualize per-class performance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
per_class_df.plot(x='Class', y=['Precision', 'Recall', 'F1-Score'], kind='bar', ax=ax)
plt.title('Per-Class Performance Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# Feature Engineering:
# 
# Let's create some new features based on the successful OBV_SMA_ratio

# %%
def create_ratio_feature(df, numerator, denominator):
    return df[numerator] / df[denominator]

# Create new ratio features
# df['RSI_SMA_ratio'] = create_ratio_feature(df, 'RSI', 'SMA_50')
df['MACD_SMA_ratio'] = create_ratio_feature(df, 'MACD', 'SMA_50')
df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['SMA_50']
df['Price_to_SMA_ratio'] = create_ratio_feature(df, 'Close', 'SMA_50')
df.columns

# %%


# %%

# Add these new features to your feature list
new_features = ['RSI_SMA_ratio', 'MACD_SMA_ratio', 'BB_width', 'Price_to_SMA_ratio']
features += new_features

train_df = df[df['YearMonth'] != last_month]
test_df = df[df['YearMonth'] == last_month]
train_df['Label_5_Encoded'] = le.transform(train_df['Label_5'])
test_df['Label_5_Encoded'] = le.transform(test_df['Label_5'])
# Retrain the model with new features
X_train = train_df[features]
y_train = train_df['Label_5_Encoded']
X_test = test_df[features]
y_test = test_df['Label_5_Encoded']

best_model.fit(X_train, y_train)
y_pred_new = best_model.predict(X_test)

print("Model Performance with New Features:")
print(classification_report(y_test, y_pred_new))

# Compare feature importances
feature_importance_new = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 Features After Engineering:")
print(feature_importance_new.head(15))

# %%
pip install shap

# %%
import shap

# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Visualize the first prediction's explanation
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:])

# Summary plot
shap.summary_plot(shap_values, X_test)

# %%
import shap

# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Check the shapes
print(f"Shape of shap_values: {shap_values[0].shape}")
print(f"Shape of X_test: {X_test.shape}")

# Visualize the first prediction's explanation
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0])

# Summary plot
shap.summary_plot(shap_values, X_test)

# %%
import shap

# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Check the shapes
print(f"Shape of shap_values: {shap_values[0].shape}")
print(f"Shape of X_test: {X_test.shape}")

# Visualize the first prediction's explanation for the first class
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0])

# Summary plot for the first class
shap.summary_plot(shap_values[0], X_test)

# %%
df.to_csv("df_training.csv",index=False)

# %%



