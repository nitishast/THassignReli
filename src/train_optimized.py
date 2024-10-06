import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from model_dispatcher import models

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

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['label_5'] = df['Label'].map(simplify_labels)
    
    le = LabelEncoder()
    df['label_5_encoded'] = le.fit_transform(df['label_5'])
    
    return df, le

def evaluate_model(model, X, y):
    results = []
    tscv = TimeSeriesSplit(n_splits=2)
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'accuracy': accuracy,
            'f1_score': f1
        })
        
        print("Fold Results:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("="*50)
    
    avg_performance = {
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'f1_score': np.mean([r['f1_score'] for r in results])
    }
    return avg_performance

def train_models(X, y):
    performances = {}
    best_model = None
    best_performance = 0
    for name, model in models.items():
        print(f"Training {name}...")
        performance = evaluate_model(model, X, y)
        performances[name] = performance
        print(f"{name} Average Performance:")
        print(f"Accuracy: {performance['accuracy']:.4f}")
        print(f"F1 Score: {performance['f1_score']:.4f}")
        print("="*50)
        
        if performance['accuracy'] > best_performance:
            best_performance = performance['accuracy']
            best_model = model.fit(X, y)  # Fit the model on all data
    
    return best_model, performances

def save_model(model, file_path):
    import pickle
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {file_path}")

def main():
    # Load and prepare data
    df, le = load_and_prepare_data('/Users/nitastha/Desktop/NitishFiles/Projects/Newrelic/predicted.csv')
    
    # Prepare features and target
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Log_Return', 'SMA_50', 'EMA_20', 
                'RSI', 'RSI_14', 'RSI_28', 'BB_upper', 'BB_lower', 'OBV', 
                'Close_Lag_1', 'Volume_Lag_1', 'Close_Lag_2', 'Volume_Lag_2', 
                'Close_Lag_3', 'Volume_Lag_3', 'Close_Lag_4', 'Volume_Lag_4', 
                'Close_Lag_5', 'Volume_Lag_5', 'MACD', 'MACD_Signal', 'BB_Pct', 
                'Volatility', 'ATR', 'OBV_SMA_ratio', 'High_Low_Range']
    
    X = df[features]
    y = df['label_5_encoded']

    # Train models and get the best one
    best_model, performances = train_models(X, y)
    
    # Save the best model
    save_model(best_model, 'best_model.pkl')

    # Make predictions using the best model
    predictions = best_model.predict(X)
    df['predicted_5'] = le.inverse_transform(predictions)
    # df.to_csv('predicted_with_5_class.csv', index=False)
    # Save results
    # df[['Label_7_Encoded', 'predicted_7', 'label_5_encoded', 'predicted_5']].to_csv('predictions_comparison.csv', index=False)
    
    # Print final results
    print("\nFinal Results:")
    print("\nClassification Report (5-class):")
    print(classification_report(df['label_5'], df['predicted_5']))
    print("\nConfusion Matrix (5-class):")
    print(confusion_matrix(df['label_5'], df['predicted_5']))
    
    # Compare distributions
    print("\nActual 5-class Distribution:")
    print(df['label_5'].value_counts(normalize=True))
    print("\nPredicted 5-class Distribution:")
    print(df['predicted_5'].value_counts(normalize=True))

if __name__ == "__main__":
    main()