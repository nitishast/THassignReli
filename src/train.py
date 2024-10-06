import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score  
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import shap
import pickle
from sklearn.preprocessing import LabelEncoder
import os
from model_dispatcher import models 


def load_data(file_path):
    """Load and preprocess the data."""
    df = pd.read_csv(file_path,low_memory=False)
    # df = df.sample(frac=0.9, random_state=42)
    df['Date'] = pd.to_datetime(df['Date'])
    return df
    
def prepare_data(df):
    """Prepare the data for modeling."""
    # df['Label_5'] = df['Label'].map(simplify_labels)
    
    # # Check if the label encoder file exists
    # if os.path.exists('label_encoder.pkl'):
    #     with open('label_encoder.pkl', 'rb') as f:
    #         le = pickle.load(f)
    # else:
    #     # Create and fit the label encoder
    #     le = LabelEncoder()
    #     le.fit(df['Label_5'])
    #     # Save the label encoder
    #     with open('label_encoder.pkl', 'wb') as f:
    #         pickle.dump(le, f)
    
    # df['Label_5_Encoded'] = le.transform(df['Label_5'])
    print("\n" + "="*50 + "\n")
    print("label distribtion before enoding",df['Label'].value_counts(dropna=False))
    print("\n" + "="*50 + "\n")
    le = LabelEncoder()
    le.fit(df['Label'])
    df['Label_7_Encoded'] = le.transform(df['Label'])
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Log_Return', 'SMA_50', 'EMA_20', 
                'RSI', 'RSI_14', 'RSI_28', 'BB_upper', 'BB_lower', 'OBV', 
                'Close_Lag_1', 'Volume_Lag_1', 'Close_Lag_2', 'Volume_Lag_2', 
                'Close_Lag_3', 'Volume_Lag_3', 'Close_Lag_4', 'Volume_Lag_4', 
                'Close_Lag_5', 'Volume_Lag_5', 'MACD', 'MACD_Signal', 'BB_Pct', 
                'Volatility', 'ATR', 'OBV_SMA_ratio', 'High_Low_Range']
    
    X = df[features]
    y = df['Label_7_Encoded']
    print("\n" + "="*50 + "\n")
    print("label distribtion after enoding", y.value_counts())
    print("\n" + "="*50 + "\n")
    
    return X, y, le, df

def create_folds(X, y, n_splits=5):
    """Create time series folds."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = list(tscv.split(X))
    return folds

def train_models(X, y, folds):
    """Train and evaluate multiple models."""
    performances = {}
    for name, model in models.items():
        print(f"Training {name}...")
        performance = evaluate_model(model, X, y, folds)
        performances[name] = performance
        
        # Print performance for each fold
        for fold_result in performance:
            print(f"  Fold {fold_result['fold']}:")
            print(f"    Accuracy: {fold_result['accuracy']:.4f}")
            print(f"    F1 Score: {fold_result['f1_score']:.4f}")
        
        # Calculate and print average performance
        avg_accuracy = np.mean([fold['accuracy'] for fold in performance])
        avg_f1 = np.mean([fold['f1_score'] for fold in performance])
        print(f"  Average Performance:")
        print(f"    Accuracy: {avg_accuracy:.4f}")
        print(f"    F1 Score: {avg_f1:.4f}")
        print("------------------------")
    
    best_model = max(models.items(), key=lambda x: np.mean([fold['accuracy'] for fold in performances[x[0]]]))[1]
    # try:
    #     model_name = best_model.__class__.__name__
    #     model_path = f'models/{model_name}.pkl'
    #     with open(model_path, 'wb') as f:
    #         pickle.dump(best_model, f)
    #     print(f"Best model saved as {model_path}")
    # except Exception as e:
    #     print(f"Failed to save the best model: {e}")
    return best_model, performances

def evaluate_model(model, X, y, folds):
    """Evaluate the model using pre-defined folds."""
    results = []
    fold_results = []
    
    for fold, (train_index, test_index) in enumerate(folds, 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        fold_result = {
            'fold': fold,
            'accuracy': accuracy,
            'f1_score': f1
        }
        fold_results.append(fold_result)
        
        print("Fold Results:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\n" + "="*50 + "\n")
    
    avg_accuracy = np.mean([fold['accuracy'] for fold in fold_results])
    avg_f1 = np.mean([fold['f1_score'] for fold in fold_results])
    
    avg_performance = {
        'accuracy': avg_accuracy,
        'avgf1' : avg_f1,
        'macro avg': {
            'precision': np.mean([report['macro avg']['precision'] for report in results]),
            'recall': np.mean([report['macro avg']['recall'] for report in results]),
            'f1-score': np.mean([report['macro avg']['f1-score'] for report in results])
        }
    }

    print("Average Performance Across Folds:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Macro Avg Precision: {avg_performance['macro avg']['precision']:.4f}")
    print(f"Macro Avg Recall: {avg_performance['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1-score: {avg_performance['macro avg']['f1-score']:.4f}")
    
    return fold_results

def main():
    # Load and prepare data
    df = load_data('/Users/nitastha/Desktop/NitishFiles/Projects/Newrelic/labeled_data.csv')
    X, y, le, df = prepare_data(df)

    # Create folds
    folds = create_folds(X, y)
    # Train and evaluate models
    best_model, performances = train_models(X, y, folds)
    # Make predictions
    predictions = best_model.predict(X)
    # Add prediction column to DataFrame
    df['predicted_7'] = predictions
    # Save the DataFrame with predictions to a CSV file
    df.to_csv('predicted.csv', index=False)

if __name__ == "__main__":
    main()
