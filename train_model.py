import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def treat_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound,
                 np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df

def train_and_evaluate():
    print("Loading data...")
    df = pd.read_csv('flood_dataset.csv')
    
    print("Handling missing values (if any)...")
    df = df.dropna()
    
    print("Treating outliers...")
    numerical_columns = ['AnnualRainfall', 'CloudVisibility', 'Temperature', 'Humidity', 'SeasonalRainfall']
    for col in numerical_columns:
        df = treat_outliers(df, col)
        
    print("Splitting data...")
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling data...")
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    # Save the scaler
    joblib.dump(sc, 'transform.save')
    print("Saved StandardScaler to transform.save")
    
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        if name == "XGBoost":
            best_model = model
            best_accuracy = acc
            
    print(f"\nSelected XGBoost with accuracy: {best_accuracy:.4f}")
    joblib.dump(best_model, 'floods.save')
    print("Saved XGBoost model to floods.save")

if __name__ == "__main__":
    train_and_evaluate()
