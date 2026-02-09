import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_autism_model():
    df = pd.read_csv('data/cleaned_autism_data.csv')
    
    # 'Class/ASD' is our target variable
    X = df.drop('Class/ASD', axis=1)
    y = df['Class/ASD']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and current column list
    model_data = {'model': model, 'columns': list(X.columns)}
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("Model trained successfully with columns:", list(X.columns))

if __name__ == "__main__":
    train_autism_model()