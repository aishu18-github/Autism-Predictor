import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load the cleaned data
df = pd.read_csv('data/cleaned_autism_data.csv')

# 1. CLASS DISTRIBUTION (Pie Chart)
plt.figure(figsize=(8, 6))
df['Class/ASD'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#27ae60', '#e74c3c'], labels=['No ASD', 'ASD Trait'])
plt.title('Distribution of ASD Traits in Dataset')
plt.ylabel('')
plt.savefig('class_distribution.png')
plt.close()

# 2. CORRELATION HEATMAP (Top 10 Features)
plt.figure(figsize=(10, 8))
# Calculating correlation and getting top features related to Class/ASD
corr = df.corr()
top_features = corr.index[abs(corr["Class/ASD"]) > 0.3] # Features with >0.3 correlation
sns.heatmap(df[top_features].corr(), annot=True, cmap='RdYlGn', fmt=".2f")
plt.title('Correlation Heatmap: Key Predictors vs ASD Class')
plt.savefig('correlation_heatmap.png')
plt.close()

# 3. AGE DISTRIBUTION BY CLASS
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='Class/ASD', multiple="stack", palette='viridis')
plt.title('Age Distribution: ASD vs Non-ASD')
plt.xlabel('Age (Years)')
plt.savefig('age_distribution.png')
plt.close()

# 4. FEATURE IMPORTANCE (From your Model)
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    cols = model_data['columns']

importances = model.feature_importances_
feat_importances = pd.Series(importances, index=cols)
plt.figure(figsize=(10, 6))
feat_importances.nlargest(10).plot(kind='barh', color='#3498db')
plt.title('Top 10 Most Predictive Behavioral Traits (AI Analysis)')
plt.xlabel('Importance Score')
plt.savefig('feature_importance.png')
plt.close()

print("Graphs generated successfully: class_distribution.png, correlation_heatmap.png, age_distribution.png, feature_importance.png")