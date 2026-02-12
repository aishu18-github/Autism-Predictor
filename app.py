from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

app = Flask(__name__)

# Load model and columns
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    model_columns = model_data['columns']

# Load dataset for context
df = pd.read_csv('data/cleaned_autism_data.csv')

def generate_medical_console(patient_age, prediction_class, input_values):
    sns.set_style("whitegrid")
    # 1x4 Grid for a Professional Diagnostic Layout
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 5))
    
    # --- CHART 1: Age Density ---
    subset = df[df['Class/ASD'] == prediction_class]
    sns.kdeplot(data=subset, x='age', fill=True, color='#4a90e2', ax=ax1)
    ax1.axvline(patient_age, color='#e74c3c', linestyle='--', linewidth=2, label='Patient')
    ax1.set_title("Age Distribution Context", fontsize=12, fontweight='bold')
    ax1.legend()

    # --- CHART 2: Response Heatmap (Personalized) ---
    # Mapping A1-A10 scores into a 2x5 grid
    trait_scores = np.array(input_values[:10]).reshape(2, 5)
    sns.heatmap(trait_scores, annot=True, cmap="YlGnBu", cbar=False, 
                xticklabels=['Q1-5', '', '', '', ''], yticklabels=['R1', 'R2'], ax=ax2)
    ax2.set_title("Behavioral Intensity Map", fontsize=12, fontweight='bold')

    # --- CHART 3: Trait Clusters ---
    # Grouping A1-A10 into clinical categories
    categories = ['Social', 'Communication', 'Attention', 'Detail', 'Focus']
    values = [input_values[0]+input_values[8], input_values[1]+input_values[5], 
              input_values[2]+input_values[3], input_values[6]+input_values[9], 
              input_values[4]+input_values[7]]
    ax3.bar(categories, values, color=sns.color_palette("viridis", 5))
    ax3.set_title("Clinical Trait Clusters", fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 2)

    # --- CHART 4: Feature Weights ---
    importances = pd.Series(model.feature_importances_, index=model_columns)
    importances.nlargest(5).sort_values().plot(kind='barh', color='#9b59b6', ax=ax4)
    ax4.set_title("AI Weighting Indicators", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

@app.route('/')
def home():
    return render_template('index.html', columns=model_columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [float(request.form.get(col, 0)) for col in model_columns]
        patient_age = float(request.form.get('age', 25))
        
        prediction = int(model.predict([np.array(inputs)])[0])
        probs = model.predict_proba([np.array(inputs)])[0]
        confidence = round(np.max(probs) * 100, 2)
        
        dashboard = generate_medical_console(patient_age, prediction, inputs)
        
        res_text, res_color = ("ASD Traits Detected", "#e74c3c") if prediction == 1 else ("No Traits Detected", "#27ae60")
        res_desc = f"The analysis identifies a {confidence}% probability for this classification."
            
        return render_template('result.html', text=res_text, desc=res_desc, color=res_color, chart=dashboard, confidence=confidence)
    except Exception as e:
        return f"System Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)