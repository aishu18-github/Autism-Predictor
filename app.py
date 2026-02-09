from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    model_columns = data['columns']

@app.route('/')
def home():
    return render_template('index.html', columns=model_columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(request.form.get(col, 0)) for col in model_columns]
        prediction = model.predict([np.array(input_features)])
        
        if prediction[0] == 1:
            res_text, res_color = "Traits Detected", "#e74c3c"
            res_desc = "The analysis suggests traits associated with ASD. Professional consultation is recommended."
        else:
            res_text, res_color = "No Traits Detected", "#27ae60"
            res_desc = "The analysis did not find patterns typically associated with ASD in this screening."
            
        return render_template('result.html', text=res_text, desc=res_desc, color=res_color)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)