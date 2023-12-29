from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load datasets
df1 = pd.read_csv("test_crop.csv")
df2 = pd.read_csv("crop .csv")

# Data preprocessing
df1.label = df1.label.map({"wheat": 0, "Sugarcane": 1})
df2.label = df2.label.map({"rice": 2, "wheat": 0, "Sugarcane": 1, "Mung Bean": 3, "Tea": 4, "millet": 5, "maize": 6,
                           "Lentil": 7, "Jute": 8, "Coffee": 9, "Cotton": 10, "Ground Nut": 11, "Peas": 12, "Rubber": 13,
                           "Tobacco": 14, "Kidney Beans": 15, "Moth Beans": 16, "Coconut": 17, "Black gram": 18,
                           "Adzuki Beans": 19, "Pigeon Peas": 20, "Chickpea": 21, "banana": 22, "grapes": 23, "apple": 24,
                           "mango": 25, "muskmelon": 26, "orange": 27, "papaya": 28})

# Logistic Regression
lr = LogisticRegression(random_state=0)
lr.fit(df2.drop('label', axis=1), df2['label'])

# Crop mapping
crop_mapping = {
    0: "wheat",
    1: "Sugarcane",
    2: "rice",
    3: "Mung Bean",
    4: "Tea",
    5: "millet",
    6: "maize",
    7: "Lentil",
    8: "Jute",
    9: "Coffee",
    10: "Cotton",
    11: "Ground Nut",
    12: "Peas",
    13: "Rubber",
    14: "Tobacco",
    15: "Kidney Beans",
    16: "Moth Beans",
    17: "Coconut",
    18: "Black gram",
    19: "Adzuki Beans",
    20: "Pigeon Peas",
    21: "Chickpea",
    22: "banana",
    23: "grapes",
    24: "apple",
    25: "mango",
    26: "muskmelon",
    27: "orange",
    28: "papaya",
}

accuracy_value = 0.0  # Initialize the accuracy value

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        ph = float(request.form['ph'])

        # Model predictions
        prediction = lr.predict([[temperature, humidity, rainfall, ph]])

        # Map the predicted crop number to its name
        predicted_crop = crop_mapping.get(prediction[0], "Unknown")

        # Model evaluation
        global accuracy_value
        accuracy_value = accuracy_score([df1.label.iloc[0]], prediction)  # Fix the order of arguments

        # Render the prediction on a new page and pass the accuracy value
        return render_template('result.html', predicted_crop=predicted_crop, accuracy=accuracy_value * 100)

if __name__ == '__main__':
    app.run(debug=True)
