from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('titanic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])

        input_data = pd.DataFrame(
            [[pclass, sex, age, sibsp, parch, fare, embarked]],
            columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        )

        prediction = model.predict(input_data)[0]
        
        if prediction == 1:
            result = "Omon qoladi (Survived) "
        else:
            result = "Halok bo'ladi (Not Survived) "

        return render_template('index.html', prediction_text=f"Natija: Yo'lovchi {result}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Xatolik yuz berdi: {e}")

if __name__ == "__main__":
    app.run(debug=True)
