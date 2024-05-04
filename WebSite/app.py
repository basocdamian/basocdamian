from flask import Flask, render_template, request
import pandas as pd
from prophet import Prophet
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    specific_date = request.form['specific_date']
    
    water_quality_model = joblib.load('Water_ML.sav')
    
    prophet_model = Prophet()
    
    df = pd.read_csv('testing2.csv')
    df.rename(columns={'Date': 'ds', 'ph': 'y'}, inplace=True)

    regressor_vars = ['Hardness', 'Solids', 'Chloramines', 'Conductivity', 'Organic_carbon',  'Turbidity']
    for var in regressor_vars:
        prophet_model.add_regressor(var)

    prophet_model.fit(df)

    future = prophet_model.make_future_dataframe(periods=365)
    for var in regressor_vars:
        future[var] = df[var][-len(future):].reset_index(drop=True)

    specific_forecast = future[future['ds'] == specific_date]
    if not specific_forecast.empty:
        predicted_values = specific_forecast[['ds', 'yhat'] + regressor_vars].values[0]
        ph = predicted_values[1]
        Hardness = predicted_values[2]
        Solids = predicted_values[3]
        Chloramines = predicted_values[4]
        Sulfate = 321.865250214064
        Conductivity = predicted_values[5]
        Organic_carbon = predicted_values[6]
        Trihalomethenes = 46.3458637767316
        Turbidity = predicted_values[7]

        input_data = [[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethenes, Turbidity]]

        scaler = StandardScaler()
        scaler.fit(input_data)
        water_data_input = scaler.transform(input_data)

        prediction = water_quality_model.predict(water_data_input)

        if prediction[0] == 0:
            result = "Water is Not safe for consumption"
        else:
            result = "Water is safe for consumption"

        return render_template('index.html', prediction_result=result)
    else:
        return render_template('index.html', prediction_result="No prediction available for the specified date")

if __name__ == '__main__':
    app.run(debug=True)
