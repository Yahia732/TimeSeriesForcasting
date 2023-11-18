# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from datetime import datetime, timedelta
import warnings

warnings.simplefilter("ignore")
from statsmodels.tsa.seasonal import STL


app = Flask(__name__)


def preprocess_data(data, dataset_id):
    # Use a breakpoint in the code line below to debug your script.
    data = data[['time', 'value']]
    #data.dropna(inplace=True)
    data['time'] = pd.to_datetime(data['time'])

    file_path = "Periods/" + str(dataset_id) + '.txt'
    with open(file_path, 'r') as file:
        total_seconds = file.read().strip()

    try:
        total_seconds = float(total_seconds)
        period = pd.to_timedelta(total_seconds, unit='s')


        stl_result = STL(data['value'], period=max(period / pd.Timedelta(days=1), 2)).fit()
    except ValueError as ve:
        return f"Error converting total_seconds to float: {ve}"
    data['Trend'] = stl_result.trend
    data['Seasonality'] = stl_result.seasonal
    data['Noise'] = stl_result.resid

    file_path = "Lags/" + str(dataset_id) + '.txt'
    with open(file_path, 'r') as file:
        best_lags = int(file.read())
    file_path = "Periods/" + str(dataset_id) + '.txt'
    with open(file_path, 'r') as file:
        seconds_to_add = float(file.read())
    last_timestamp = data['time'].iloc[-1]
    new_timestamp = last_timestamp + timedelta(seconds=seconds_to_add)
    new_row = {'time': new_timestamp, 'value': None, 'Trend': None, 'Seasonality': None, 'Noise': None}
    data.loc[len(data)] = new_row
    print(data)
    date_feature = pd.to_datetime(data["time"]).dt
    data['month'] = date_feature.month
    data['day'] = date_feature.day
    data['hour'] = date_feature.hour
    data['minute'] = date_feature.minute
    data['day_of_week'] = date_feature.dayofweek

    for i in range(1, best_lags + 1):
        data[f'value_lag_{i}'] = data['value'].shift(i)
        data[f'trend_lag_{i}'] = data['Trend'].shift(i)
        data[f'seasonality_lag_{i}'] = data['Seasonality'].shift(i)
        data[f'noise_lag_{i}'] = data['Noise'].shift(i)
    data = data.drop(columns=['value','time','Trend','Seasonality','Noise'])
    # Drop rows with NaN resulting from the shift
    data.dropna(inplace=True)
    return data


def load_model(data, dataset_id):
    with open('Models/' + 'model_' + str(dataset_id) + '.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Make predictions on the validation set
    # Assuming 'data' is the input for the model prediction
    print(data)
    y_val_pred = model.predict(data)
    return y_val_pred[-1]


# Press the green button in the gutter to run the script.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        request_data = request.get_json()

        # Get dataset ID and values from the request
        dataset_id = request_data['dataset_id']
        values = request_data['values']
        data = pd.DataFrame(values)

        # Load the corresponding model based on the dataset ID

        data = preprocess_data(data, dataset_id)
        prediction = load_model(data, dataset_id)

        # Return the prediction in the response body
        response_body = {'prediction': prediction}
        return jsonify(response_body)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
