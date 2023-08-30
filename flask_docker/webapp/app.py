import pickle
# import numpy as np
import json
import pandas as pd
# import sys
# from sklearn.neighbors import KNeighborsClassifier


# Импортируем Flask для создания API
from flask import Flask, request

# load model
filename = 'clf_model.sav'
best_clf = pickle.load(open(filename, 'rb'))

# codes of departments
with open('department_codes.json', 'r') as f:
    department_codes = json.load(f)

# Инициализируем приложение Flask
app = Flask(__name__)


# Создайте конечную точку API
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    body_json = request.json
    df = pd.json_normalize(body_json['predict_delivery'])

    # Preparation data
    df[["OrderDate", "DeliveryDate"]] = df[["OrderDate", "DeliveryDate"]].apply(pd.to_datetime)
    df['DaysDelivery'] = (df['DeliveryDate'] - df['OrderDate']).dt.days
    df['Weekday'] = df['DeliveryDate'].dt.weekday
    df['Department'] = df['Department'].apply(lambda x: department_codes[x])

    X = df[['DeliveryAttempt', 'DaysDelivery', 'Department', 'OrderAmount', 'Weekday']]
    predict_proba_result = best_clf.predict_proba(X)
    predict_proba_result_list = predict_proba_result.tolist()
    result_list = list()
    for i in range(len(predict_proba_result_list)):
        result_list.append({'OrderNumber': body_json['predict_delivery'][i]['OrderNumber'],
                            'delivery_prediction': round(predict_proba_result_list[i][1], 2)})

    return {
        'statusCode': 200,
        'body': json.dumps(
            {'response': result_list
            },
            default=vars,
        ),
    }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
