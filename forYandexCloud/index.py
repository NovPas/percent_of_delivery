import pandas as pd
import pickle
import json

def handler(event, context):
    
    body_str = event['body']
    body_json = json.loads(body_str)
    
    # load model
    filename = 'clf_model.sav'
    best_clf = pickle.load(open(filename, 'rb'))
    best_clf

    # codes of departments
    with open('department_codes.json', 'r') as f:
        department_codes = json.load(f)

    df = pd.json_normalize(body_json['predict_delivery'])
    
    # Preparation data
    df[["OrderDate", "DeliveryDate"]] = df[["OrderDate", "DeliveryDate"]].apply(pd.to_datetime)
    df['DaysDelivery'] = (df['DeliveryDate'] - df['OrderDate']).dt.days
    df['Weekday'] = df['DeliveryDate'].dt.weekday
    df['Department'] = df['Department'].apply(lambda x: department_codes[x])
    
    X = df[['DeliveryAttempt', 'DaysDelivery', 'Department', 'OrderAmount', 'Weekday']]
    predict_proba_result = best_clf.predict_proba(X)
    predict_proba_result_list = result.tolist() 
    result_list = list()
    for i in range(len(predict_values)):
        result_list.append({'OrderNumber': body_json['predict_delivery'][i]['OrderNumber'],
                            'delivery_prediction': round(predict_values[i][1],2)})
    
    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                'context': context,
                'response': result_list
            }, 
            default=vars,
        ),
    }