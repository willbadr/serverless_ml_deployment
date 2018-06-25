from __future__ import print_function

import json
import ast
import cv2
import boto3
import numpy as np
import urllib
print('Loading function')

import time
start_time = time.time()

client = boto3.client('lambda')
s3 = boto3.resource('s3')
bucket="sagemaker-serverless"

def upload_to_s3(channel, file, file_name):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file_name
    s3.Bucket(bucket).put_object(Key=key, Body=data)
    
def get_image(fname):
    # download and show the image
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
         return None

    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img
    
def lambda_handler(event, context):
    file_path = "/tmp/image.jpeg"
    url = event['queryStringParameters']['url']
    urllib.urlretrieve(url,file_path)
    img = get_image(file_path)
    print("--- %s time one ---" % (time.time() - start_time))
    payload = { "object": img.tolist() }
    with open('/tmp/img_numpy.json', 'w') as outfile:
        json.dump(payload, outfile)
    upload_to_s3('temp','/tmp/img_numpy.json','img_numpy.json')
    print("--- %s time two ---" % (time.time() - start_time))
    result = client.invoke(
    FunctionName='arn:aws:lambda:us-west-2:625616379791:function:cloud9-prediction-prediction-1E13U9IV591XN',
    InvocationType='RequestResponse')
    print("--- %s seconds ---" % (time.time() - start_time))
    results_dict = ast.literal_eval(json.loads(result['Payload'].read()))
    #print(results_dict)
    prob = "{0:.2f}".format(results_dict.get('prob') * 100)
    msg = "There is %" + str(prob) + " chance that the picture is " + results_dict.get('label')[0]
    html_body = '<html><body><h2>' + msg + ' </h2><img src="' + url +  '" alt="flower" width="500" height="377"></body></html>'
    response = {
        'statusCode': 200,
        'body': html_body
    }
    return response
    