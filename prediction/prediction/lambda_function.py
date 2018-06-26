from __future__ import print_function
import json
import mxnet as mx
import numpy as np
import boto3

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
BUCKET_NAME = 'INSERT S3 BUCKET' # replace with your bucket name
KEY = 'temp/img_numpy.json' # replace with your object key
s3 = boto3.resource('s3')
s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/img_numpy.json')
synsets = loadCategories()
model = loadModel("Inception-BN")

def loadModel(modelname, gpu=False):
        sym, arg_params, aux_params = mx.model.load_checkpoint(modelname, 0)
        arg_params['prob_label'] = mx.nd.array([0])
        arg_params['softmax_label'] = mx.nd.array([0])
        if gpu:
            mod = mx.mod.Module(symbol=sym, context=mx.gpu(0))
        else:
            mod = mx.mod.Module(symbol=sym)
        mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
        mod.set_params(arg_params, aux_params)
        return mod
        
        
def loadCategories():
        synsetfile = open('synset.txt', 'r')
        synsets = []
        for l in synsetfile:
                synsets.append(l.rstrip())
        return synsets
    

def predict(img, model, categories, n):
        # compute the predict probabilities
        model.forward(Batch([mx.nd.array(img)]))
        prob = model.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        sortedprobindex = np.argsort(prob)[::-1]
        
        topn = []
        for i in sortedprobindex[0:n]:
                topn.append((prob[i], categories[i]))
        return topn
        

def lambda_handler(event, context):
    s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/img_numpy.json')
    with open('/tmp/img_numpy.json') as json_data:
        d = json.load(json_data)
    predictions = predict(np.array(d['object']),model , synsets, 1)
    prob, label = predictions[0]
    response = json.dumps({"prob": prob.tolist(), "label": [label] })
    return response

