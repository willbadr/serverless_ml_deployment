from __future__ import print_function
import json
import mxnet as mx
import numpy as np
import boto3

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


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
    
synsets = loadCategories()
model = loadModel("Inception-BN")
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

'''
sym, arg_params, aux_params = mx.model.load_checkpoint('102flowers', 5)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
labels = [ 'alpine sea holly','anthurium','artichoke','azalea','ball moss','balloon flower','barbeton daisy','bearded iris','bee balm','bird of paradise','bishop of llandaff','black-eyed susan','blackberry lily','blanket flower','bolero deep blue','bougainvillea','bromelia','buttercup','californian poppy','camellia','canna lily','canterbury bells','cape flower','carnation','cautleya spicata','clematis','colt\'s foot','columbine','common dandelion','corn poppy','cyclamen ','daffodil','desert-rose','english marigold','fire lily','foxglove','frangipani','fritillary','garden phlox','gaura','gazania','geranium','giant white arum lily','globe thistle','globe-flower','grape hyacinth','great masterwort','hard-leaved pocket orchi','hibiscus','hippeastrum ','japanese anemone','king protea','lenten rose','lotus','love in the mist','magnolia','mallow','marigold','mexican aster','mexican petunia','monkshood','moon orchid','morning glory','orange dahlia','osteospermum','oxeye daisy','passion flower','pelargonium','peruvian lily','petunia','pincushion flower','pink primrose','pink-yellow dahlia?','poinsettia','primula','prince of wales feathers','purple coneflower','red ginger','rose','ruby-lipped cattleya','siam tulip','silverbush','snapdragon','spear thistle','spring crocus','stemless gentian','sunflower','sweet pea','sweet william','sword lily','thorn apple','tiger lily','toad lily','tree mallow','tree poppy','trumpet creeper','wallflower','water lily','watercress','wild pansy','windflower','yellow iris']
BUCKET_NAME = 'sagemaker-walebadr' # replace with your bucket name
KEY = 'temp/img_numpy.json' # replace with your object key

s3 = boto3.resource('s3')

s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/img_numpy.json')
def predict(img):
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    i = a[0]
    probability = prob[i]
    label = labels[i]
    print('probability=%f, class=%s' %(prob[i], labels[i]))
    return probability, label
'''
BUCKET_NAME = 'sagemaker-serverless' # replace with your bucket name
KEY = 'temp/img_numpy.json' # replace with your object key
s3 = boto3.resource('s3')
s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/img_numpy.json')
def lambda_handler(event, context):
    s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/img_numpy.json')
    with open('/tmp/img_numpy.json') as json_data:
        d = json.load(json_data)
    predictions = predict(np.array(d['object']),model , synsets, 1)
    prob, label = predictions[0]
    response = json.dumps({"prob": prob.tolist(), "label": [label] })
    return response
