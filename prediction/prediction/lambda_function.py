from __future__ import print_function
import json
import mxnet as mx
import numpy as np
import boto3

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mx.model.load_checkpoint('102flowers', 5)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
labels = [ 'alpine sea holly','anthurium','artichoke','azalea','ball moss','balloon flower','barbeton daisy','bearded iris','bee balm','bird of paradise','bishop of llandaff','black-eyed susan','blackberry lily','blanket flower','bolero deep blue','bougainvillea','bromelia','buttercup','californian poppy','camellia','canna lily','canterbury bells','cape flower','carnation','cautleya spicata','clematis','colt\'s foot','columbine','common dandelion','corn poppy','cyclamen ','daffodil','desert-rose','english marigold','fire lily','foxglove','frangipani','fritillary','garden phlox','gaura','gazania','geranium','giant white arum lily','globe thistle','globe-flower','grape hyacinth','great masterwort','hard-leaved pocket orchi','hibiscus','hippeastrum ','japanese anemone','king protea','lenten rose','lotus','love in the mist','magnolia','mallow','marigold','mexican aster','mexican petunia','monkshood','moon orchid','morning glory','orange dahlia','osteospermum','oxeye daisy','passion flower','pelargonium','peruvian lily','petunia','pincushion flower','pink primrose','pink-yellow dahlia?','poinsettia','primula','prince of wales feathers','purple coneflower','red ginger','rose','ruby-lipped cattleya','siam tulip','silverbush','snapdragon','spear thistle','spring crocus','stemless gentian','sunflower','sweet pea','sweet william','sword lily','thorn apple','tiger lily','toad lily','tree mallow','tree poppy','trumpet creeper','wallflower','water lily','watercress','wild pansy','windflower','yellow iris']
BUCKET_NAME = 'sagemaker-demo-sydsummit' # replace with your bucket name
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

def lambda_handler(event, context):
    s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/img_numpy.json')
    with open('/tmp/img_numpy.json') as json_data:
        d = json.load(json_data)
    prob, label = predict(np.array(d['object']))
    print(label)
    response = json.dumps({ "prob": prob.tolist(), "label": [label] })
    return response
