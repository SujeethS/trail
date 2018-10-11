from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
#from flask import Flask, render_template
import flask
from flask import Flask, request, Response
from PIL import Image
import io
import jsonpickle
from keras import backend as K
import os


# construct the argument parse and parse the arguments
'''
ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--model", required=True,
	#$help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (150, 150))
print("The image shape is ",image.shape)
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
print(image.shape)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model('my_model.h5')
#print(model.layers)


# classify the input image
pred_vals=model.predict(image)[0][0]*100
print(pred_vals)
if((pred_vals)>50.00):
    print ('You have been diagnosed with Pnuemonia on a probabality score of :',str(pred_vals))
    result='You have been diagnosed with Pnuemonia on a probabality score of :',str(pred_vals)
else:
    print('You are Normal with a Probabality score of:',str(100-(pred_vals)))
    result='You have been diagnosed with Pnuemonia on a probabality score of :',str(100-(pred_vals))
'
print('the shape of the image is :' ,image.shape)
# build the label
label = "Pnuemonia" if pnuemonia > normal else "Normal"
proba = pnuemonia if pnuemonia > normal else normal
label = "{}: {:.2f}%".format(label, proba * 100)

print(label)

output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

 #show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
'''

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    image = Image.open(io.BytesIO(r.data))
    image.save('meow.jpeg')
    im = cv2.imread('meow.jpeg', 1)
    imagee = cv2.resize(im, (150, 150))
    print("The image shape is ", imagee.shape)
    imagee = imagee.astype("float") / 255.0
    imagee = img_to_array(imagee)
    imagee = np.expand_dims(imagee, axis=0)
    model = load_model('my_model.h5')
    pred_vals = model.predict(imagee)[0][0] * 100
    print(pred_vals)
    if ((pred_vals) > 50.00):
        print ('You have been diagnosed with Pnuemonia on a probabality score of :', str(pred_vals))
        result = 'You have been diagnosed with Pnuemonia on a probabality score of :', str(pred_vals)
    else:
        print('You are Normal with a Probabality score of:', str(100 - (pred_vals)))
        result = 'You have been diagnosed with Pnuemonia on a probabality score of :', str(100 - (pred_vals))

    #nparr = np.fromstring(r.data, np.uint8)
    # decode image
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'RESULT': '{}'.format(result)
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    K.clear_session()
    os.remove('meow.jpeg')

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)
