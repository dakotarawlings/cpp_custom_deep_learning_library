# -*- coding: utf-8 -*-
"""
Flask application that recieves a post request with a "handrawn image" from a HTML canvas drawing and 
predicts the charachter by implementing a FFNN trained using the MNIST data base and built using the custom cpp library
"""

#import flask tools
from flask import Flask, jsonify, render_template, request
#import the custom FFNN building library
try:
    import FFNN_pymodule
except:
    pass
#Import image and file processing tools
from PIL import Image, ImageOps
import io
import numpy as np
import pickle

#Call flask constructor
app=Flask(__name__)

#define a function that loads the model
def load_model():
    loaded_model=pickle.load(open('model_file.p','rb'))
    return loaded_model

#Function to process the raw image data from the POST request (jpeg image from HTML canvas)
def format_image(image_data):
    #opent the image file
    img=Image.open(io.BytesIO(image_data))
    #convert the image to black and white
    bw_img=img.convert(mode='L')
    #invert the coloring in the image so that the blank space is 0 (this is the format of MNIST and it makes it easy to center the image)
    inv_img = ImageOps.invert(bw_img)
    # use getbbox function to get coorinates that crop out blank space
    bbox = inv_img.getbbox()
    #Create a new croped image that crops out blank space and perfectly frames our digit
    crop=inv_img.crop(bbox)
    #the new croped image is not square so we ge the largest dimension of the cropped image and set that as our new square dimension
    new_size=max(abs(bbox[2]-bbox[0]),abs(bbox[3]-bbox[1]))
    #set the amount that we want to expand the width
    delta_w=new_size-crop.size[0]
    #set the amount that we want to expand the height
    delta_h = new_size - crop.size[1]
    #define our pixel padding for the cropped image to make it square
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    #use the expand function to add zeroes to square the image with our padding dimensions
    new_im = ImageOps.expand(crop, padding)
    #preliminarily resize the image to 28x28 like the mnist dataset
    newsize = (28, 28)
    im28 = new_im.resize(newsize)
    #add a small amount of additional padding similar to what is observed by eye in the MNIST dataset
    im28 = ImageOps.expand(im28, (5,5,5,4))
    #finally resize the image again to 28x28 
    newsize = (28, 28)
    im28 = im28.resize(newsize)
    #convert the image to a numpy array
    array=np.array(im28)
    #flatten the image pixel values to match the MNIST dataset
    array1D=array.reshape(1,-1)
    #convert pixel values to floats and renormalize to 0,1
    array1D=array1D.astype('float32')
    array1D/=255
    #return a 1D array of the pixel values
    return np.array(array1D[0])
    #return list(np.array(array1D[0]))

#Define flask endpoint for the main html page
@app.route('/')
def index():
    return render_template('index.html')


#define an API endpoint that takes in an image file from a post reqest and returns a class prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #monitor the success of the API through a success attribute
    response={'success': False}
    #Check for a post request    
    if request.method=='POST':
        #Check for a media attribute in the json input where we will store our image data
        if request.files.get('media'):
            #retrieve the file sent by the post request
            img_requested=request.files['media'].read()
            #format our image with our formatting function defined above
            formatted_img=format_image(img_requested)
            #load our pickled FFNN model
            model=load_model()
            #pass our image into our model to get a predicted class
            prediction=np.array(model.predict(formatted_img))
            #use argmax to convert our output to a single digit class
            prediction=str(prediction.argmax())
            #add a prediction tag to our response dictionary
            response['predictions']=[]
            #make a label attribute that indicates our models predicted class
            pred={'label':prediction}
            #append our label attribute to our predictions 
            response['predictions'].append(pred)
            #set our success attribute to true
            response['success']=True
            
    #convert our response to a JSON for output        
    return jsonify(response)
   
if __name__=='__main__':
    app.run(debug=False)
    
      
    
    
    
    
    
    
    