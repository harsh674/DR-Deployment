import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
#from keras.models import load_model


import efficientnet.tfkeras as efn
model5b = tf.keras.Sequential([
        efn.EfficientNetB3(
            input_shape=(224,224, 3),
            weights='imagenet',
            include_top=False
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

model5b.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['acc'])
model5b.load_weights('dref.hdf5')
import streamlit as st
st.write("""
         #Diabetic Retinopathy Detection
         """
         )
st.write("This is a ML Model for Classification of Diabetic Retinopathy")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import numpy as np
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = np.expand_dims(img_resize, axis=0)
    
        prediction = model5b.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model5)
    
    if np.argmax(prediction) == 0:
        st.write("Mild!")
    elif np.argmax(prediction) == 1:
        st.write("Moderate!")
    elif np.argmax(prediction) == 2:
        st.write("No DR!")
    elif np.argmax(prediction) == 3:
        st.write("Proliferate!")
    else:
        st.write("Severe!")
    
    st.text("Probability (0: Mild, 1: Moderate, 2: No DR,3: Proliferate,4: Severe")
    st.write(prediction)
