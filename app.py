import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from skimage.transform import resize
import streamlit as st
import pickle
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding',False)
st.title('Image Classifier')
st.text('Upload image')
model=pickle.load(open('model.pkl','rb'))
upload_file=st.file_uploader('Choose an image',type='jpg')
if upload_file is not None:
  img=Image.open(upload_file)
  st.image(img,caption='Image uploaded')

if st.button('Predict'):
  labels=['sunflower','rugby ball','ice cream cone']
  st.write('Result')
  flat_data=[]
  img=np.array(img)
  img_resized=resize(img,(150,150,3))
  flat_data.append(img_resized.flatten())
  flat_data=np.array(flat_data)
  y_out=model.predict(flat_data)
  y_out=labels[y_out[0]]
  st.title(f'Predicted output: {y_out}')
  q=model.predict_proba(flat_data)
  for index,item in enumerate(labels):
    st.write(f'{item} : {q[0][index]*100}%')

