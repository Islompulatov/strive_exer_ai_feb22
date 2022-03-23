from sre_constants import SUCCESS
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
df =pd.read_csv('./data_action.csv')

st.title('The Best Action Movie')
st.header('Choose your best movie')
st.subheader('Good Luck')
st.write('Have a fun!')
img = Image.open("action.webp")
st.image(img, width = 700, caption="Action Image")
status = st.sidebar.radio('How is your mood today?',('Good','Bad'))
if status == 'Good':
    st.success('You can watch movie today')
else:
    st.warning('You can movie another day, go to sleep')    
select_classifier = st.sidebar.selectbox('select your favourite actor/actress',('Tom Hardy','Ben Affleck','Charlize Theron','Natalie Portman'))

    
st.write(select_classifier)
data_set=st.sidebar.selectbox('select the director',('James Cameron', 'David Fincher', 'George Miller','Pierre Morel'))
st.write(data_set)
level = st.sidebar.slider('Choose your amount of movie', 1,100)



vid_file = open("file.mp4","rb").read()
st.video(vid_file)