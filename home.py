import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from streamlit.logger import get_logger
LOGGER = get_logger(__name__)

@st.cache_data
def load_data():
    LOGGER.info('Loading data...')
    Fer2013_data = pd.read_csv('dataset/fer2013.csv')
    train_data = Fer2013_data[Fer2013_data['Usage'] == 'Training']
    val_data = Fer2013_data[Fer2013_data['Usage'] == 'PrivateTest']
    test_data = Fer2013_data[Fer2013_data['Usage'] == 'PublicTest']
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    LOGGER.info('Data loaded.')
    return train_data, val_data, test_data
    
def home_page():
    with open('README.md', 'r') as file:
        readme = file.read()
    st.markdown(readme, unsafe_allow_html=True)
    st.sidebar.success("Select a model above.")

if __name__ == '__main__':
    home_page()
