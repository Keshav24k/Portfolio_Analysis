import streamlit as st
import matplotlib.pyplot as plt
from main import main  # Make sure to replace 'your_main_script' with the actual name of your Python file

st.title("Hi Team!")

st.header("Portfolio Analysis")

def display_data():
    main() 

if __name__ == '__main__':
    display_data()
