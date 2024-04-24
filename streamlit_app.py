import streamlit as st
import matplotlib.pyplot as plt
from main import main  # Make sure to replace 'your_main_script' with the actual name of your Python file

st.title("Hello Streamlit")

st.header("Portfolio Analysis")

st.subheader("Crypos used: Badger ")



def display_data():
    main() 

    #st.title('Crypto DataFrame')
    #st.write(crypto_df)


if __name__ == '__main__':
    display_data()
