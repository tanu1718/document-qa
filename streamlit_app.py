import streamlit as st


lab1 = st.Page("Lab1.py",title="Lab 1")
lab2 = st.Page("Lab2.py",title="Lab 2")
lab3 = st.Page("Lab3.py",title="Lab 3")
lab4 = st.Page("Lab4.py",title="Lab 4", default=True)
pg = st.navigation([lab1, lab2, lab3, lab4])
st.set_page_config(page_title="Labs Manager")
pg.run()