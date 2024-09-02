import streamlit as st # type: ignore

create_page = st.Page("main.py", title="one image", icon=":material/add_circle:")
delete_page = st.Page("test.py", title="Two image differance", icon=":material/add_circle:")

pg = st.navigation([create_page, delete_page])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()