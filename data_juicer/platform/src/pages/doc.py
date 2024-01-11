# -*- coding:utf-8 -*-
"""
:Date: 2023-02-19 15:05:02
:LastEditTime: 2023-02-19 15:05:04
:Description: 
"""

import streamlit as st
import data_juicer.platform.src.utils.st_components as st_components

def write():
    st.title("操作指南")  
    # html = "https://www.tensorflow.org/?hl=zh-cn"
    # st.components.v1.html(html, width=None, height=None, scrolling=False)
    # Read the contents of the Markdown file
    with open('./data_juicer/platform/src/docs/MANNUL_ZH.MD', 'r') as file:
        markdown_text = file.read()

    # Display the Markdown content in the Streamlit app
    st.markdown(markdown_text)