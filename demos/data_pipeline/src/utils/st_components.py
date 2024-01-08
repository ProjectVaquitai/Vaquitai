# -*- coding:utf-8 -*-
"""
:Date: 2023-02-19 14:55:21
:LastEditTime: 2023-02-19 14:58:53
:Description: 
"""
# -*- coding:utf-8 -*-
"""
:Date: 2023-02-19 14:55:21
:LastEditTime: 2023-02-19 14:55:22
:Description: 
"""
import streamlit as st


def video_youtube(src: str, width=560, height=315):
    """An extension of the video widget

    Arguments:
        src {str} -- A youtube url like https://www.youtube.com/embed/B2iAodr0fOo

    Keyword Arguments:
        width {str} -- The width of the video (default: {"100%"})
        height {int} -- The height of the video (default: {315})
    """
    st.write(
        f'<iframe width="{width}" height="{height}" src="{src}" frameborder="0" '
        'allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" '
        "allowfullscreen></iframe>",
        unsafe_allow_html=True,
    )
