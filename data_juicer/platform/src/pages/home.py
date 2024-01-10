# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:39:47
:LastEditTime: 2023-02-18 23:41:34
:Description: Home page shown when the user enters the application
"""

import streamlit as st
import data_juicer.platform.src.utils.st_components as st_components

def write():
    st.title("Welcome to MtBuller!")

    st.write(
        """        
        MtBuler是一个数据探索性分析的App，该应用程序可以执行各种数据分析任务，包括数据分布、数据异常检测、数据可视化等。用户只需要上传数据集，设置数据分析任务，即可获得自动化的分析结果。
    

        ### App使用说明
        - [App使用说明](https://iron-sheet-c6a.notion.site/App-8c6daa98ec984341ad54872cd5153fb2)
        
        """
    )
    # st_components.video_youtube('https://www.youtube.com/embed/B2iAodr0fOo')
