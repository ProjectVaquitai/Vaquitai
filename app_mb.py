# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:39:47
:LastEditTime: 2023-02-18 23:44:52
:Description: 
"""
import PIL
import streamlit as st
import data_juicer.platform.src.pages.doc
import data_juicer.platform.src.pages.home
import data_juicer.platform.src.pages.data_process
import data_juicer.platform.src.pages.data_analysis
# import data_juicer.platform.src.pages.model_analysis
import streamlit as st
from streamlit_option_menu import option_menu

# Page Favicon
favicon = PIL.Image.open('./data_juicer/platform/src/assets/favicon.png')
st.set_page_config(page_title='MtBuller', page_icon=favicon, layout='wide', initial_sidebar_state='auto')

def main():
    # Page Title
    # Bootstrap Icons: https://icons.getbootstrap.com/
    apps = [
        {"func": data_juicer.platform.src.pages.home, "title": "首页", "icon": "house"},
        {"func": data_juicer.platform.src.pages.data_process, "title": "数据处理", "icon": "caret-right-square"},
        {"func": data_juicer.platform.src.pages.data_analysis, "title": "数据结果分析", "icon": "bar-chart-line"},
        # {"func": src.pages.model_analysis, "title": "Model Analysis", "icon": "pie-chart"},
        {"func": data_juicer.platform.src.pages.doc, "title": "文档", "icon": "book"},
    ]

    titles = [app["title"] for app in apps]
    titles_lower = [title.lower() for title in titles]
    icons = [app["icon"] for app in apps]

    params = st.experimental_get_query_params()
    default_index = titles_lower.index(params.get("page", [titles[0].lower()])[0].lower())

    with st.sidebar:
        selected = option_menu(
            "MtBuller",
            options=titles,
            icons=icons,
            menu_icon="box",
            default_index=default_index,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "20px"},
                "nav-link": {"font-size": "17px", "text-align": "left", "margin": "2px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#24A608"}
            }
        )

    for app in apps:
        if app["title"] == selected:
            app["func"].write()
            break

if __name__ == '__main__':
    main()