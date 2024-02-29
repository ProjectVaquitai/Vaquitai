# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:39:47
:LastEditTime: 2023-02-18 23:41:34
:Description: Home page shown when the user enters the application
"""

import streamlit as st
import data_juicer.platform.src.utils.st_components as st_components
from data_juicer.platform.src.utils.st_components import get_remote_ip
from loguru import logger

def write():
    # st.title("Welcome to MtBuller!")
    logger.info(f"enter home page, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")

    # st.write(
    #     """        
    #     MtBuler是一个数据探索性分析的App，该应用程序可以执行各种数据分析任务，包括数据分布、数据异常检测、数据可视化等。用户只需要上传数据集，设置数据分析任务，即可获得自动化的分析结果。
    
    #     """
    # )
    # st_components.video_youtube('https://www.youtube.com/embed/B2iAodr0fOo')
    
        # Read the contents of the Markdown file

    # Vaquita瓦奇塔能带来什么？
    st.title('Vaquita瓦奇塔能带来什么？')
    # st.image('caeefa0c-7dcb-4b57-b6a6-ba7b4ed2aebb/Untitled.png?id=ff8d46ef-fca0-4b02-b90d-e1c44994ca55&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709294400000&signature=eNWDjXJqL3rVaLC1IeL2nvsx1dWyr1WzcVlsHwTihiU&downloadName=Untitled.png', width=400)

    
    st.markdown("[【Notion 链接】](https://celestialanthem.notion.site/Vaquita-4cd9a3240c724f9abde709a2358b15bf?pvs=4)")
    st.markdown("""
        
        ### 简介
        Vaquita瓦奇塔，开源的 AI 数据处理平台，帮助用户快速理解数据集，发现其中的异常或“坏蛋”，并以最高效的方式找到用户所需的数据
    """)
    # 使命和愿景
    st.markdown("""
        
        ### 使命
        让 AI 数据处理变得更简单、更智能
        
        ### 愿景
        我们致力于解决 AI 数据处理中的挑战，让更多人能够轻松训练出更优秀的 AI 模型，从而释放 AI 的潜力，改变世界
    """)

    # 目标用户
    st.subheader('目标用户')
    st.write("包括但不限于数据科学家、数据分析师、数据工程师和 AI 工程师")

    # 优势
    st.subheader('优势')
    st.write("""
        1. **无需编程**：我们的平台提供直观易用的界面，无需编程知识即可进行数据处理和分析
        2. **自动化数据分析**：借助我们的智能算法，用户可以自动分析大规模数据集，节省时间和精力
        3. **交互式数据可视化**：我们提供丰富的可视化工具，帮助用户更直观地理解数据
        4. **数据异常检测**：我们的平台能够快速识别数据中的异常或不一致性，帮助用户及时发现并解决问题
    """)

    # 发展规划
    st.subheader('发展规划')
    st.write("""
        1. **功能拓展**：我们将持续增加新功能和支持更多数据类型，以满足不断变化的用户需求
        2. **开发者生态**：我们鼓励开发者参与，支持他们开发自定义功能并共享给其他用户，构建一个互助共赢的生态系统
        3. **智能化技术**：我们将不断优化算法和技术，实现更智能的数据输入和模块调度，提升用户体验
        4. **自然语言交互**：我们计划引入自然语言处理技术，让用户可以通过自然语言与平台进行交互，实现更便捷的操作和查询
    """)
    # 目录
    # st.markdown('''
    # - [揪出您数据集中的坏蛋](#揪出您数据集中的坏蛋)
    #     - [图像数据](#图像数据)
    #         - [低信息](#低信息)
    #         - [低信息-模糊](#低信息-模糊)
    #         - [重复](#重复)
    #         - [特殊比例](#特殊比例)
    #         - [NSFW](#NSFW)
    #     - [图文对儿数据](#图文对儿数据)
    #         - [文本-图片不匹配](#文本-图片不匹配)
    #         - [文本照抄图像文字](#文本照抄图像文字)
    # - [搜寻您想要的数据](#搜寻您想要的数据)
    #     - [文本检索](#文本检索)
    #     - [图像检索](#图像检索)
    # - [了解您的数据集](#了解您的数据集)
    #     - [交互式数据分布](#交互式数据分布)
    #     - [数据集对比](#数据集对比)
    #     - [关联性展示](#关联性展示)
    #     - [桑基图](#桑基图)
    # ''', unsafe_allow_html=True)

    ## 揪出您数据集中的"坏蛋"！
    st.markdown('## 揪出您数据集中的"坏蛋"！')

    st.markdown('### 图像数据')
    st.markdown('#### 低信息')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/404.png', width=800)

    st.markdown('#### 低信息-模糊')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/low_info.png', width=800)

    st.markdown('#### 重复')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/duplicated.png', width=800)
    st.markdown('#### 特殊比例')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/ratio.png', width=800)

    st.markdown('#### NSFW (NSFW 通常用来指代那些包含不适宜在工作场合或公共场合观看的内容，例如色情、暴力或其他可能引起不适的内容。)')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/nsfw.png', width=800)

    st.markdown('### 图文对数据')
    st.markdown('#### 文本-图片不匹配')
    st.markdown('<p>文本：BDD-A Dataset | Papers With Code</p>', unsafe_allow_html=True)
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/mismatch.png', width=800)

    st.markdown('#### 文本未描述图片仅照抄文字')
    st.markdown('<p>文本：Flower Seekers by Blue Derby Foods Ride</p>', unsafe_allow_html=True)
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/flower_seekers.png', width=800)

    ## 搜寻您想要的数据
    st.markdown('## 搜寻您想要的数据')

    st.markdown('### 文本检索')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/text_retrieval.png', width=800)

    st.markdown('### 图像检索')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/image_retrieval.png', width=800)

    ## 了解您的数据集
    st.markdown('## 了解您的数据集')

    st.markdown('### 交互式数据分布')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/data_distribution.png', width=800)

    st.markdown('### 数据集对比')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/compare_datasets.png', width=800)

    st.markdown('### 关联性展示')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/associations.png', width=800)

    st.markdown('### 桑基图')
    st.image('https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/sankey.png', width=800)
