# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:39:47
:LastEditTime: 2023-02-18 23:41:34
:Description: Home page shown when the user enters the application
"""

import streamlit as st
import data_juicer.platform.src.utils.st_components as st_components

def write():
    # st.title("Welcome to MtBuller!")

    # st.write(
    #     """        
    #     MtBuler是一个数据探索性分析的App，该应用程序可以执行各种数据分析任务，包括数据分布、数据异常检测、数据可视化等。用户只需要上传数据集，设置数据分析任务，即可获得自动化的分析结果。
    
    #     """
    # )
    # st_components.video_youtube('https://www.youtube.com/embed/B2iAodr0fOo')
    
        # Read the contents of the Markdown file

    # Vaquita瓦奇塔能带来什么？
    st.title('Vaquita瓦奇塔能带来什么？')
    st.markdown("[【Notion 链接】](https://celestialanthem.notion.site/Vaquita-4cd9a3240c724f9abde709a2358b15bf?pvs=4)")

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
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/d87f1ce3-4272-4ccc-8ac3-f33197b5477a/Untitled.png?id=31ebc759-ce49-4fbd-be85-3f1285ab4906&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709251200000&signature=5jj_HIBn0BMVHtDie24IqjO04xmj5jVxtjVY9XzaZsU&downloadName=Untitled.png', width=800)

    st.markdown('#### 低信息-模糊')
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/f61217db-3931-45a5-87bf-a9fae9b16a99/Untitled.png?id=550a3bfd-94c9-4538-a33c-024f302dfc56&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709222400000&signature=8eyMdlX6c1oaOtK0-KvkEMNFyVsNdx0h1aAEUkYEfiE&downloadName=Untitled.png', width=800)

    st.markdown('#### 重复')
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/2677086f-17c2-4589-910d-c0612d192adc/Untitled.png?id=9d42ef73-1ce5-4857-b635-77a7ce1bf4d6&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709222400000&signature=Pq8oKBQXbSXa7_jrQ2W1fZatJOs9kzmb3EdFG25C90U&downloadName=Untitled.png', width=800)

    st.markdown('#### 特殊比例')
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/808601bb-302e-4ce5-98e7-13bd88773ea1/Untitled.png?id=156ab113-0578-4691-80da-1ca7c26b9144&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709222400000&signature=PQus7ISfzKbEKgUu0-ugatMl422HxxV0NacSQ8Wte4c&downloadName=Untitled.png', width=800)

    st.markdown('#### NSFW (NSFW 通常用来指代那些包含不适宜在工作场合或公共场合观看的内容，例如色情、暴力或其他可能引起不适的内容。)')
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/9cdbeb19-5c9e-4e20-b29c-879560f2031d/Untitled.png?id=c15e4c84-5dcb-464c-afa9-5ecbdd00d266&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709222400000&signature=-tdQeMkUG7LdQUmGCz3XDMqSmHr8gWfzDUH_scE5MXM&downloadName=Untitled.png', width=800)

    st.markdown('### 图文对儿数据')
    st.markdown('#### 文本-图片不匹配')
    st.markdown('<p>文本：BDD-A Dataset | Papers With Code</p>', unsafe_allow_html=True)
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/e69a9fca-12cd-49a0-899e-536aa4a94829/Untitled.png?id=bdcb21cf-5eb1-4572-b91b-76ff1625effd&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709222400000&signature=Szd9Iz5BJTnFUD2aSwYqx6r4cjlIkTcat7nt5G00hMU&downloadName=Untitled.png', width=800)

    st.markdown('#### 文本照抄图像文字')
    st.markdown('<p>文本：Flower Seekers by Blue Derby Foods Ride</p>', unsafe_allow_html=True)
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/0d7d98c7-04f2-444d-ab72-16a8ab4429d8/Untitled.png?id=a3b05258-7956-4b48-bb2a-8e7933ed3fbe&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709222400000&signature=TPq4GiknPzyPmyz5--DP3SXkzJerBn-CAKjfP_NBT1A&downloadName=Untitled.png', width=800)

    ## 搜寻您想要的数据
    st.markdown('## 搜寻您想要的数据')

    st.markdown('### 文本检索')
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/3a366679-3810-43b6-afd8-77a8dd342311/Untitled.png?id=18855209-0ce7-4243-879b-e083a2eb78c3&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709272800000&signature=GOE9a_EPbAFgsn3bfcjq4CBQHE22SjlEAa5ER62xzpw&downloadName=Untitled.png', width=800)

    st.markdown('### 图像检索')
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/76fedb68-dc68-46aa-bc80-29c1702d8c9f/Untitled.png?id=097dc595-ad73-4cc7-85a3-3847692c865e&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709272800000&signature=-1Zw6HagIWPw7IMHtPO7bQK0UcODna7IayZJ-DhnLFQ&downloadName=Untitled.png', width=800)

    ## 了解您的数据集
    st.markdown('## 了解您的数据集')

    st.markdown('### 交互式数据分布')
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/903be8d5-c793-4742-a523-685f040bb276/Untitled.png?id=4d74845e-31e0-4b25-b520-b10d7ed313bf&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709272800000&signature=J4Fy52k_7DKPyk3pmBSFlWIEMcBZVj5TL3VG1oWAyC4&downloadName=Untitled.png', width=800)

    st.markdown('### 数据集对比')
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/2bafbcff-536f-4fdb-83cc-845953d3e4e8/Untitled.png?id=7040b0d7-4d56-4f40-bba0-b08109931dc2&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709272800000&signature=DimmVRB8HuP7M3xvId6Ne3ed4CYwrhpOjCsSfAAJgwQ&downloadName=Untitled.png', width=800)

    st.markdown('### 关联性展示')
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/f8c2aaa8-3208-4bc8-9dbc-6993042245bc/Untitled.png?id=e2b53930-72f5-4a51-b9f4-d6cecf7e6d6f&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709272800000&signature=0NSotb0U9BfOJFitqXn_ja6diu_ZirRqI9rDcJMARRE&downloadName=Untitled.png', width=800)

    st.markdown('### 桑基图')
    st.image('https://file.notion.so/f/f/f5292c1b-d6eb-49ee-9a74-ad98cd50f569/f7f67615-6c64-45b9-a2a0-af30e813b47f/Untitled.png?id=ff95575f-ec90-4749-a90c-875e3e5ef08f&table=block&spaceId=f5292c1b-d6eb-49ee-9a74-ad98cd50f569&expirationTimestamp=1709272800000&signature=5DwHi47VaWkjMZKMWKqKgNoTuJ7xaVAUtEQJlZxwam8&downloadName=Untitled.png', width=800)
