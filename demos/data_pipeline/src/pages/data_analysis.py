# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:58:23
:LastEditTime: 2023-02-19 14:01:40
:Description: 
"""
import numpy as np
import pandas as pd
import faiss
import PIL
from PIL import Image
import altair as alt
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import torch.nn.functional as F
import streamlit as st
from data_juicer.format.load import load_formatter
from data_juicer.utils.model_utils import get_model, prepare_model


@st.cache_resource
def load_model():
    model_key = prepare_model(model_type='hf_blip', model_key='Salesforce/blip-itm-base-coco')
    model, processor = get_model(model_key)
    return model, processor


def display_dataset(dataframe, cond, show_num, desp, type, all=True):
    examples = dataframe.loc[cond]
    if all or len(examples) > 0:
        st.subheader(
            f'{desp}: :red[{len(examples)}] of '
            f'{len(dataframe.index)} {type} '
            f'(:red[{len(examples)/len(dataframe.index) * 100:.2f}%])')

        st.dataframe(examples[:show_num], use_container_width=True)


def display_image_grid(urls, cols=3, width=300):
    num_rows = int(np.ceil(len(urls) / cols))
    image_grid = np.zeros((num_rows * width, cols * width, 3), dtype=np.uint8)

    for i, url in enumerate(urls):
        image = Image.open(url)
        image = image.resize((width, width))
        image = np.array(image)
        row = i // cols
        col = i % cols
        image_grid[row * width:(row + 1) * width, col * width:(col + 1) * width, :] = image

    st.image(image_grid, channels="RGB")


@st.cache_data
def convert_to_jsonl(df):
    return df.to_json(orient='records', lines=True, force_ascii=False).encode('utf_8_sig')


def plot_image_clusters(dataset):
    __dj__image_embedding_2d = np.array(dataset['__dj__image_embedding_2d'])
    df = pd.DataFrame(__dj__image_embedding_2d, columns=['x', 'y'])
    df['image'] = dataset['image']
    df['description'] = dataset['__dj__image_caption']

    marker_chart = alt.Chart(df).mark_circle().encode(
        x='x',
        y='y',
        href='image:N',
        tooltip=['image', 'description']
    ).properties(
        width=800,
        height=600,
    ).configure_legend(
        disable=True
    )
    return marker_chart


def write():
    theme_plotly = None
    tab_data_cleaning, tab_data_mining, tab_data_insights = st.tabs(['数据清洗', '数据挖掘', '数据洞察'])
    formatter = load_formatter('/home/beacon/data-juicer/demos/data_pipeline/outputs/demo-process/demo-processed.jsonl')
    processed_dataset = formatter.load_dataset(4)

    with tab_data_cleaning:
        filter_condition = {'__dj__blurriness_label': True, '__dj__brightness_label':'bright', '__dj__duplicated':True}
        filter_nums = {}
        for key, value in filter_condition.items():
            filter_nums[key] = sum(np.array(processed_dataset[key]) == value)

        def draw_sankey_diagram(source_data, target_data, value_data, labels):
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=labels
                ),
                link=dict(
                    source=source_data,
                    target=target_data,
                    value=value_data
                )
            )])
            fig.update_layout(title_text="数据清洗比例统计", title_font=dict(size=25), font_size=16)
            st.plotly_chart(fig)

        cnt = 1
        source_data = [0]
        target_data = [cnt]
        value_data = [1 - sum(filter_nums.values())/len(processed_dataset['image'])]
        labels = ['Origin', 'Retained_'+str(value_data[0]*100)+'%']
        for key, value in filter_nums.items():
            if value == 0:
                continue
            cnt += 1
            source_data.append(0)
            target_data.append(cnt)
            value_data.append(value/len(processed_dataset[key]))
            labels.append('Discarded_' + key+str(value_data[-1]*100)+'%')
        
        draw_sankey_diagram(source_data, target_data, value_data, labels)
        ds = pd.DataFrame(processed_dataset)
        all_conds = np.ones(len(ds.index), dtype=bool)
        display_dataset(ds, all_conds, 10, 'Retained sampels', 'images')
        st.download_button('Download Retained data as JSONL',
                           data=convert_to_jsonl(ds.loc[all_conds]),
                           file_name='retained.jsonl')
        display_dataset(ds, np.invert(all_conds), 10, 'Discarded sampels', 'images')
        st.download_button('Download Discarded data as JSONL',
                           data=convert_to_jsonl(ds.loc[np.invert(all_conds)]),
                           file_name='discarded.jsonl')

    with tab_data_mining:
        image_embeddings = np.array(processed_dataset['__dj__image_embedding']).astype('float32')
        faiss_index = faiss.IndexFlatL2(image_embeddings.shape[1])
        faiss_index.add(image_embeddings)
        model, processor = load_model()
        input_text = st.text_input("以文搜图: 输入相应的文本信息", 'a picture of dog')
        inputs = processor(text=input_text, return_tensors="pt")
        text_output = model.text_encoder(inputs.input_ids, attention_mask=inputs.attention_mask, return_dict=True) 
        text_feature = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1).detach().cpu().numpy() 

        D, I = faiss_index.search(text_feature.astype('float32'), 10)
        retrieval_image_list = [processed_dataset['image'][i] for i in I[0]]
        display_image_grid(retrieval_image_list, 5, 300)

    with tab_data_insights:
        st.markdown("<h1 style='text-align: center; font-size:25px; color: black;'>数据分布图", unsafe_allow_html=True)
        plot = plot_image_clusters(processed_dataset)
        st.altair_chart(plot)
