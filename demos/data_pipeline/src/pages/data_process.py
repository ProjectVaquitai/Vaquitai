# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:49:03
:LastEditTime: 2023-02-18 23:49:37
:Description: 
"""
import copy
import os

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core import Analyser, Executor
from data_juicer.ops.base_op import OPERATORS


@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf_8_sig')


@st.cache_data
def convert_to_jsonl(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_json(orient='records', lines=True,
                      force_ascii=False).encode('utf_8_sig')


def pretty_out(d):
    res = ''
    process = ''
    op_names = set(OPERATORS.modules.keys())
    for key, value in d.items():
        if key == 'process':
            process = yaml.dump(value,
                                allow_unicode=True,
                                default_flow_style=False)
        elif key == 'config' or key.split('.')[0] in op_names:
            continue
        else:
            res += f'{key}:\n \t {value}\n'
    res += 'process:\n' + \
           '\n'.join(['\t' + line for line in process.splitlines()])

    return res



def parse_cfg():

    cfg_file = st.session_state.input_cfg_file
    cfg_cmd = st.session_state.input_cfg_cmd

    cfg_f_name = 'null'
    del_cfg_file = False
    if cfg_file is not None:
        cfg_f_name = cfg_file.name
        file_contents = cfg_file.getvalue()
        with open(cfg_f_name, 'wb') as f:
            f.write(file_contents)
        cfg_cmd = f'--config {cfg_f_name}'
        del_cfg_file = True

    args_in_cmd = cfg_cmd.split()

    if len(args_in_cmd) >= 2 and args_in_cmd[0] == '--config':
        cfg_f_name = args_in_cmd[1]
    else:
        st.warning('Please specify a config command or upload a config file.')
        st.stop()

    if not os.path.exists(cfg_f_name):
        st.warning('do not parse'
                   f'config file does not exist with cfg_f_name={cfg_f_name}')
        st.stop()

    with open(cfg_f_name, 'r') as cfg_f:
        specified_cfg = yaml.safe_load(cfg_f)

    try:
        parsed_cfg = init_configs(args=args_in_cmd)
        st.session_state.cfg = parsed_cfg
        if del_cfg_file:
            os.remove(cfg_f_name)
        return pretty_out(parsed_cfg), pretty_out(specified_cfg), parsed_cfg
    except Exception as e:
        return str(e), pretty_out(specified_cfg), None


def analyze_and_show_res():
    images_ori = []
    cfg = st.session_state.get('cfg', parse_cfg()[2])
    if cfg is None:
        raise ValueError('you have not specify valid cfg')
    # force generating separate figures
    cfg['save_stats_in_one_file'] = True

    logger.info('=========Stage 1: analyze original data=========')
    analyzer = Analyser(cfg)
    analyzed_dataset = analyzer.run()

    overall_file = os.path.join(analyzer.analysis_path, 'overall.csv')
    analysis_res_ori = pd.DataFrame()
    if os.path.exists(overall_file):
        analysis_res_ori = pd.read_csv(overall_file)

    if os.path.exists(analyzer.analysis_path):
        for f_path in os.listdir(analyzer.analysis_path):
            if '.png' in f_path and 'all-stats' in f_path:
                images_ori.append(os.path.join(analyzer.analysis_path, f_path))

    st.session_state.analyzed_dataset = analyzed_dataset
    st.session_state.original_overall = analysis_res_ori
    st.session_state.original_imgs = images_ori


def process_and_show_res():
    images_processed = []
    cfg = st.session_state.get('cfg', parse_cfg()[2])
    if cfg is None:
        raise ValueError('you have not specify valid cfg')
    # force generating separate figures
    cfg['save_stats_in_one_file'] = True
    logger.info('=========Stage 2: process original data=========')
    executor = Executor(cfg)
    processed_dataset = executor.run()


    logger.info('=========Stage 3: analyze the processed data==========')
    # analysis_res_processed = pd.DataFrame()
    # try:
    #     if len(processed_dataset) > 0:
    #         cfg_for_processed_data = copy.deepcopy(cfg)
    #         cfg_for_processed_data.dataset_path = cfg.export_path

    #         cfg_for_processed_data.export_path = os.path.dirname(
    #             cfg.export_path) + '_processed/data.jsonl'

    #         analyzer = Analyser(cfg_for_processed_data)
    #         analyzer.analysis_path = os.path.dirname(
    #             cfg_for_processed_data.export_path) + '/analysis'
    #         analyzer.run()

    #         overall_file = os.path.join(analyzer.analysis_path, 'overall.csv')
    #         if os.path.exists(overall_file):
    #             analysis_res_processed = pd.read_csv(overall_file)

    #         if os.path.exists(analyzer.analysis_path):
    #             for f_path in os.listdir(analyzer.analysis_path):
    #                 if '.png' in f_path and 'all-stats' in f_path:
    #                     images_processed.append(
    #                         os.path.join(analyzer.analysis_path, f_path))
    #     else:
    #         st.warning('No sample left after processing. Please change \
    #             anther dataset or op parameters then rerun')
    # except Exception as e:
    #     st.warning(f'Something error with {str(e)}')

    logger.info('=========Stage 4: Render the analysis results==========')
    st.session_state.processed_dataset = processed_dataset
    # st.session_state.processed_overall = analysis_res_processed
    # st.session_state.processed_imgs = images_processed


class Visualize:

    @staticmethod
    def setup():
        st.markdown(
            '<div align = "center"> <font size = "30"> 数据处理 \
            </font> </div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def parser():
        with st.expander('配置', expanded=True):
            st.markdown('请指定配置文件：可通过&nbsp;&nbsp;&nbsp;方式一：配置文件路径或者&nbsp;&nbsp;&nbsp;方式二：上传配置文件', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                example_cfg_f = './configs/demo.yaml'
                example_cfg_f = os.path.abspath(example_cfg_f)

                st.text_area(label='方式一：配置文件路径',
                             key='input_cfg_cmd',
                             value=f'--config {example_cfg_f}')


            with col2:
                st.file_uploader(label='方式二：上传配置文件',
                                 key='input_cfg_file',
                                 type=['yaml'])

            btn_show_cfg = st.button('1. 解析配置文件', use_container_width=True)
            if btn_show_cfg:
                text1, text2, cfg = parse_cfg()
                st.session_state.cfg_text1 = text1
                st.session_state.cfg_text2 = text2

            else:
                text1 = st.session_state.get('cfg_text1', '')
                text2 = st.session_state.get('cfg_text2', '')
            
            st.text_area(label='配置文件解析结果', value=text2)


    @staticmethod
    def analyze_process():
        start_btn_process = st.button('3. 执行数据处理',
                                      type = 'primary',
                                      use_container_width=True)

        with st.expander('Data Analysis Results', expanded=True):
            if start_btn_process:
                with st.spinner('Wait for process...'):
                    process_and_show_res()
                st.markdown('<font color="green">数据处理已完成，请查看数据分析结果</font>', unsafe_allow_html=True)

            # original_overall = st.session_state.get('original_overall', None)
            # original_imgs = st.session_state.get('original_imgs', [])
            # processed_overall = st.session_state.get('processed_overall', None)
            # processed_imgs = st.session_state.get('processed_imgs', [])



    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.parser()
        Visualize.analyze_process()


def write():
    Visualize.visualize()


if __name__ == '__main__':
    write()
