# -*- coding:utf-8 -*-
"""
:Date: 2023-02-19 13:59:16
:LastEditTime: 2023-02-19 13:59:16
:Description: 
"""
# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:58:23
:LastEditTime: 2023-02-18 23:58:24
:Description: 
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import PIL
def write():
    # Global Variables
    theme_plotly = None # None or streamlit
    week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Layout
    # st.set_page_config(page_title='Macro - NEAR Mega Dashboard', layout='wide')
    st.title('Data Exploratory Analysis')

    # Style
    # with open('style.css')as f:
    #     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

    # Data Sources
    @st.cache(ttl=1000, allow_output_mutation=True)
    def get_data(query):
        if query == 'Prices Daily':
            return pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/60300b70-dd1e-4716-bc75-3bfc5709250f/data/latest')
        elif query == 'Blocks Overview':
            return pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/024b2e03-1063-4bcf-a8de-b35d17e01cbd/data/latest')
        elif query == 'Blocks Daily':
            return pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/1d55b381-77a7-4e03-b951-58421139cb09/data/latest')
        elif query == 'Transactions Overview':
            return pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/3479cc40-da43-4231-b8e8-c5e62974720d/data/latest')
        elif query == 'Transactions Daily':
            return pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/c984af6f-5955-45f4-bffd-2bab056ee78f/data/latest')
        elif query == 'Transactions Heatmap':
            return pd.read_json('https://node-api.flipsidecrypto.com/api/v2/queries/f0f4e88d-4c8c-4ed8-acf9-ccbd213bcec1/data/latest')
        elif query == 'Transactions Status Overview':
            return pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/bb9626e1-16ac-49dc-a101-54622b0bc96c/data/latest')
        elif query == 'Transactions Status Daily':
            return pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/1979b40a-c981-486f-a4b6-ca774cc835f5/data/latest')
        return None

    prices_daily = get_data('Prices Daily')
    blocks_overview = get_data('Blocks Overview')
    blocks_daily = get_data('Blocks Daily')
    transactions_overview = get_data('Transactions Overview')
    transactions_daily = get_data('Transactions Daily')
    transactions_heatmap = get_data('Transactions Heatmap')
    transactions_status_overview = get_data('Transactions Status Overview')
    transactions_status_daily = get_data('Transactions Status Daily')

    # Content
    tab_overview, tab_heatmap, tab_status = st.tabs(['**Overview**', '**Heatmap**', '**Success Rate**'])

    with tab_overview:
        
        st.subheader('Overview')

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label='**Total Blocks**', value=str(blocks_overview['Blocks'].map('{:,.0f}'.format).values[0]))
            st.metric(label='**Average Block Time**', value=blocks_overview['BlockTime'].round(2), help='seconds')
        with c2:
            st.metric(label='**Total Transactions**', value=str(transactions_overview['Transactions'].map('{:,.0f}'.format).values[0]))
            st.metric(label='**Average TPS**', value=str(transactions_overview['TPS'].map('{:,.2f}'.format).values[0]))
        with c3:
            st.metric(label='**Total Unique Addresses**', value=str(transactions_overview['Users'].map('{:,.0f}'.format).values[0]))
            st.metric(label='**Average Daily Active Users**', value=str(transactions_overview['Users/Day'].map('{:,.0f}'.format).values[0]))

        st.subheader('Activity Over Time')

        interval = st.radio('**Time Interval**', ['Daily', 'Weekly', 'Monthly'], key='transactions_interval', horizontal=True)

        if st.session_state.transactions_interval == 'Daily':
            price_over_time = prices_daily
            blocks_over_time = blocks_daily
            transactions_over_time = transactions_daily
        elif st.session_state.transactions_interval == 'Weekly':
            price_over_time = prices_daily
            price_over_time = price_over_time.groupby([pd.Grouper(freq='W', key='Date')]).agg('mean').reset_index()
            blocks_over_time = blocks_daily
            blocks_over_time = blocks_over_time.groupby([pd.Grouper(freq='W', key='Date')]).agg(
                {'Blocks': 'sum', 'Transactions': 'sum', 'Validators': 'sum', 'BlockTime': 'mean'}).reset_index()
            transactions_over_time = transactions_daily
            transactions_over_time = transactions_over_time.groupby([pd.Grouper(freq='W', key='Date')]).agg(
                {'Blocks': 'sum', 'Transactions': 'sum', 'Users': 'sum', 'TPS': 'mean'}).reset_index()
        elif st.session_state.transactions_interval == 'Monthly':
            price_over_time = prices_daily
            price_over_time = price_over_time.groupby([pd.Grouper(freq='MS', key='Date')]).agg('mean').reset_index()
            blocks_over_time = blocks_daily
            blocks_over_time = blocks_over_time.groupby([pd.Grouper(freq='MS', key='Date')]).agg(
                {'Blocks': 'sum', 'Transactions': 'sum', 'Validators': 'sum', 'BlockTime': 'mean'}).reset_index()
            transactions_over_time = transactions_daily
            transactions_over_time = transactions_over_time.groupby([pd.Grouper(freq='MS', key='Date')]).agg(
                {'Blocks': 'sum', 'Transactions': 'sum', 'Users': 'sum', 'TPS': 'mean'}).reset_index()

        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Line(x=price_over_time['Date'], y=price_over_time['Price'], name='Price'), secondary_y=False)
        fig.add_trace(go.Bar(x=price_over_time['Date'], y=price_over_time['Change'], name='Change'), secondary_y=True)
        fig.update_layout(title_text='NEAR Price and Its Percentage Change Over Time')
        fig.update_yaxes(title_text='Price [USD]', secondary_y=False)
        fig.update_yaxes(title_text='Change [%]', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Line(x=blocks_over_time['Date'], y=blocks_over_time['Blocks'], name='Blocks'), secondary_y=False)
        fig.add_trace(go.Line(x=transactions_over_time['Date'], y=transactions_over_time['Transactions'], name='Transactions'), secondary_y=True)
        fig.update_layout(title_text='Number of Blocks and Transactions Over Time')
        fig.update_yaxes(title_text='Blocks', secondary_y=False)
        fig.update_yaxes(title_text='Transactions', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Line(x=blocks_over_time['Date'], y=blocks_over_time['BlockTime'].round(2), name='Block Time'), secondary_y=False)
        fig.add_trace(go.Line(x=transactions_over_time['Date'], y=transactions_over_time['TPS'].round(2), name='TPS'), secondary_y=True)
        fig.update_layout(title_text='Average Block Time and TPS Over Time')
        fig.update_yaxes(title_text='Block Time [s]', secondary_y=False)
        fig.update_yaxes(title_text='TPS', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

        fig = px.area(transactions_over_time, x='Date', y='Users', title='Active Addresses Over Time')
        fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title=None)
        fig.add_annotation(x='2022-09-13', y=200000, text='SWEAT', showarrow=False, xanchor='left')
        fig.add_shape(type='line', x0='2022-09-13', x1='2022-09-13', y0=0, y1=1, xref='x', yref='paper', line=dict(width=1, dash='dot'))
        fig.add_annotation(x='2022-05-12', y=200000, text='Terra Collapse', showarrow=False, xanchor='left')
        fig.add_shape(type='line', x0='2022-05-12', x1='2022-05-12', y0=0, y1=1, xref='x', yref='paper', line=dict(width=1, dash='dot'))
        fig.add_annotation(x='2022-11-10', y=200000, text='FTX Collapse', showarrow=False, xanchor='left')
        fig.add_shape(type='line', x0='2022-11-10', x1='2022-11-10', y0=0, y1=1, xref='x', yref='paper', line=dict(width=1, dash='dot'))
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

    with tab_heatmap:

        st.subheader('Activity Heatmap')

        fig = px.density_heatmap(transactions_heatmap, x='Hour', y='Day', z='Transactions', histfunc='avg', title='Heatmap of Transactions', nbinsx=24)
        fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title=None, xaxis={'dtick': 1}, coloraxis_colorbar=dict(title='Transactions'))
        fig.update_yaxes(categoryorder='array', categoryarray=week_days)
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

        fig = px.density_heatmap(transactions_heatmap, x='Hour', y='Day', z='Blocks', histfunc='avg', title='Heatmap of Blocks', nbinsx=24)
        fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title=None, xaxis={'dtick': 1}, coloraxis_colorbar=dict(title='Blocks'))
        fig.update_yaxes(categoryorder='array', categoryarray=week_days)
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

        fig = px.density_heatmap(transactions_heatmap, x='Hour', y='Day', z='Users', histfunc='avg', title='Heatmap of Active Addresses', nbinsx=24)
        fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title=None, xaxis={'dtick': 1}, coloraxis_colorbar=dict(title='Users'))
        fig.update_yaxes(categoryorder='array', categoryarray=week_days)
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

    with tab_status:

        st.subheader('Overview')

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.pie(transactions_status_overview, values='Transactions', names='Status', title='Share of Total Transactions', hole=0.4)
            fig.update_traces(showlegend=False, textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)
        with c2:
            fig = px.pie(transactions_status_overview, values='Users', names='Status', title='Share of Total Users', hole=0.4)
            fig.update_traces(showlegend=False, textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)
        with c3:
            fig = px.pie(transactions_status_overview, values='Fees', names='Status', title='Share of Total Fees', hole=0.4)
            fig.update_traces(showlegend=False, textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

        st.subheader('Success Rate Over Time')

        interval = st.radio('**Time Interval**', ['Daily', 'Weekly', 'Monthly'], key='success_interval', horizontal=True)

        if st.session_state.success_interval == 'Daily':
            df = transactions_status_daily
        elif st.session_state.success_interval == 'Weekly':
            df = transactions_status_daily
            df = df.groupby([pd.Grouper(freq='W', key='Date'), 'Status']).agg(
                {'Blocks': 'sum', 'Transactions': 'sum', 'Users': 'sum', 'Gas': 'sum', 'Fees': 'sum'}).reset_index()
        elif st.session_state.success_interval == 'Monthly':
            df = transactions_status_daily
            df = df.groupby([pd.Grouper(freq='MS', key='Date'), 'Status']).agg(
                {'Blocks': 'sum', 'Transactions': 'sum', 'Users': 'sum', 'Gas': 'sum', 'Fees': 'sum'}).reset_index()

        fig = px.line(df, x='Date', y='Transactions', color='Status', custom_data=['Status'], title='Transactions Over Time')
        fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title='Transactions', hovermode='x unified')
        fig.update_traces(hovertemplate='%{customdata}: %{y:,.0f}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

        fig = px.line(df, x='Date', y='Users', color='Status', custom_data=['Status'], title='Users Over Time')
        fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title='Transactions', hovermode='x unified')
        fig.update_traces(hovertemplate='%{customdata}: %{y:,.0f}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

        fig = px.line(df, x='Date', y='Fees', color='Status', custom_data=['Status'], title='Fees Over Time')
        fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title='Fees [USD]', hovermode='x unified')
        fig.update_traces(hovertemplate='%{customdata}: %{y:,.2f}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)