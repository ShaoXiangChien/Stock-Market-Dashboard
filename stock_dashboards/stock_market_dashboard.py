import streamlit as st
import pandas as pd
import numpy as np
import plotly as plt
import requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly


st.sidebar.title("Dashboard configuration")
option = st.sidebar.selectbox(
    "Options", ("Stocktwits", "Chart", "Forecast"), 2)


st.header(option)


if option == "Chart":
    stock_id = st.sidebar.text_input("輸入股票代號", "0056")
    st.subheader(stock_id)
    stock = yf.Ticker(stock_id + ".TW").history(period='1y')
    Date = [date.strftime("%Y/%m/%d") for date in stock.index]
    stock.index = Date
    fig = go.Figure(data=[go.Candlestick(x=Date,
                                         open=stock['Open'],
                                         high=stock['High'],
                                         low=stock['Low'],
                                         close=stock['Close'],
                                         name=stock_id)])

    fig.update_xaxes(type='category')
    fig.update_layout(height=600)

    st.plotly_chart(fig, user_container_width=True)

    st.write(stock)
    st.write("Recommendations")
    yf.Ticker(stock_id+".TW").info


if option == "Stocktwits":
    # st.subheader("stockWits")
    stock_name = st.sidebar.text_input("Stock Symbol", "AAPL")
    r = requests.get(
        f"https://api.stocktwits.com/api/2/streams/symbol/{stock_name}.json")
    data = r.json()
    # st.write(data)
    st.image(f"https://finviz.com/chart.ashx?t={stock_name}")
    st.write(f"Discussions about {stock_name}")
    for message in data["messages"]:
        st.image(message['user']['avatar_url'])
        st.write(message['user']['username'])
        st.write(message['created_at'])
        st.write(message['body'])
        st.write("---")


if option == "Forecast":
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    st.title("Stock Prediction")
    stocks = ('GOOG', 'TSLA', 'AAPL', 'GME', 'MSFT')
    selected_box = st.selectbox("Select dataset for prediction", stocks)
    n_years = st.slider("Years of Prediction", 1, 4)
    period = n_years * 365

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("Loading data...")
    data = load_data(selected_box)
    data_load_state.text("Loading data...done!")

    st.subheader("Raw Data")
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text="Time Series Data",
                          xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Forecasting
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader("Forecast data")
    st.write(forecast.tail())

    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast Component")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
