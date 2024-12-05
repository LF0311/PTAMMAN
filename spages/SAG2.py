# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def PebbleRateChart(mt, pb_rate_transient, pb_rate_ma, pb_rate_linear):
    # Add plot
    fig = go.Figure()
    # 绘制散点图
    fig.add_trace(go.Scatter(x = mt, y = pb_rate_transient, 
                            name = 'Transient Pebble Rate', mode = 'lines', opacity = .1, 
                            line = dict(color = 'black') 
                            ))
    # 绘制移动平均线
    fig.add_trace(go.Scatter(x = mt, y = pb_rate_ma, 
                             name = 'Moving Average Pebble Rate', mode = 'lines', 
                             line = dict(color='royalblue', width = 2) 
                             ))
    # 绘制线性拟合曲线
    fig.add_trace(
        go.Scatter(x = mt, y = pb_rate_linear, 
                   name = 'Linear Regression', mode = 'lines',  
                   line = dict(color = 'black', width = 3) 
                   ))
    fig.update_yaxes(title_text = "Pebble Rate (t/h)")
    fig.update_xaxes(title_text = "Cumulative MT Milled")

    # Update axis format
    fig.update_yaxes(range = [100, 800])

    # Update figure format
    fig.update_layout(
        margin = dict(l = 1, r = 1, t = 50, b = 50),
        template="simple_white"
    )

    fig.update_layout(
        showlegend = True,
        font = dict(
            size = 12,
            color = "Black"
        )
    )

    fig.update_layout(legend = dict(
            yanchor = "top",
            y = 0.99,
            xanchor = "left",
            x = 0.03
        )
    )

    return fig


def read_xlsx(path):
    df = pd.read_excel(path, header = None)
    df = df.fillna('')
    df.index = ['' for _ in range(len(df))]  # 动态设置索引长度
    return df


def GrateWearChart(mt, outer_grate, outer_pebble):
    # Add plot
    fig = go.Figure()
    # 绘制散点图
    fig.add_trace(go.Scatter(x = mt, y = outer_grate, 
                            name = '22mm Outer Grate:  OA = 24,451 MT + 151,708 [mm²]', mode = 'lines', 
                            line = dict(color = 'orange', width = 3) 
                            ))
    # 绘制移动平均线
    fig.add_trace(go.Scatter(x = mt, y = outer_pebble, 
                             name = '60mm/65mm Outer Grate:  OA = 13,258 MT + 240,166 [mm²]', mode = 'lines', 
                             line = dict(color='royalblue', width = 3) 
                             ))

    fig.update_yaxes(title_text = "Open Area - mm²")
    fig.update_xaxes(title_text = "Cumulative MT Milled")

    # Update axis format
    #fig.update_yaxes(range = [100, 650])

    # Update figure format
    fig.update_layout(
        margin = dict(l = 1, r = 1, t = 50, b = 50),
        template="simple_white"
    )

    fig.update_layout(
        showlegend = True,
        font = dict(
            size = 12,
            color = "Black"
        )
    )

    fig.update_layout(legend = dict(
            yanchor = "top",
            y = 0.99,
            xanchor = "left",
            x = 0.03
        )
    )

    return fig



def app():
    #########################################################################
    ############################## Section 1 ################################
    st.subheader("1. Campagin Pebble Rate Data - SuperVortex DE with Pebble Crusher Fully Off", divider = 'rainbow')
    # st.subheader("", divider='red')
    st.markdown("Pebble Rate vs Cumulative Tons Milled Correlation")

    st.latex(r'''
                Pebble Rate = 104.43 \times (Cumulative Tons Milled) + 373.56 \hspace{1em} [t/h]
            ''')

    #st.markdown('<style>h5{color: red; margin-left:10%; }</style>', unsafe_allow_html=True)
    #st.markdown(" <h5 > Pebble Rate = 38.154 x (Cumulative Milled Tons) +279.48 <h5>",unsafe_allow_html = True)
    start_row = 2  # Assuming the first row contains headers
    column_names = ['MT Processed', 'Pebble Rate (t/h)']
    
    # file_path = './resources/module1/RedialDE_SitePebbleRateData.xlsx'
    df = pd.read_excel('./resources/module2/SVDE_CrusherOff_SitePebbleRateData.xlsx', skiprows = start_row - 1)
    df.columns = column_names
    dfs = df
    df['Moving Average'] = df['Pebble Rate (t/h)'].rolling(window = 100).mean()
    
    # 线性拟合 | Linera Regression
    df_filtered = df[(df['MT Processed'] >= 0) & (df['MT Processed'] <= 1.6)]
    x = df_filtered['MT Processed']
    y = df_filtered['Pebble Rate (t/h)']
    coefficients = np.polyfit(x, y, 1)
    linear_fit = np.polyval(coefficients, x)

    plotlyChart = PebbleRateChart(df['MT Processed'], df['Pebble Rate (t/h)'], df['Moving Average'], linear_fit)
    st.plotly_chart(plotlyChart, use_container_width = True)
    
    # Some Space
    st.markdown("###")

    # show the recycle date
    df_selected = df.iloc[:, :2]
    # st.table(df_selected)
    st.markdown("Filtered Pebble Rate Data")
    st.dataframe(df_selected.style.format({"MT Processed": "{:.3f}", "Pebble Rate (t/h)": "{:.1f}"}), use_container_width = True, hide_index = True, column_config = {"MT Processed": {"alignment": "center"}, "Pebble Rate (t/h)": {"alignment": "center"}})
    # Some Space
    st.markdown("###")
    
    
    
    
    #########################################################################
    ############################## Section 2 ################################
    # 22mm Outer Grate
    st.subheader("2. Open Area Wear Tracking", divider = 'rainbow')
    st.markdown("22mm Outer Grate")
    radial22grate1, radial22grate2 = st.columns([2,3], gap="large", vertical_alignment="center")
    radial22grate1.image("./resources/module2/SVGrate.jpg", caption = "Grate Slot Locations")
    radial22grate2.image( "./resources/module2/SV_CoffGrateWear.jpg")
    with st.expander("22mm Outer Grate Wear Table"):
        df1 = read_xlsx("./resources/module2/SV_CoffGrateWear.xlsx")
        column_names1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']
        df1.columns = column_names1
        st.dataframe(df1, use_container_width = True, hide_index = True)
    
    # 65mm Outer Grate
    st.subheader("", divider='gray')
    st.markdown("60mm/65mm Outer Pebble")
    radial65pebble1, radial65pebble2 = st.columns([2,3], gap="large", vertical_alignment="center")
    radial65pebble1.image('./resources/module2/SVPebble.jpg', caption = "Pebble Slot Locations")
    radial65pebble2.image('./resources/module2/SV_CoffPebbleWear.jpg')
    with st.expander("60mm/65mm Outer Grate Wear Table"):
        df2 = read_xlsx("./resources/module2/SV_CoffPebbleWear.xlsx")
        column_names2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']
        df2.columns = column_names2
        st.dataframe(df2, use_container_width = True, hide_index = True)    
    st.subheader("", divider='gray')

    # Ploting the wear   
    st.markdown("###")
    st.markdown("Open Area vs Cumulative Milled Tons Correlation")
    # 创建一些示例数据
    # 假设Y的取值范围是从0到100
    X = np.arange(0, 4, 0.05)
    # 根据公式计算X的值
    Grate22mm = 24451 * X + 151708
    Pebble65mm = 13258 * X + 240166
    # 绘制原始数据点和拟合线
    fig2 = GrateWearChart(X, Grate22mm, Pebble65mm)
    st.plotly_chart(fig2, use_container_width = True)
    st.markdown("###")

    # Display Metrics and clac
    #st.subheader("", divider = 'gray')
    st.markdown("Open Area/Grate | Pebble Predictions")
    
    grate22, grate6065 = st.columns(2, gap="large", vertical_alignment="bottom")
    # 22mm Outer Grate
    grate22.markdown("22mm Grate")
    grate22input=grate22.text_input("Please input cumulative million tons","0", key="model2grate22")
    grate22.markdown("###")
    grate22.metric('Open Area Predicted - mm²:', int(24451*float(grate22input) + 151708))
    # 65mm Outer Grate    
    grate6065.markdown("60mm/65mm Grate")
    grate6065input=grate6065.text_input("Please input cumulative million tons","0", key="model2grate6065")
    grate6065.markdown("###")
    grate6065.metric('Open Area Predicted - mm²', int(13258*float(grate6065input) + 240166))

    
    #########################################################################
    ############################## Section 3 ################################
    # Display Pebble Rate Metrics
    st.markdown("###")
    st.subheader("3. Pebble Rate Forecast Modelling", divider='rainbow')
    st.markdown("Pebble Rate Generation / Each Liner")
    pebblerate1, pebblerate2 = st.columns(2, gap="medium")
    pebblerate1.markdown("22mm Outer Grate")
    pebblerate2.markdown("60mm/65mm Outer Grate")

    pebblerate1.metric(label="Pebble rate / part ratio",value="1.0")
    pebblerate1.metric(label="Pebble rate / part @ new - tph",value="5.56")
    pebblerate1.metric(label="Pebble rate / part @ 3.0 MT- tph",value="12.37")
    
    pebblerate2.metric(label="Pebble rate / part ratio",value="4.079")
    pebblerate2.metric(label="Pebble rate / part @ new - tph",value="22.68")
    pebblerate2.metric(label="Pebble rate / part @ 3.0 MT- tph",value="50.46")
    
    st.markdown("###")
    
    # Display User Input
    MTO=st.text_input("Please input cumulative million tons for forecast","0",key="model1MTO")
    
    X0input, Y0input = st.columns(2, gap="medium", vertical_alignment="bottom")
    
    X0 = X0input.number_input("Please input number of grates", min_value = 0, max_value = 36, value = 27, step = 1)
    Y0 = Y0input.number_input("Please input number of pebble", min_value = 0, max_value = 36, value = 9, step = 1)
    st.markdown("###")

    # Calculate
    calculate = st.button("Calculate Total Open Area and Forecast Pebble Rate", use_container_width=True)
    st.markdown("###")
    calculateresult1, calculateresult2 = st.columns(2)
    if calculate:
       if int(X0) + int(Y0) == 36:
        calculateresult1.metric(label="Total Open Area m² is:", value = round(((24451 * float(MTO) + 151708) * float(X0) + (13258 * float(MTO) + 240166) * float(Y0)) / 1000000,  2))
        calculateresult2.metric(label="Forecast Pebble Rate - t/h is:", value = round((2.27 * float(MTO) + 5.56) * float(X0) + (9.26 * float(MTO) + 22.68) * float(Y0), 2))
       else:
           st.error("Please check your input,grates plus pebble should be 36!")