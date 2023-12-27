import streamlit as st
import pandas as pd
import altair as altair
from urllib.error import URLError
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
  page_title="Exploratory Data Analysis",
  page_icon="📊"
)

st.markdown("## Exploratory Data Analysis")
st.sidebar.header("Exploratory Data Analysis")
st.write(
  """This demo shows dataset used by this project. """
)

@st.cache_data
def get_network_data():
  df = pd.read_csv('./gdrive/MyDrive/norm_attack_test.csv')
  df_formated = pd.read_csv('./gdrive/MyDrive/formated_norm_attack_test.csv')
  return df

@st.cache_data
def get_formated_network_data():
  df_formated = pd.read_csv('./gdrive/MyDrive/formated_norm_attack_test.csv')
  return df_formated



try:

  top10 = ['ip_protocol', 'sport', 'b_packet_max', 'b_packet_median',
       'b_packet_third_q', 'connections_from_this_port',
       'connections_ratio_to_this_host', 'connections_ratio_from_this_port',
       'PCA1', 'MDSAE2', 'label']

  df = get_network_data()
  df_formated=get_formated_network_data()

  st.write("### Dataset Description")
  st.dataframe(pd.DataFrame({
    "Aspect": ["Name",
              "Integrated Dataset",
              "Size",
              "Sampled Size",
              "Features",
              "No. Features",
              "Author",
              "Year Created",
              "Time Span",
              "License"
              ],

    "Description": ["VHS-22",
                    "ISOT, CICIDS2017, Booters, CTU-13, MTA",
                    "27.7 million (20.3 million legitimate and 7.4 million attacks)",
                    "1 million (733k legitimate and 267k attack)",
                    "Flow-level and Network-level information",
                    "45 + 3 (labels)",
                    "Paweł Szumełda, Natan Orzechowski, Mariusz Rawski, and Artur Janicki",
                    "2022",
                    "1 Jan 2022 - 23 Jan 2022",
                    "Attribution 4.0 International (CC BY 4.0)"
                    ]

  }))

  st.write("### Network Traffic Flow Dataset Overview", df.head(10))

  st.write("### Statistical Summary", df.describe())

  xticklabels = ['1','2','3','4','5','6','7','8','9','10', 'label']
  yticklabels = ['ip_p[1]', 'sport[2]', 'b_p_max[3]', 'b_p_med[4]',
       'b_p_3q[5]', 'con_f_p[6]',
       'con_r_to_h[7]', 'con_r_f_p[8]',
       'PCA1[9]', 'MDSAE2[10]', 'label']

  st.write("### Correlation Graph of Chosen Features")
  fig, ax = plt.subplots()
  sns.heatmap(df_formated[top10].corr(), annot=True, cmap="coolwarm", xticklabels=xticklabels, yticklabels=yticklabels)
  st.pyplot(fig)

  st.write("### Metadata of Chosen Features")
  st.write("10 most correlated features are chosen through the variance threshold filter, chi-squared filter and pairwise correlation function.")
  table = pd.DataFrame({
    "Features": ['ip_protocol',
                 'sport',
                 'b_packet_max',
                 'b_packet_median',
                 'b_packet_third_q',
                 'connections_from_this_port',
                 'connections_ratio_to_this_host',
                 'connections_ratio_from_this_port',
                 'PCA1',
                 'MDSAE2',
                 'label'],

    "Acronym": ['ip_p[1]',
                'sport[2]',
                'b_p_max[3]',
                'b_p_med[4]',
                'b_p_3q[5]',
                'con_f_p[6]',
                'con_r_to_h[7]',
                'con_r_f_p[8]',
                'PCA1[9]',
                'MDSAE2[10]',
                'label'],

    "Description": ["Fourth layer protocol",
                    "Source port",
                    "Size of the largest packet",
                    "Median packet size",
                    "3rd quantile of packet size",
                    "No. of connections with the same source port number",
                    "% of connections from the host with the same source address",
                    "% of connections from host with the same source port",
                    "Engineered feature by principal component analysis using 45 original features",
                    "Engineered feature by multi-noise autoenconder using 45 original features",
                    'Label for Attack or Normal traffic.']

    })
  st.table(table)



except URLError as e:
  st.error(
    """**This demo requires internet access. **
    Connection error: %s
    """
    % e.reason
  )
