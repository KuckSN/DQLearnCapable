import streamlit as st
import pandas as pd

st.set_page_config(
  page_title="Introduction",
  page_icon="👋",
)

col1, col2 = st.columns([25, 1])
with col1:
  st.write("# Welcome to DQLearnCapable")
with col2:
  st.image("./gdrive/MyDrive/images/logo.png", width=60)

st.sidebar.success("Navigation")
st.sidebar.write("Page 1 - Dashboard")
st.sidebar.write("Page 2 - Exploratory Data Analysis")
st.sidebar.write("Page 3 - Model Performance")
st.sidebar.write("Page 4 - Playground")

st.write("Threat Detector")
st.image("./gdrive/MyDrive/images/Cover_Page.jpg")

st.write("### Background")
st.write("As the technology has become advanced, variation of cyber threats has come to the surface of the networking world, posing new and dynamic attack challenges to network users. Hence, there is a demand for AI-assisted or self-governing threat detection systems that have capability of adaptive learning. Several reinforcement learning methods and traditional machine learning (ML) based methods for automated threat detection systems have been proposed in recent years. In this study, we introduce a new generation of threat detection methods with adaptive learning capability, which combines Q-learning of reinforcement learning with deep neural networks. The proposed Deep Q-Learning (DQL) model learns through a subset of the environment iteratively using trial-error approach to optimize its decision making. Several traditional ML-based approaches are built for benchmarking purposes. Through intensive comparison based on VHS-22 dataset, we can confirm that Random Forest Classifier (RF) and Extra Tree Classifier (ET) yield slightly better performance than DQL but DQL still outperforms other similar ML approaches. Another focus in this paper is to improve generalizability of the model, where we use heterogeneous datasets to train DQL with adaptive learning capability.")

st.markdown("### Navigation")
st.dataframe(pd.DataFrame({
    "Pages": ["Dashboard",
              "Exploratory Data Analysis",
              "Model Performance",
              "Playground",
              ],

    "Description": ["Embedded PowerBI interactive dashboard. Overview of Network Analysis.",
                    "Datasets used in DQLearnCapable and important features to identify threats.",
                    "Performance evaluation of 6 ML models.",
                    "Play around with embedded ML models to detect threats in your PCAP file.",
                    ]

  }))
