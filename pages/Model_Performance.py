import streamlit as st
import time
import pickle
import numpy as np
import plotly.graph_objects as go
import pandas as pd

@st.cache_data
def get_performance_data(file):
  with open(file, 'rb') as outfile:
    df = pickle.load(outfile)
  df = pd.DataFrame(df)
  return df

st.set_page_config(
  page_title="Model Performance",
  page_icon="📈",
)

st.markdown("## Model Performance")
st.write("""This demo illustrates a model performance. Enjoy!""")

st.sidebar.header("Model Performance")
with st.sidebar:
  type_performance = st.radio(
    "See performance based on:",
    ("Attack Network Flow only (0N:100A)", "Attack and Normal Network Flow (70N:30A)")
  )

if type_performance == "Attack Network Flow only (0N:100A)":
  df = get_performance_data("./gdrive/MyDrive/Model_test_performance/performance_full_attack.csv")
else:
  df = get_performance_data("./gdrive/MyDrive/Model_test_performance/performance_norm_attack.csv")

categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

fig = go.Figure()

for i in range(len(df)):
  fig.add_trace(go.Scatterpolar(
      r = df.iloc[[i]].values[:, 2:6][0],
      theta = categories,
      fill = 'toself',
      name = df.iloc[[i]].values[:, 0][0]
  ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=False
        )
    ),
    showlegend=True
)

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x = df['model'].values,
    y = df['accuracy'].values*100,
    name="Accuracy"
))

fig2.add_trace(go.Bar(
    x = df['model'].values,
    y = df['precision'].values*100,
    name="Precision"
))

fig2.add_trace(go.Bar(
    x = df['model'].values,
    y = df['recall'].values*100,
    name="Recall"
))

fig2.add_trace(go.Bar(
    x = df['model'].values,
    y = df['f1'].values*100,
    name="F1 Score"
))

fig2.add_trace(go.Scatter(
    x = df['model'].values,
    y = df['score_time'].values,
    name="Score Time (s)",
    connectgaps=True
))



st.write("### Performance Overview",fig)
st.write("### Model Performance with Time", fig2)
st.write("### Model Evaluation Details")
st.dataframe(df, use_container_width=True)

st.write("")
st.write("### Details of each model")

with st.expander("##### Deep Q-Learning Based Reinforcement Learning (DQL)"):
  st.write("Deep Q-Learning (DQL) model provides an ongoing auto-learning capability for a network environment that can detect different types of network intrusions using an automated trial-error approach and continuously enhance its detection capabilities.")
  st.write("")
  st.write("Flow Chart")
  st.image("./gdrive/MyDrive/images/Figure 7 DQL traning flow chart.png")
  st.write("")
  st.write("Model Diagram")
  st.image("./gdrive/MyDrive/images/Figure 8 DQL learning process.png")
  st.write("")
  st.write("Hyperparameter")
  st.dataframe(pd.DataFrame({
    "Hyperparameter": ["Neural Network Layer",
                       "Hidden Layer Size",
                       "Batch Size",
                       "Number of Episode",
                       "Episode Iteration",
                       "Epsilon",
                       "Decay Rate",
                       "Gamma",
                       ],

    "Value": ["4 [Input, Hidden, Hidden, Output]",
              "100",
              "500",
              "200",
              "100",
              "0.1",
              "0.99",
              "0.001",
              ],

    "Description": ["This refers to the number of layers in the neural network used to approximate the Q-function in DQL. The complexity of the problem often dictates the number of layers.",
                    "This is the number of neurons in each hidden layer of the neural network. A larger size can increase the capacity of the model to learn complex patterns, but it may also lead to overfitting.",
                    "This is the number of experiences sampled from the memory to train the network at each step. A larger batch size can lead to more stable updates, but it also requires more computational resources.",
                    "This is the number of complete sequences of interaction between the agent and the environment. Each episode is a complete game or sequence from start to finish.",
                    "This refers to the maximum number of steps in each episode. It determines how long an episode can last",
                    "This is the exploration rate in the epsilon-greedy policy. It determines the probability of taking a random action. A high epsilon value encourages more exploration, while a low value encourages more exploitation. The epsilon will change over the training period based on gamma and decay rate",
                    "This is the rate at which epsilon decreases over time. A high decay rate makes the agent shift from exploration to exploitation more quickly",
                    "This is the discount factor used in the Q-learning update. It determines the importance of future rewards. A gamma close to 0 makes the agent short-sighted by only considering current rewards, while a gamma close to 1 makes the agent aim for long-term rewards",
                    ]

  }))

with st.expander("##### Random Forest Classifier (RF)"):
  st.write("A Random Forest Classifier (RF) is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.")
  st.write("")
  st.write("Hyperparameter")
  st.dataframe(pd.DataFrame({
    "Hyperparameter": ["n_estimators",
                       "min_samples_split",
                       "min_samples_leaf",
                       "max_features",
                       "max_depth",
                       "bootstrap",
                       ],

    "Value": ["800",
              "2",
              "2",
              "auto",
              "50",
              "False",
              ],

    "Description": ["The number of trees in the forest. More trees can lead to better performance but also to longer computation time.",
                    "The minimum number of samples required to split an internal node. If the value is an integer, it is considered as the minimum number. If it's a float, it's a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.",
                    "The minimum number of samples required to be at a leaf node. This parameter prevents splitting nodes in a way that it leaves too few samples in any of the child nodes.",
                    "The number of features to consider when looking for the best split. It can be an integer, float or string (sqrt, log2, None).",
                    "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.",
                    "Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree."
                    ]

  }))

with st.expander("##### Extra Tree Classifier (ET)"):
  st.write("The Extra Tree Classifier (ET) a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.")
  st.write("")
  st.write("Hyperparameter")
  st.dataframe(pd.DataFrame({
    "Hyperparameter": ["n_estimators",
                       "min_samples_split",
                       "min_samples_leaf",
                       "max_features",
                       "max_depth",
                       "bootstrap",
                       ],

    "Value": ["800",
              "10",
              "1",
              "sqrt",
              "70",
              "False",
              ],

    "Description": ["The number of trees in the forest. More trees can lead to better performance but also to longer computation time.",
                    "The minimum number of samples required to split an internal node. If the value is an integer, it is considered as the minimum number. If it's a float, it's a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.",
                    "The minimum number of samples required to be at a leaf node. This parameter prevents splitting nodes in a way that it leaves too few samples in any of the child nodes.",
                    "The number of features to consider when looking for the best split. It can be an integer, float or string (sqrt, log2, None).",
                    "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.",
                    "Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree."
                    ]

  }))

with st.expander("##### Linear Support Vector Machine (LSVC)"):
  st.write("Linear Support Vector Machine (LSVC) is a supervised machine learning algorithm specifically designed for classifying linearly separable data. It constructs a hyperplane in high-dimensional space that separates different classes with the largest margin possible, aiming for a robust and generalizable model.")
  st.write("")
  st.write("Hyperparameter")
  st.dataframe(pd.DataFrame({
    "Hyperparameter": ["penalty",
                       "loss",
                       "C",
                       ],

    "Value": ["l2",
              "hinge",
              "0.8",
              ],

    "Description": ["Controls the type of regularization used to prevent overfitting. L2 regularization (Ridge) encourages smaller weights overall, making the model less sensitive to outliers.",
                    "Specifies the loss function used to measure model errors during training. The 'hinge' is standard hinge loss for SVMs",
                    "Inverse of regularization strength. Higher C values prioritize minimizing training errors, increasing model complexity. Lower C values emphasize regularization, reducing model complexity and potentially improving generalization.",
                    ]

  }))

with st.expander("##### Stochastic Gradient Descent (SGD)"):
  st.write("Stochastic Gradient Descent (SGD) is an efficient optimization algorithm often used to train machine learning models, especially with large datasets. It iteratively updates model parameters to minimize a loss function, but unlike traditional gradient descent, it processes data samples one at a time (or in small batches) rather than the entire dataset in each iteration. This makes it faster and more memory-efficient, especially for large-scale problems.")
  st.write("")
  st.write("Hyperparameter")
  st.dataframe(pd.DataFrame({
    "Hyperparameter": ["penalty",
                       "loss",
                       "max_iter",
                       ],

    "Value": ["l1",
              "modified_huber",
              "2",
              ],

    "Description": ["Specifies the type of regularization to apply. L1 regularization for potential feature selection.",
                    "Determines the loss function used to measure error during training. ",
                    "Sets the maximum number of iterations (epochs) for the training process.",
                    ]

  }))

with st.expander("##### Logistic Regression (LR)"):
  st.write("Logistic Regression (LR) is a popular statistical method used for binary classification tasks. It estimates the probability of an instance belonging to a particular class (usually 0 or 1) based on its features. It's known for its interpretability and relatively simple implementation.")
  st.write("")
  st.write("Hyperparameter")
  st.dataframe(pd.DataFrame({
    "Hyperparameter": ["solver",
                       "C",
                       ],

    "Value": ["lbfgs",
              "0.01",
              ],

    "Description": ["Specifies the algorithm used to optimize the model's parameters. 'lbfgs': Limited-memory BFGS, an approximation of Newton's method that's less memory-intensive.",
                    "Inverse of regularization strength.",
                    ]

  }))
