
import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError
from tensorflow import keras
import numpy as np
import joblib
import tensorflow as tf
from keras.models import model_from_json
import pickle
import json

top10 = ['ip_protocol', 'sport', 'b_packet_max', 'b_packet_median',
       'b_packet_third_q', 'connections_from_this_port',
       'connections_ratio_to_this_host', 'connections_ratio_from_this_port',
       'PCA1', 'MDSAE2', 'label']

def formating (X):
    @keras.saving.register_keras_serializable(package="MyLayers")
    class Sampling(keras.layers.Layer):
      def call(self, input_data):
        mean, log_var = input_data
        return keras.backend.random_normal(tf.shape(log_var)) * keras.backend.exp(log_var / 2) + mean

    X = X

    columns = ["ip_src_str", "ip_dst_str", "ip_protocol", "sport", "dport", "in_packets",
            "b_packet_total", "first_timestamp", "last_timestamp", "duration", "flags_sum",
            "urg_nr_count", "ack_nr_count", "rst_nr_count", "fin_nr_count", "psh_nr_count", "syn_nr_count",
            "b_packet_max", "b_packet_min", "b_packet_mean", "b_packet_median", "b_packet_first_q",
            "b_packet_third_q", "b_packet_std", "iat_min", "iat_max", "iat_first_q",
            "iat_third_q", "iat_std", "iat_mean", "iat_median", "iat_var",
            "connections_from_this_host", "connections_to_this_host", "connections_rst_to_this_host",
            "connections_rst_from_this_host", "connections_to_this_port", "connections_from_this_port",
            "connections_ratio_from_this_host", "connections_ratio_to_this_host", "connections_ratio_rst_to_this_host",
            "connections_ratio_rst_from_this_host", "connections_ratio_to_this_port", "connections_ratio_from_this_port",
            "label", "attack_label", "attack_file"]

    X = X.reindex(columns=columns)

    to_drop = ['connections_ratio_rst_from_this_host', #0 values, meaningless
            'connections_rst_from_this_host', #0 values, meaningless
            'iat_third_q', #0 values, meaningless
            'iat_first_q', #0 values, meaningless
            "urg_nr_count", #covered by flags_sum
            "ack_nr_count", #covered by flags_sum
            "rst_nr_count", #covered by flags_sum
            "fin_nr_count", #covered by flags_sum
            "psh_nr_count", #covered by flags_sum
            "syn_nr_count", #covered by flags_sum
            "attack_label",
            "attack_file",
            "Unnamed: 0.1",
            "Unnamed: 0"
          ]

    X.drop(columns=to_drop, axis=1, inplace=True, errors='ignore')

    ori_label = X.columns[2:34]
    ori_label = ori_label.drop(['first_timestamp', 'last_timestamp'], errors='ignore')

    array = X.values
    X_train = array[:, 2:34]
    X_train = np.delete(X_train, [5,6], axis=1)#ip address and timestamp data excluded
    y_train = array[:, 34]
    y_train = y_train.astype('int32')

    # joblib.dump(norm_scaler, './gdrive/MyDrive/clean/scaler.save')
    norm_scaler = joblib.load('./gdrive/MyDrive/clean/scaler.save')
    n_X_train = norm_scaler.transform(X_train)

    # var_encoder.save('./gdrive/MyDrive/clean/var_encoder_v1.h5')
    var_encoder = tf.keras.models.load_model('./gdrive/MyDrive/clean/var_encoder_v1.h5', compile=False)
    encoded = var_encoder
    gen_feat = encoded.predict(n_X_train)
    gen_featdf = pd.DataFrame(gen_feat)
    # joblib.dump(norm_scaler4, './gdrive/MyDrive/clean/vae_scaler.save')
    norm_scaler4 = joblib.load('./gdrive/MyDrive/clean/vae_scaler.save')
    n_gen_feat = norm_scaler4.transform(gen_feat)
    n_gen_feat = pd.DataFrame(n_gen_feat)
    new_col_headers = ["VAE1", "VAE2", "VAE3", "VAE4"]
    n_gen_feat.columns = new_col_headers

    # joblib.dump(pca, './gdrive/MyDrive/clean/pca.sav')
    pca = joblib.load('./gdrive/MyDrive/clean/pca.sav')
    X_pca_train = pca.fit_transform(n_X_train)
    pca_featdf = pd.DataFrame(X_pca_train)
    # joblib.dump(norm_scalerpca, './gdrive/MyDrive/clean/pca_scaler.save')
    norm_scalerpca = joblib.load('./gdrive/MyDrive/clean/pca_scaler.save')
    n_pca_feat = norm_scalerpca.transform(pca_featdf)
    n_pca_feat = pd.DataFrame(n_pca_feat)
    new_col_headers = ["PCA1", "PCA2"]
    n_pca_feat.columns = new_col_headers

    # mdsae_encoder.save('./gdrive/MyDrive/clean/mdsae_encoder.h5')
    mdsae_encoder = tf.keras.models.load_model('./gdrive/MyDrive/clean/mdsae_encoder.h5', compile=False)
    mdsae_encoded = mdsae_encoder
    mdsae_gen_feat = mdsae_encoded.predict(n_X_train)
    mdsae_gen_featdf = pd.DataFrame(mdsae_gen_feat)
    # joblib.dump(norm_scaler_ae, './gdrive/MyDrive/clean/mdsae_scaler.save')
    norm_scaler_ae = joblib.load('./gdrive/MyDrive/clean/mdsae_scaler.save')
    n_mdsae_gen_feat = norm_scaler_ae.transform(gen_feat)
    n_mdsae_gen_feat = pd.DataFrame(n_mdsae_gen_feat)
    new_col_headers = ["MDSAE1", "MDSAE2", "MDSAE3", "MDSAE4"]
    n_mdsae_gen_feat.columns = new_col_headers

    n_ori_feat = pd.DataFrame(n_X_train, columns = ori_label)
    feat_label = pd.DataFrame(y_train, columns=["label"])
    whole_all_feat = pd.concat([n_ori_feat, n_gen_feat, n_pca_feat, n_mdsae_gen_feat, feat_label], axis=1)
    whole_all_feat['label'] = pd.to_numeric(whole_all_feat['label'])

    return whole_all_feat

st.set_page_config(
  page_title="Playground",
  page_icon="🌍"
  )

st.markdown("## IP Threat Detector")
st.sidebar.header("Playground")
st.write("""This demo shows ip threat detection using selected models""")

@st.cache_data
def get_model(model_file, json_file=None):
  if json_file != None:
    with open("./gdrive/MyDrive/DQL_model_tune/models/" + model_file + ".json", "r") as jfile:
      model = model_from_json(json.load(jfile))
    model.load_weights("./gdrive/MyDrive/DQL_model_tune/models/" + model_file + ".h5")
  else:
    with open('./gdrive/MyDrive/SVEM_model_tune/' + model_file + 'model.pkl', 'rb') as jfile:
      model = pickle.load(jfile)
  return model

def back_color_text(text, color):
   st.markdown(f'<p style="background-color:{color};text-indent: 50px;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

try:

  #st input for a csv file
  uploaded_file = st.file_uploader("Choose a CSV file")
  if uploaded_file is not None:
    features = pd.read_csv(uploaded_file)
    st.write("### Extracted Data from Network Packet", features)
    dest_ip_address = features['ip_dst_str'][0]
    src_ip_address = features['ip_src_str'][0]
    features = features.append(features)
    formated = formating(features)
    states = formated[top10]
    del(states['label'])
    st.write("### Formated features for ML detection", states.iloc[[0]])


  #st selectbox for a model
  st.write("### Choose a Machine Learning Model for Threat Detection")
  option = st.selectbox(
    "### Choose a Machine Learning Model for Threat Detection",
    ("DQL", "RF", "ET", "LSVC", "SGD", "LR"),
    index=None,
    placeholder="Select a ML model...",
    label_visibility="collapsed",
  )

  st.write("### Actions")
  st.write("Act now or it will be too late!")

  if option == "DQL":
    model = get_model("final_model", "final_model")
    q = model.predict(states.iloc[[0]])
    actions = np.argmax(q, axis=1, )
    if actions == 1:
      back_color_text(f""" <br>
      &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Threat Detected. Suspected IP Address: {dest_ip_address} <br>
      &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Please shut down {src_ip_address} immediately <br> .
      """,
      "#FF0000")
    else:
      back_color_text(f""" <br>
      &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp This network flow is safe from threat <br> .
      """, "#008000")


  if option in ["RF", "ET", "LSVC", "SGD", "LR"]:
    model = get_model(option)
    q = model.predict(states.iloc[[0]])
    actions = q
    if actions == 1:
      back_color_text(f""" <br>
      &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Threat Detected. Suspected IP Address: {dest_ip_address} <br>
      &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Please shut down {src_ip_address} immediately <br> .
      """,
      "#FF0000")
    else:
      back_color_text(f""" <br>
      &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp This network flow is safe from threat <br> .
      """, "#008000")

except URLError as e:
  st.error(
    """
    **This demo requires internet access.##
    Connection error: %s
    """
      % e.reason
  )