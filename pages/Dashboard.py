import streamlit as st

st.set_page_config(
  page_title="Network Dahsboard",
  page_icon="📊",
  layout="wide")

st.markdown("# Network Dashboard")
st.sidebar.header("Network Dashboard")
st.write("""
  Network Analysis Dahsboard
""")

st.components.v1.html("""
    <iframe title="Report Section" width="1024" height="612" src="https://app.powerbi.com/view?r=eyJrIjoiNjA3ZGMyMDgtMzg4My00ZDUxLWE2MDMtNmMwOWM3YjRhY2Q3IiwidCI6ImE2M2JiMWE5LTQ4YzItNDQ4Yi04NjkzLTMzMTdiMDBjYTdmYiIsImMiOjEwfQ%3D%3D" frameborder="0" allowFullScreen="true"></iframe>
  """,
  height=612
  )

