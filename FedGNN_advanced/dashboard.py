# dashboard.py
import streamlit as st
import json, os, time
import pandas as pd

st.set_page_config(page_title="FedGNN Dashboard", layout="wide")
st.title("FedGNN - Training Monitor")

out_dir = st.text_input("Outputs directory", "./outputs")
auto = st.checkbox("Auto-refresh every 2s", value=True)
status = st.empty()
chart = st.empty()

def load_results(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path,'r') as f:
            return json.load(f)
    except Exception:
        return None

while True:
    data = load_results(os.path.join(out_dir, "results.json"))
    if data:
        rounds = data.get('rounds', [])
        rows = []
        for r in rounds:
            rows.append({
                'round': r.get('round'),
                'avg_loss': r.get('avg_loss'),
                'avg_acc': r.get('avg_acc'),
                'round_seconds': r.get('times',{}).get('round_seconds', None),
                'comm_bytes': sum(r.get('comm_bytes',[]))
            })
        df = pd.DataFrame(rows).set_index('round')
        if not df.empty:
            chart.line_chart(df[['avg_loss','avg_acc','round_seconds','comm_bytes']])
            status.text(f"Loaded {len(df)} rounds")
    if not auto:
        break
    time.sleep(2)
    st.experimental_rerun()
