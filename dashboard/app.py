#!/usr/bin/env python3
'''
dashboard.py: streamlit dashboard for load test results.

Usage:
    streamlit run dashboard/app.py
'''

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title='llama.cpp load test results',
    page_icon=':llama:',
    layout='centered',
)

st.title('llama.cpp load test results')

# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

results_dir = Path(__file__).resolve().parent.parent / 'tests' / 'results'
csv_files = sorted(results_dir.glob('*.csv'), reverse=True)

if not csv_files:
    st.warning(f'No CSV files found in {results_dir}. Run the load test first.')
    st.stop()

selected_file = st.selectbox(
    label='Results file',
    options=csv_files,
    format_func=lambda p: p.name,
)

# ---------------------------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------------------------

df = pd.read_csv(selected_file, na_values='NaN')

required_cols = {'concurrency', 'latency_s', 'error'}
if not required_cols.issubset(df.columns):
    st.error(f'Unexpected CSV format. Required columns: {required_cols}')
    st.stop()

# Show only successful rows for latency plots
successes = df[df['error'].isna()].copy()

if successes.empty:
    st.error('No successful requests found in this file.')
    st.stop()

# ---------------------------------------------------------------------------
# Summary stats per concurrency level
# ---------------------------------------------------------------------------

stats = (
    successes
    .groupby('concurrency')['latency_s']
    .agg(
        mean='mean',
        count='count',
        std='std',
        p95=lambda x: x.quantile(0.95),
    )
    .reset_index()
)

# SEM = std / sqrt(n)
stats['sem'] = stats['std'] / np.sqrt(stats['count'])

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

run_ts = df['timestamp'].iloc[0] if 'timestamp' in df.columns else 'unknown'
total = len(df)
n_errors = df['error'].notna().sum()

col1, col2, col3 = st.columns(3)
def _small_metric(col, label: str, value) -> None:
    col.markdown(
        f'<p style="margin:0;font-size:0.8rem;color:grey">{label}</p>'
        f'<p style="margin:0;font-size:1rem;font-weight:600">{value}</p>',
        unsafe_allow_html=True,
    )

_small_metric(col1, 'Run timestamp', run_ts)
_small_metric(col2, 'Total requests', total)
_small_metric(col3, 'Errors', n_errors)

st.divider()

# ---------------------------------------------------------------------------
# Plot: mean ± SEM latency vs concurrency, markers connected by dashed line
# ---------------------------------------------------------------------------

st.subheader('Latency vs. concurrency')

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=stats['concurrency'],
    y=stats['mean'],
    error_y=dict(type='data', array=stats['sem'].tolist(), visible=True),
    mode='lines+markers',
    line=dict(dash='dash', width=1.5),
    marker=dict(size=8),
    name='Mean \u00b1 SEM',
))

fig.add_trace(go.Scatter(
    x=stats['concurrency'],
    y=stats['p95'],
    mode='lines+markers',
    line=dict(dash='dash', width=1.5),
    marker=dict(size=8, symbol='diamond'),
    name='p95',
))

fig.update_layout(
    xaxis_title='Concurrency (simultaneous requests)',
    yaxis_title='Latency (s)',
    xaxis=dict(tickmode='array', tickvals=stats['concurrency'].tolist()),
    showlegend=True,
    margin=dict(l=40, r=20, t=20, b=40),
)

st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# Raw stats table
# ---------------------------------------------------------------------------

st.subheader('Summary table')

display = stats.rename(columns={
    'concurrency': 'Concurrency',
    'mean': 'Mean (s)',
    'sem': 'SEM (s)',
    'std': 'Std dev (s)',
    'p95': 'p95 (s)',
    'count': 'Requests',
}).set_index('Concurrency')

st.dataframe(
    display.style.format('{:.3f}', subset=['Mean (s)', 'SEM (s)', 'Std dev (s)', 'p95 (s)']),
    width='stretch',
)

# ---------------------------------------------------------------------------
# Raw data expander
# ---------------------------------------------------------------------------

with st.expander('Raw data'):
    st.dataframe(df, width='stretch')
