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
    layout='wide',
)

st.title('llama.cpp load test results')

results_dir = Path(__file__).resolve().parent.parent / 'tests' / 'results'


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _small_metric(col, label: str, value) -> None:
    '''Render a compact label/value metric using small HTML text.'''

    col.markdown(
        f'<p style="margin:0;font-size:0.8rem;color:grey">{label}</p>'
        f'<p style="margin:0;font-size:1rem;font-weight:600">{value}</p>',
        unsafe_allow_html=True,
    )


def _summarize(successes: pd.DataFrame, group_col: str) -> pd.DataFrame:
    '''Compute mean, SEM, p95 grouped by group_col.'''

    stats = (
        successes
        .groupby(group_col)['latency_s']
        .agg(
            mean='mean',
            count='count',
            std='std',
            p95=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])

    return stats


def _latency_figure(stats: pd.DataFrame, x_col: str, x_label: str) -> go.Figure:
    '''Build a plotly latency figure with mean ± SEM and p95 traces.'''

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stats[x_col],
        y=stats['mean'],
        error_y=dict(type='data', array=stats['sem'].tolist(), visible=True),
        mode='lines+markers',
        line=dict(dash='dash', width=1.5),
        marker=dict(size=8),
        name='Mean \u00b1 SEM',
    ))

    fig.add_trace(go.Scatter(
        x=stats[x_col],
        y=stats['p95'],
        mode='lines+markers',
        line=dict(dash='dash', width=1.5),
        marker=dict(size=8, symbol='diamond'),
        name='p95',
    ))

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title='Latency (s)',
        xaxis=dict(tickmode='array', tickvals=stats[x_col].tolist()),
        showlegend=True,
        margin=dict(l=40, r=20, t=20, b=40),
    )

    return fig


def _stats_table(stats: pd.DataFrame, index_col: str, index_label: str) -> pd.DataFrame:
    '''Rename and format stats dataframe for display.'''

    return (
        stats
        .rename(columns={
            index_col: index_label,
            'mean': 'Mean (s)',
            'sem': 'SEM (s)',
            'std': 'Std dev (s)',
            'p95': 'p95 (s)',
            'count': 'Requests',
        })
        .set_index(index_label)
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_load, tab_ctx = st.tabs(['Load test', 'Context length'])


# ---------------------------------------------------------------------------
# Tab: Load test
# ---------------------------------------------------------------------------

with tab_load:

    load_files = sorted(
        [f for f in results_dir.glob('*.csv') if not f.name.startswith('context_test_')],
        reverse=True,
    )

    if not load_files:
        st.warning(f'No load test CSV files found in {results_dir}. Run the load test first.')

    else:

        selected_file = st.selectbox(
            label='Results file',
            options=load_files,
            format_func=lambda p: p.name,
            key='load_file',
        )

        df = pd.read_csv(selected_file, na_values='NaN')

        required_cols = {'concurrency', 'latency_s', 'error'}

        if not required_cols.issubset(df.columns):
            st.error(f'Unexpected CSV format. Required columns: {required_cols}')

        else:

            successes = df[df['error'].isna()].copy()

            if successes.empty:
                st.error('No successful requests found in this file.')

            else:

                stats = _summarize(successes, 'concurrency')

                # Metadata
                run_ts = df['timestamp'].iloc[0] if 'timestamp' in df.columns else 'unknown'
                col1, col2, col3 = st.columns(3)
                _small_metric(col1, 'Run timestamp', run_ts)
                _small_metric(col2, 'Total requests', len(df))
                _small_metric(col3, 'Errors', df['error'].notna().sum())

                st.divider()

                # Plot and summary table side by side
                st.subheader('Latency vs. concurrency')

                col_plot, col_table = st.columns([3, 2])

                with col_plot:
                    fig = _latency_figure(
                        stats,
                        x_col='concurrency',
                        x_label='Concurrency (simultaneous requests)',
                    )
                    st.plotly_chart(fig, width='stretch')

                with col_table:
                    display = _stats_table(stats, 'concurrency', 'Concurrency')
                    st.dataframe(
                        display.style.format(
                            '{:.3f}',
                            subset=['Mean (s)', 'SEM (s)', 'Std dev (s)', 'p95 (s)'],
                        ),
                        width='stretch',
                    )

                with st.expander('Raw data'):
                    st.dataframe(df, width='stretch')


# ---------------------------------------------------------------------------
# Tab: Context length
# ---------------------------------------------------------------------------

with tab_ctx:

    ctx_files = sorted(results_dir.glob('context_test_*.csv'), reverse=True)

    if not ctx_files:
        st.info('No context length results yet. Run tests/context_length_test.py first.')

    else:

        selected_ctx = st.selectbox(
            label='Results file',
            options=ctx_files,
            format_func=lambda p: p.name,
            key='ctx_file',
        )

        ctx_df = pd.read_csv(selected_ctx, na_values='NaN')

        required_ctx_cols = {'prompt_tokens', 'latency_s', 'error'}

        if not required_ctx_cols.issubset(ctx_df.columns):
            st.error(f'Unexpected CSV format. Required columns: {required_ctx_cols}')

        else:

            ctx_successes = ctx_df[ctx_df['error'].isna()].copy()

            if ctx_successes.empty:
                st.error('No successful requests found in this file.')

            else:

                ctx_stats = _summarize(ctx_successes, 'prompt_tokens')

                # Metadata
                ctx_run_ts = ctx_df['timestamp'].iloc[0] if 'timestamp' in ctx_df.columns else 'unknown'
                ctx_concurrency = ctx_df['concurrency'].iloc[0] if 'concurrency' in ctx_df.columns else '?'
                col1, col2, col3, col4 = st.columns(4)
                _small_metric(col1, 'Run timestamp', ctx_run_ts)
                _small_metric(col2, 'Total requests', len(ctx_df))
                _small_metric(col3, 'Errors', ctx_df['error'].notna().sum())
                _small_metric(col4, 'Concurrency', ctx_concurrency)

                st.divider()

                # Plot and summary table side by side
                st.subheader('Latency vs. input context length')

                col_plot, col_table = st.columns([3, 2])

                with col_plot:
                    ctx_fig = _latency_figure(
                        ctx_stats,
                        x_col='prompt_tokens',
                        x_label='Input tokens',
                    )
                    st.plotly_chart(ctx_fig, width='stretch')

                with col_table:
                    ctx_display = _stats_table(ctx_stats, 'prompt_tokens', 'Input tokens')
                    st.dataframe(
                        ctx_display.style.format(
                            '{:.3f}',
                            subset=['Mean (s)', 'SEM (s)', 'Std dev (s)', 'p95 (s)'],
                        ),
                        width='stretch',
                    )

                with st.expander('Raw data'):
                    st.dataframe(ctx_df, width='stretch')
