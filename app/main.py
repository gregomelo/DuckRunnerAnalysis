"""
Streamlit app to analyze running FIT files with pace zones and interactive charts.

This module provides a Streamlit interface to upload a single FIT file, transform it
into structured DataFrames, and visualize running metrics (pace, heart rate, altitude)
over elapsed time using Plotly. Users can configure pace-zone thresholds (Z1–Z5)
via the sidebar and see them as horizontal reference lines. Lap boundaries are also
displayed as vertical markers. Tabular views of laps, splits, and raw records are
rendered below the chart.

The transformation from FIT → DataFrames is handled by ``utils.loader.fit_to_df``.
It is expected to return a 5-tuple in the order:
``(df_session, _, df_lap, df_record, df_split)``. The line chart uses
``df_record`` (with non-null ``speed_pace``), assumes ``elapsed_time`` on the x-axis,
and reverses the y-axis to present pace in min/km (lower is faster).

Notes
-----
- The app enforces a wide page layout and reduces default padding for a denser view.
- Pace-zone inputs are captured as minutes and seconds; lines are added at
  ``minute + second / 60`` in min/km units.
- Hover is unified on the x-axis and shows distance, heart rate, altitude, and lap.
- Lap start markers are drawn from ``df_lap['cum_elapsed_time']`` (for laps > 1).

Usage
-----
Run locally from the project root::

    streamlit run app/main.py

Expected Columns
----------------
``df_record`` requires, at minimum:

- ``elapsed_time`` : int or float
    Seconds since the start of the activity.
- ``speed_pace`` : float
    Pace in minutes per kilometer (already converted upstream).
- ``distance`` : float
    Distance in meters.
- ``heart_rate`` : int or float
    Heart rate in bpm.
- ``enhanced_altitude`` : float
    Altitude in meters.
- ``lap`` : int
    Lap number associated with each record.

See Also
--------
utils.loader.fit_to_df : Parses the FIT binary and returns the DataFrames used here.

Examples
--------
After launching the app, use the sidebar to:

1. Set Z1–Z5 pace thresholds (minute and second fields).
2. Upload a ``.fit`` file.
3. Inspect the chart and the data tables (laps, splits, raw records).

"""

import plotly.express as px
import streamlit as st
from utils.loader import fit_to_df

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3.75rem;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PACE_Z1_ACTUAL_MINUTE = 6
PACE_Z1_ACTUAL_SECOND = 6

PACE_Z2_ACTUAL_MINUTE = 5
PACE_Z2_ACTUAL_SECOND = 20

PACE_Z3_1_ACTUAL_MINUTE = 4
PACE_Z3_1_ACTUAL_SECOND = 59

PACE_Z3_2_ACTUAL_MINUTE = 4
PACE_Z3_2_ACTUAL_SECOND = 38

PACE_Z4_ACTUAL_MINUTE = 4
PACE_Z4_ACTUAL_SECOND = 33

PACE_Z5_ACTUAL_MINUTE = 4
PACE_Z5_ACTUAL_SECOND = 2

st.sidebar.write("**1\\.** Configure suas zonas de treino.")

# Z1
col_z1_minute, col_z1_second = st.sidebar.columns(2)

with col_z1_minute:
    pace_z1_minute = st.number_input(
        "Término Z1 (min)", value=PACE_Z1_ACTUAL_MINUTE, min_value=0, max_value=59
    )

with col_z1_second:
    pace_z1_second = st.number_input(
        "Término Z1 (s)", value=PACE_Z1_ACTUAL_SECOND, min_value=0, max_value=59
    )

# Z2
col_z2_minute, col_z2_second = st.sidebar.columns(2)

with col_z2_minute:
    pace_z2_minute = st.number_input(
        "Término Z2 (min)", value=PACE_Z2_ACTUAL_MINUTE, min_value=0, max_value=59
    )

with col_z2_second:
    pace_z2_second = st.number_input(
        "Término Z2 (s)", value=PACE_Z2_ACTUAL_SECOND, min_value=0, max_value=59
    )

# Z3 Início
col_z3_1_minute, col_z3_1_second = st.sidebar.columns(2)

with col_z3_1_minute:
    pace_z3_1_minute = st.number_input(
        "Início Z3 (min)", value=PACE_Z3_1_ACTUAL_MINUTE, min_value=0, max_value=59
    )

with col_z3_1_second:
    pace_z3_1_second = st.number_input(
        "Início Z3 (s)", value=PACE_Z3_1_ACTUAL_SECOND, min_value=0, max_value=59
    )

# Z3 Fim
col_z3_2_minute, col_z3_2_second = st.sidebar.columns(2)

with col_z3_2_minute:
    pace_z3_2_minute = st.number_input(
        "Término Z3 (min)", value=PACE_Z3_1_ACTUAL_MINUTE, min_value=0, max_value=59
    )

with col_z3_2_second:
    pace_z3_2_second = st.number_input(
        "Término Z3 (s)", value=PACE_Z3_1_ACTUAL_SECOND, min_value=0, max_value=59
    )

# Z4
col_z4_minute, col_z4_second = st.sidebar.columns(2)

with col_z4_minute:
    pace_z4_minute = st.number_input(
        "Início Z4 (min)", value=PACE_Z4_ACTUAL_MINUTE, min_value=0, max_value=59
    )

with col_z4_second:
    pace_z4_second = st.number_input(
        "Início Z4 (s)", value=PACE_Z4_ACTUAL_SECOND, min_value=0, max_value=59
    )

# Z5
col_z5_minute, col_z5_second = st.sidebar.columns(2)

with col_z5_minute:
    pace_z5_minute = st.number_input(
        "Início Z5 (min)", value=PACE_Z5_ACTUAL_MINUTE, min_value=0, max_value=59
    )

with col_z5_second:
    pace_z5_second = st.number_input(
        "Início Z5 (s)", value=PACE_Z5_ACTUAL_SECOND, min_value=0, max_value=59
    )


# Create file uploader winget
st.sidebar.write("**2\\.** Carrege o arquivo FIT.")
fit_file = st.sidebar.file_uploader(
    "**2\\.** Carrege o arquivo FIT.",
    accept_multiple_files=False,
    type="fit",
    label_visibility="collapsed",
)

if fit_file is not None:
    df_session, _, df_lap, df_record, df_split = fit_to_df(fit_file)

    df_plot = df_record.dropna(subset=["speed_pace"]).ffill()

    fig = px.line(
        df_plot,
        x="elapsed_time",
        y="speed_pace",
        hover_data={
            "distance": True,
            "heart_rate": True,
            "enhanced_altitude": True,
            "lap": True,
            "elapsed_time": True,
            "speed_pace": True,
        },
    )

    fig.add_hline(
        y=pace_z1_minute + pace_z1_second / 60, line_color="blue", layer="below"
    )

    fig.add_hline(
        y=pace_z2_minute + pace_z2_second / 60, line_color="green", layer="below"
    )

    fig.add_hline(
        y=pace_z3_1_minute + pace_z3_1_second / 60, line_color="yellow", layer="below"
    )

    fig.add_hline(
        y=pace_z3_2_minute + pace_z3_2_second / 60, line_color="yellow", layer="below"
    )

    fig.add_hline(
        y=pace_z4_minute + pace_z4_second / 60, line_color="red", layer="below"
    )

    fig.add_hline(
        y=pace_z5_minute + pace_z5_second / 60, line_color="purple", layer="below"
    )

    start_time_laps = df_lap.iloc[:-1, :][df_lap["lap"] > 1]["cum_elapsed_time"]
    for _, lap_time in start_time_laps.items():
        fig.add_vline(x=lap_time, line_color="silver", layer="below", line_width=1)

    fig.update_traces(
        hovertemplate=(
            "Tempo decorrido: <b>%{x}</b>s<br>"
            + "Distância: <b>%{customdata[0]}</b>m<br>"
            + "FC: <b>%{customdata[1]}</b>bpm<br>"
            + "Altitude: <b>%{customdata[2]}</b>m<br>"
            + "Volta: <b>%{customdata[3]}</b><extra></extra>"
        )
    )

    fig.update_layout(
        height=600,
        hovermode="x unified",
        xaxis=dict(
            title_text="Tempo decorrido (s)", unifiedhovertitle=dict(text="Dados")
        ),
        yaxis=dict(
            title_text="Ritmo (min/km)",
            autorange="reversed",
            autorangeoptions={"clipmax": 10},
        ),
    )

    st.write("Análise Ritmo")
    st.plotly_chart(fig, config={"locale": "pt-BR"})

    st.write("Dados Voltas")
    st.dataframe(
        df_lap,
        hide_index=True,
        # Futuro
        # column_config={
        #     "lap": "Volta",
        #     "start_time": st.column_config.DatetimeColumn(
        #         "Início",
        #         # format="%d ⭐",
        #     ),
        #     "end_time": st.column_config.DatetimeColumn(
        #         "Término",
        #         # format="%d ⭐",
        #     ),
        #     "total_elapsed_time": st.column_config.NumberColumn(
        #         "Tempo (s)",
        #         format="%d",
        #     ),
        #     "total_distance": st.column_config.NumberColumn(
        #         "Distância (m)",
        #         format="%f",
        #     ),
        # },
    )

    st.write("Dados Parciais")
    st.dataframe(df_split, hide_index=True)

    st.write("Dados Brutos")
    st.dataframe(df_record, hide_index=True)
