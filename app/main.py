"""
Streamlit app for analyzing running FIT files with dynamic pace zones and interactive charts.

This module provides a Streamlit interface to upload a single FIT file, transform it
into structured DataFrames, and visualize running metrics (pace, heart rate, altitude)
over elapsed time using Plotly. Users can configure pace-zone thresholds (Z1–Z5)
directly from the sidebar, which dynamically renders input fields for all defined zones
based on ``ZONE_REF``. Each zone's thresholds are stored in ``st.session_state`` and
displayed as horizontal reference lines in the chart. Lap boundaries are also shown
as vertical markers. Tabular views of laps, splits, and raw records are rendered below
the chart.

Data transformation from FIT → DataFrames is handled by ``utils.loader.fit_to_df``.
It is expected to return a 5-tuple in the order:
``(df_session, _, df_lap, df_record, df_split)``.

The line chart uses ``df_record`` (with non-null ``speed_pace``), assumes
``elapsed_time`` on the x-axis, and reverses the y-axis to represent pace in min/km
(lower is faster).

Notes
-----
- The app enforces a wide page layout and reduced padding for a denser view.
- Zone configuration is dynamically generated using iteration over
  ``st.session_state["zones_current"]``.
- Each zone stores its current parameters (min/s) and color in ``st.session_state``.
- Pace-zone lines are computed as ``minute + second / 60`` (min/km).
- Hover tooltips are unified on the x-axis and show distance, heart rate, altitude, and lap.
- Lap markers are drawn from ``df_lap['cum_elapsed_time']`` (excluding the final lap).

Usage
-----
Run locally from the project root::

    streamlit run app/main.py

Workflow
---------
1. The sidebar dynamically renders input fields for each zone based on ``ZONE_REF``.
2. The user uploads a ``.fit`` file.
3. The file is parsed by ``fit_to_df`` into structured DataFrames.
4. A Plotly line chart is generated with:
   - Horizontal lines for configured pace zones.
   - Vertical lines for lap boundaries.
   - Unified hover tooltips for key metrics.
5. Tabular data (laps, splits, and raw records) is displayed below the chart.

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
utils.zone_input.zone_input_st : Renders sidebar inputs for each pace zone.
utils.zone_input.pace_minsec_to_float : Converts minute/second inputs into min/km values.

Examples
--------
After launching the app, use the sidebar to:

1. Set Z1–Z5 pace thresholds (minute and second fields).
2. Upload a ``.fit`` file.
3. View the interactive chart with zone and lap reference lines.
4. Explore lap, split, and raw record data tables below the visualization.
"""

import copy

import plotly.graph_objects as go
import streamlit as st
from utils.loader_class import DataLoaderFromFIT
from utils.unit_converter import pace_minsec_to_float
from utils.zone_input import ZONE_REF, zone_input_st

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3rem;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 1rem;
        }

        /* Sidebar opened */
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 325px !important;
            min-width: 325px !important;
            max-width: 400px !important;
        }

        /* Sidebar closed */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "zones_current" not in st.session_state:
    st.session_state["zones_current"] = copy.deepcopy(ZONE_REF)

with st.sidebar:
    st.write("**🦆DuckRunner🏃‍♂️**")

    st.write("**1\\.** Configure suas zonas de treino.")

    # Creating number_input for each zone
    for zone, data in st.session_state["zones_current"].items():
        for info, value in data.items():
            if info in ["start", "end"]:
                if value:
                    col1, col2 = st.columns(2)
                    with col1:
                        zone_input_st(zone, info, "min")
                    with col2:
                        zone_input_st(zone, info, "s")

    # Create file uploader winget
    st.write("**2\\.** Carrege o arquivo FIT.")
    fit_file = st.file_uploader(
        "**2\\.** Carrege o arquivo FIT.",
        accept_multiple_files=False,
        type="fit",
        label_visibility="collapsed",
    )

    # Choose x axis information
    st.write("**3\\.** Escolha a medida do eixo x.")
    xaxis_measure = st.selectbox(
        "Medida eixo x?",
        ("Distância", "Tempo"),
        label_visibility="collapsed",
    )


if fit_file is not None:
    with st.spinner("Preparando os dados...", show_time=True):
        # Loading data to dataframes
        data_loader = DataLoaderFromFIT(fit_file)
        data_loader = data_loader.start()

        df_session = data_loader.raw_session
        df_lap = data_loader.process_lap
        df_record = data_loader.process_record
        df_split = data_loader.process_lap

        if xaxis_measure == "Distância":
            xasis_series = df_record["distance_processed"]
            xasis_series_cum_name = "cum_distance_processed"
            xasis_unit = "m"
        else:
            xasis_series = df_record["elapsed_time"]
            xasis_series_cum_name = "cum_elapsed_time"
            xasis_unit = "s"

        # Creating the plot
        fig = go.Figure()

        # Pace line
        fig.add_trace(
            go.Scatter(
                x=xasis_series,
                y=df_record["pace_processed"],
                mode="lines",
                name="Ritmo",
                yaxis="y",
                line=dict(width=1.5),
                customdata=df_record[["pace_processed_time"]],
                hovertemplate=("<b>%{customdata[0]}</b>min/km"),
            )
        )

        # Enhanced altitude line
        # Scaling altitude
        max_enhanced_altitude = df_record["enhanced_altitude"].max()
        max_enhanced_altitude_scale = 20
        if max_enhanced_altitude < max_enhanced_altitude_scale:
            coef_enhanced_altitude = 1
        else:
            coef_enhanced_altitude = max_enhanced_altitude_scale / max_enhanced_altitude

        fig.add_trace(
            go.Scatter(
                x=xasis_series,
                y=(df_record["enhanced_altitude"] * coef_enhanced_altitude),
                mode="lines",
                name="Altitude",
                fill="tozeroy",
                yaxis="y2",
                line=dict(width=0.8),
                opacity=0.1,
                customdata=df_record[["enhanced_altitude"]],
                hovertemplate=("<b>%{customdata[0]:.1f}</b>m"),
            )
        )

        # FC line
        fig.add_trace(
            go.Scatter(
                x=xasis_series,
                y=df_record["heart_rate"],
                mode="lines",
                name="FC",
                yaxis="y2",
                hovertemplate=("<b>%{y}</b>bpm"),
            )
        )

        # Adding reference zones
        for _zone_name, zone_data in st.session_state["zones_current"].items():
            color = zone_data["color"]

            for bound in ("start", "end"):
                if zone_data.get(bound):
                    y_ref = pace_minsec_to_float(
                        zone_data[bound]["min"], zone_data[bound]["s"]
                    )
                    fig.add_hline(y=y_ref, line_color=color, layer="below")

        # Adding reference laps
        start_time_laps = df_lap.iloc[:-1, :][xasis_series_cum_name]
        for _, lap_time in start_time_laps.items():
            fig.add_vline(x=lap_time, line_color="silver", layer="below", line_width=1)

        # Updating hover and line width
        # fig.update_traces(
        #     line=dict(width=1.5),
        #     # hovertemplate=(
        #     #     "Tempo decorrido: <b>%{x}</b>s<br>"
        #     #     + "Distância: <b>%{customdata[0]}</b>m<br>"
        #     #     + "FC: <b>%{customdata[1]}</b>bpm<br>"
        #     #     + "Altitude: <b>%{customdata[2]}</b>m<br>"
        #     #     + "Volta: <b>%{customdata[3]}</b><extra></extra>"
        #     # ),
        # )

        # Updating layout
        fig.update_layout(
            height=650,
            showlegend=False,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=50, b=50),
            title="Análise Ritmo",
            xaxis=dict(
                title_text=f"{xaxis_measure} ({xasis_unit})",
                unifiedhovertitle=dict(
                    text=(xaxis_measure + ": <b>%{x}</b>" + xasis_unit)
                ),
            ),
            yaxis=dict(
                title_text="Ritmo (min/km)",
                autorange="reversed",
                autorangeoptions={"minallowed": 0, "clipmax": 12},
                ticksuffix=":00",
            ),
            yaxis2=dict(
                title="Freq. Card. (bpm) / Altitude (m)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

        st.plotly_chart(fig, config={"locale": "pt-BR"}, use_container_width=True)

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

        st.write("Dados Brutos")
        st.dataframe(df_record, hide_index=True)
