import logging
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from datetime import timedelta
from IPython.display import display, HTML
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path

from lsst.summit.utils.efdUtils import (
    getEfdData, 
    getDayObsEndTime, 
    getDayObsStartTime
)
from lsst.ts.xml.enums.MTHexapod import ApplicationStatus, EnabledSubstate
from astropy.time import Time
from io import BytesIO
import base64
import requests


# Set global font size for labels, titles, and ticks
plt.rcParams.update({
    "axes.grid": True,
    "axes.labelsize": 12,  
    "axes.titlesize": 14,
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": False,
    "axes.formatter.limits": (-100, 100),
    "figure.figsize": (11, 6),
    "font.size": 12,
    "grid.color": "#b0b0b0", 
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.75,
    "xtick.labelsize": 12,  
    "ytick.labelsize": 12,  
})

level_colors = {
    "D": "#1f77b4",  # blue
    "I": "#2ca02c",  # green
    "W": "#ff7f0e",  # orange
    "E": "#d62728",  # red
    "C": "#9467bd",  # purple
}


def collapsible_table(client, df, states, detail_col, sal_index, index_format="%Y-%m-%d %H:%M:%S.%f", expanded=False):
    """
    Display each row of a DataFrame as a collapsible block using the index and label columns.

    Parameters
    ----------
    client : EfdClient
        EFD client to query data.
    df : pd.DataFrame
        Data to display.
    states : pd.DataFrame
        Data containing the states to display.
    detail_col : str
        Column with the detailed content (e.g. tracebacks).
    index_format : str
        strftime format for displaying the index.
    expanded : bool
        Whether to display all rows expanded by default.
    """
    html = """
    <style>
    details:hover {
        background-color: #f0f0f0;
        border-radius: 4px;
    }
    </style>
    """

    for idx, row in df.iterrows():

        # Format index
        try:
            idx_str = idx.strftime(index_format)[:-3]  # trim microseconds to milliseconds
        except Exception:
            idx_str = str(idx)

        summary = f"[{idx_str}]"

        if "level" in row.index:
            level_letter = logging.getLevelName(row["level"])[0]
            color = level_colors.get(level_letter, "black")
            summary += f"<span style='color:{color};'>[{level_letter}]</span>"

        if "functionName" in row.index:
            summary += f" {row['functionName']} "

        # Format detail content
        content = str(row[detail_col]).replace("\n", "<br>").replace(" ", "&nbsp;")
        
        # Add state information if available
        if states is not None:
            t_min = idx - pd.Timedelta(seconds=1)
            t_max = idx
            state_window = states[(states.index >= t_min) & (states.index <= t_max)]
            state_window = state_window.drop(columns=["salIndex", "applicationStatus", "enabledSubstate"], errors="ignore")
            state_window = state_window.rename(columns={"applicationStatusName": "Application Status", "enabledSubstateName": "Enabled Substate"})  
            state_window["Application Status"] = state_window["Application Status"].str.replace("|", " ", regex=False)
            if not state_window.empty:
                state_window.index = state_window.index.strftime(index_format)  # Format the index
                state_summary = f"""
                <div style="margin: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: white;">
                    <strong>State Information:</strong><br>
                    {state_window.to_html(index=True, escape=False)}
                </div>
                """

        open_attr = " open" if expanded else ""
        
        df = query_hexapod_actuator_telemetry(client, idx, sal_index)
        if not df.empty:
            # Generate the plot and save it to a BytesIO buffer
            buffer = BytesIO()
            plot_hexapod_actuator_errors(df, idx)
            plt.savefig(buffer, format="png")
            plt.close()
            buffer.seek(0)

            # Encode the image as a base64 string
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()

            # Embed the image as an HTML <img> tag
            image = f"""
            <div style="margin: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: white;">
                <img src="data:image/png;base64,{image_base64}" alt="Hexapod Actuator Errors" style="max-width: 100%; height: auto;">
                <p style="text-align: center; font-size: 12px;">Hexapod Actuator Errors near {idx}</p>
            </div>
            """
        else:
            image = ""

        chronograf_link = display_chronograf_link(sal_index, idx - pd.Timedelta(seconds=1), idx)

        html += f"""
        <details style="margin-bottom: 5px; padding: 10px"{open_attr}>
          <summary style="font-family: monospace; padding: 5px">{summary}</summary>
          <pre style="margin-left: 1em; font-family: monospace;">{content}<br><br></pre>
          {state_summary if 'state_summary' in locals() else ''}
          {chronograf_link}
          {image}
        </details>
        """

    display(HTML(html))


def extract_log_window(df, log_file_path, window_seconds=1):
    """
    For each timestamp in df.index, extract lines from the log file within ±window_seconds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index (UTC).
    log_file_path : str
        Path to the log file.
    window_seconds : int
        Time window in seconds.

    Returns
    -------
    dict : index timestamp -> list of matching log lines
    """
    results = {}
    html = ""

    # Read all log lines first
    with open(log_file_path, "r") as f:
        log_lines = f.readlines()

    # Pre-parse timestamps from log lines
    parsed_log = []
    for line in log_lines:
        try:
            ts_str = line.split()[0]  # first part is timestamp
            ts = pd.to_datetime(ts_str)
            parsed_log.append((ts, line))
        except Exception:
            continue  # skip malformed lines

    # For each index in df, find matching log lines
    for idx, row in df.iterrows():

        if "level" in row.index:
            level_letter = logging.getLevelName(row["level"])[0]
            color = level_colors.get(level_letter, "black")
            letter = f"<span style='color:{color};'>[{level_letter}]</span>"

        t_min = idx - pd.Timedelta(seconds=window_seconds)
        t_max = idx + pd.Timedelta(seconds=window_seconds)
        matches = [line for ts, line in parsed_log if t_min <= ts <= t_max]

        html += f"""
        <details>
            <summary><strong>{idx} {letter} </strong></summary>
            <pre>{"".join(matches)}</pre>
        </details>
        """

    display(HTML(html))


def query_hexapod_faults(client, day_obs, sal_index):
    """Query all the hexapod faults for a given `day_obs`"""
    start_time = getDayObsStartTime(day_obs)
    end_time = getDayObsEndTime(day_obs)

    _df = getEfdData(
        client=client,
        topic="lsst.sal.MTHexapod.logevent_errorCode",
        columns=["errorCode", "errorReport", "salIndex", "traceback"],
        begin=start_time,
        end= end_time
    )
    
    return _df[_df.salIndex == sal_index]


def query_hexapod_log_messages(client, day_obs, sal_index):
    """Query error messages from the log messages given a `day_obs`"""
    start_time = getDayObsStartTime(day_obs)
    end_time = getDayObsEndTime(day_obs)

    _df = getEfdData(
        client=client,
        topic="lsst.sal.MTHexapod.logevent_logMessage",
        columns=["functionName", "level", "lineNumber", "message", "salIndex", ],
        begin=start_time,
        end=end_time
    )

    _df = _df[_df.salIndex == sal_index]
    _df = _df[_df.level >= logging.CRITICAL]

    return _df


def query_hexapod_controller_state(client, day_obs, sal_index):
    """Query the hexapod controller state for a given `day_obs`"""
    start_time = getDayObsStartTime(day_obs)
    end_time = getDayObsEndTime(day_obs)

    _df = getEfdData(
        client=client,
        topic="lsst.sal.MTHexapod.logevent_controllerState",
        columns=["applicationStatus", "enabledSubstate", "salIndex"],
        begin=start_time,
        end=end_time
    )

    _df = _df[_df.salIndex == sal_index]
    _df["applicationStatusName"] = _df["applicationStatus"].apply(lambda x: ApplicationStatus(x).name)
    _df["enabledSubstateName"] = _df["enabledSubstate"].apply(lambda x: EnabledSubstate(x).name)

    return _df


def query_hexapod_actuator_telemetry(client, t_stamp, sal_index, delta_t=1):
    """Query the hexapod actuator telemetry for a given `t_stamp`"""
    t_stamp = Time(t_stamp)
    start_time = t_stamp - timedelta(seconds=delta_t)
    end_time = t_stamp

    _df = getEfdData(
        client=client,
        topic="lsst.sal.MTHexapod.actuators",
        columns="*",
        begin=start_time,
        end=end_time
    )

    _df = _df[_df.salIndex == sal_index]
    _df = _df.drop(columns=["salIndex", "timestamp"], errors="ignore")

    return _df


def plot_hexapod_actuator_errors(df, t_fault):
    """Plot the actuator errors for a given timestamp."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(6):
        ax.plot(df.index, df[f"positionError{i}"], label=f"Actuator {i}")  

    ax.set_title(f"Hexapod Actuator Errors near {t_fault}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Error Value")
    ax.legend(ncols=2, loc="best", fontsize=10)
    ax.grid(":", alpha=0.5)

    # Set x-axis tick format to HH:MM:SS.sss (milliseconds with three decimal digits)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S.%f"))
    # Adjust tick labels to avoid overlap
    fig.autofmt_xdate()
    

def create_url_for_chronograf(sal_index, t_start, t_end):
    
    if sal_index == 1:
        sal_index = "Camera"
    elif sal_index == 2:
        sal_index = "M2"
    else:
        raise ValueError(f"Invalid sal_index: {sal_index}. Must be 1 (Camera) or 2 (M2).")
    
    base_url = "https://usdf-rsp.slac.stanford.edu/chronograf/sources/1/dashboards/113"
    params = {
        "refresh": "Paused",
        "tempVars[SalIndex]": sal_index,
        "tempVars[sampling]": "High Performance",
        "lower": t_start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "upper": t_end.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    }
    
    return f"{base_url}?{requests.compat.urlencode(params)}"


def display_chronograf_link(sal_index, t_start, t_end):
    """
    Display a link to the Chronograf dashboard for a given sal_index and time range.
    """
    url = create_url_for_chronograf(sal_index, t_start, t_end)
    html = f"""
    <div style="margin: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
        <a href="{url}" target="_blank" style="text-decoration: none; color: #007bff;">
            Open Chronograf Dashboard
        </a>
    </div>
    """
    return html
