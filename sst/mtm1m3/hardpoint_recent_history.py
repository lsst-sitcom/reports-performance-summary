import asyncio
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.time import Time
from typing import List, Optional

from lsst_efd_client import EfdClient
from lsst.ts.xml.enums.MTM1M3 import HardpointTest


__all__ = ["build_labels", "ensure_utc", "find_end_of_test", "group_by_gaps"]


# HP BreakAway Limits from https://sitcomtn-082.lsst.io/
COMPRESSION_LOWER_LIMIT = 2981  # N
COMPRESSION_UPPER_LIMIT = 3959  # N
TENSION_LOWER_LIMIT = -4420  # N
TENSION_UPPER_LIMIT = -3456  # N

# Used to select the window size near zero measured forces.
DISPLACEMENT_CROP_RANGE = 300  # um
DISPLACEMENT_CROP_RANGE_FOR_FIT = 100  # um

# Used to determine if a test was valid or not
MEASURED_FORCE_MAXIMUM_TOLERANCE = 1000  # N

# Meter to micrometer
METER_TO_MICROMETER = 1e6  # um/m

# Maximum spec stiffness
SPEC_STIFFNESS = 100  # N/um

# Status Plotting
STATUS_COLORS = {
    HardpointTest.NOTTESTED: "#9e9e9e",
    HardpointTest.MOVINGNEGATIVE: "#42a5f5",
    HardpointTest.TESTINGPOSITIVE: "#66bb6a",
    HardpointTest.TESTINGNEGATIVE: "#ffa726",
    HardpointTest.MOVINGREFERENCE: "#26c6da",
    HardpointTest.PASSED: "#2e7d32",
    HardpointTest.FAILED: "#ef5350",
}


def add_custom_time_format(ax: plt.Axes) -> None:
    """Convenient function to deal with time xaxis"""
    locator = mdates.MinuteLocator(interval=1)
    ax.xaxis.set_major_locator(locator)

    formatter = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(formatter)


def add_plot_cosmetics(
    ax: plt.Axes, x_label: str, y_label: str, title: str, right_axis: bool = False
) -> None:
    """Convenient function to deal with cosmetics applied to every plot"""
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(TENSION_LOWER_LIMIT - 500, COMPRESSION_UPPER_LIMIT + 500)

    ax.set_title(title)
    ax.grid(":", alpha=0.2)

    if right_axis:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()


def add_ranges_where_breakaway_should_happen(ax: plt.Axes) -> None:
    """
    Add two shaded areas representing where the breakaway mechanism
    must be triggered.
    """
    ax.axhspan(COMPRESSION_LOWER_LIMIT, COMPRESSION_UPPER_LIMIT, fc="cornflowerblue", alpha=0.2)
    ax.axhspan(TENSION_LOWER_LIMIT, TENSION_UPPER_LIMIT, fc="firebrick", alpha=0.2)


def add_stiffness(ax: plt.Axes, df: pd.DataFrame) -> list[plt.Line2D]:
    """Add a dashed lines representing the calculated and the spec stiffness"""

    l1 = ax.plot(
        df.displacement,
        df.displacement * SPEC_STIFFNESS,
        ":",
        color="k",
        label=f"Spec Stiff. [{SPEC_STIFFNESS} N/um]",
    )

    p = get_stiffness_polynomial(df)
    if p is None:
        return [l1]

    idx = df.hardpoint.iloc[0] - 1
    l2 = ax.plot(
        df.displacement,
        np.polyval(p, df.displacement),
        c=f"C{idx}",
        ls="--",
        label=f"Meas. Stiff. [{np.round(p[0])} N/um]",
    )

    return [l1, l2]


def add_time_vertical_lines(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Add vertical lines for t_begin, t_positive, t_negative, and t_end"""
    for col_name in ["t_begin", "t_positive", "t_negative", "t_end"]:
        timestamp = get_timestamp_from_dataframe(df, col_name)
        ax.axvline(timestamp, ls="--", lw=0.5, c="black")


def build_status_timeline(
    start_time: pd.Timestamp, status: pd.DataFrame, end_reasons: List[int] | None = None
) -> pd.DataFrame:
    """
    Create a new dataframe containing the time windows associated with status
    events.
    """
    if end_reasons is None:
        end_reasons = [
            HardpointTest.NOTTESTED,
            HardpointTest.PASSED,
            HardpointTest.FAILED,
        ]

    slice = status.loc[start_time:]
    end_time = find_end_of_test(slice, end_reasons)
    end_time = (
        pd.Timestamp(end_time) + pd.Timedelta("1s") if end_time is not None else None
    )
    slice = slice.loc[:end_time] if end_time is not None else slice
    print(f"Start time: {start_time}\nEnd Time: {end_time}")

    timeline = pd.DataFrame(
        index=[hp.name for hp in HardpointTest],
        columns=[col for col in status.columns if col != "index"],
    )

    for col in status.columns:
        for hp in HardpointTest:
            mask = slice[col].isin([hp])
            tstamp = slice[col][mask].first_valid_index()
            timeline.loc[hp.name, col] = tstamp if tstamp is not None else pd.NaT

    return timeline


def build_labels(cmds: pd.DataFrame, sts: pd.DataFrame) -> pd.DataFrame:
    """
    Once our dataframe has a new column called "group_id", we can use it to
    group our tests by this identifier.
    """
    summary = (
        cmds.reset_index()
        .groupby("group_id")
        .agg(
            reference_time=("index", lambda s: s.dt.floor("min").min()),
            n=("group_id", "size"),
            uniq_hp=("hardpointActuator", lambda s: sorted(s.unique())),
        )
        .reset_index()
    )

    summary["start_time"] = summary["reference_time"].apply(
        lambda x: sts.index[sts.index.searchsorted(x, side="left")]
    )

    summary["end_time"] = summary["reference_time"].apply(
        lambda x: find_end_of_test(sts, x)
    )

    return summary


def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Problems with timezones are common when querying data from the EFD.
    This function ensures that we localize the timestamps.
    """
    if df.index.tz is None:
        return df.tz_localize("UTC")
    else:
        return df.tz_convert("UTC")


def find_end_of_test(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_reasons: List[int] | None = None,
    end_buffer: str = "10s",
) -> Optional[pd.Timestamp]:
    """
    Find the end of a test run by looking for the first row where all columns
    have the same value and that value is in end_reasons.
    """
    if end_reasons is None:
        end_reasons = [
            HardpointTest.NOTTESTED,
            HardpointTest.PASSED,
            HardpointTest.FAILED,
        ]

    # Ensure we don't get a fake NOTTESTED in the beginning of the test
    df = df.loc[df.index >= start_time + pd.Timedelta("5s")]

    for idx, row in df.iterrows():
        unique_values = set(row.values)
        if len(unique_values) == 1 and list(unique_values)[0] in end_reasons:
            return idx + pd.Timedelta(end_buffer)

    return None


def group_by_gaps(gap: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    It is uncommon to have more than one execution of hardpoint breakaway
    tests in a single day. But hey happen. This function defines group labels
    that we can use to distinguish between each run of this test.
    """
    time_gap = pd.Timedelta(gap)
    group_id = df.index.to_series().diff().gt(time_gap).cumsum()
    return df.assign(group_id=group_id)


def _status_segments(status_s: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp):
    """Return list of (start, end, state) where state is constant in [t0,t1]."""
    s = status_s.sort_index()
    s = s[(s.index <= t1) & (s.index >= (t0 - pd.Timedelta("1s")))]
    if s.empty:
        return []
    if s.index[0] > t0:
        s = pd.concat([pd.Series([s.iloc[0]], index=[t0]), s])
    if s.index[-1] < t1:
        s.loc[t1] = s.iloc[-1]
    s = s.sort_index()
    changed = s.ne(s.shift()).to_numpy()
    segs = []
    idxs = [i for i, c in enumerate(changed) if c]
    idxs.append(len(s) - 1)
    for i in range(len(idxs) - 1):
        a = s.index[idxs[i]]
        b = s.index[idxs[i + 1]]
        state = int(s.iloc[idxs[i]])
        beg = max(pd.Timestamp(a), t0)
        end = min(pd.Timestamp(b), t1)
        if end > beg:
            segs.append((beg, end, state))
    return segs


def query_hardpoints_telemetry(
    client: EfdClient, summary_row: pd.Series, sampling: str = "500ms"
) -> pd.DataFrame:
    start_time = Time(summary_row["start_time"]).isot
    end_time = Time(summary_row["end_time"]).isot

    columns_displacement = [
        f"mean(displacement{i - 1}) as mean_displacement_{i - 1}"
        for i in summary_row["uniq_hp"]
    ]
    columns_forces = [
        f"mean(measuredForce{i - 1}) as mean_measured_force_{i - 1}"
        for i in summary_row["uniq_hp"]
    ]
    columns = ", ".join(columns_displacement + columns_forces)

    query = f"""
        SELECT {columns}
        FROM "lsst.sal.MTM1M3.hardpointActuatorData"
        WHERE time >= '{start_time}Z'
        AND time <= '{end_time}Z'
        GROUP by time({sampling})
    """

    # Perform the query
    telemetry = asyncio.run(client.influx_client.query(query))

    # Convert displacement from m to um
    for c in columns_displacement:
        c = c.split(" ")[-1]
        telemetry[c] = telemetry[c].astype(float) * METER_TO_MICROMETER

    return telemetry


def plot_forces_timeline(ax: plt.Axes, tel: pd.DataFrame, hpi: int):
    """
    Plot the telemetry for a single hardpoint actuator.
    """
    ax.clear()
    
    col_force = f"mean_measured_force_{hpi}"
    if col_force in tel.columns:
        ax.plot(tel.index, tel[col_force], label=col_force, color="black")

    add_plot_cosmetics(
        ax, "Time [UTC]", "Measured Force [N]", "Measured Force Timeline"
    )
    add_ranges_where_breakaway_should_happen(ax)
    # add_time_vertical_lines(ax, df)
    add_custom_time_format(ax)
    ax.legend(fontsize=9)


def plot_status_timeline(
    ax, sub_status_df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp
):
    """Plot colored boxes per strut vs time for states 1..7 in [t0,t1]."""
    ax.clear()
    ax.set_xlim(t0, t1)
    ax.set_ylim(0.5, 6.5)
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels([f"HP{i}" for i in range(1, 7)])
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Strut")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for i in range(6):
        col = f"testState{i}"
        if col not in sub_status_df.columns:
            continue
        segs = _status_segments(sub_status_df[col], t0, t1)
        y = i + 1
        for a, b, state in segs:
            enum_state = HardpointTest(state)
            ax.broken_barh(
                [(mdates.date2num(a), mdates.date2num(b) - mdates.date2num(a))],
                (y - 0.4, 0.8),
                facecolors=STATUS_COLORS.get(enum_state, "#bdbdbd"),
                edgecolors="none",
            )

    handles = [
        plt.Line2D(
            [0], [0], lw=8, color=STATUS_COLORS[st], label=f"{st.value}-{st.name}"
        )
        for st in HardpointTest
    ]
    ax.legend(
        handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0
    )
    ax.grid(True, linestyle=":", alpha=0.3)


def plot_stiffness(ax: plt.Axes, tel: pd.DataFrame, hpi: int):
    print("Plot stiffness")


def plot_stiffness_positive(ax: plt.Axes, df: pd.DataFrame, idx: int) -> None:
    """
    Plot the measured forces versus the displacement for when moving the
    hardpoint in the positive (compression) direction. The slant of the curve
    near zero corresponds to the stiffness of the hardpoint in that direction.
    """
    # Select relevant area
    df = select_region_where_forces_are_changing(df, "t_positive", "t_negative")

    # Plot the displacement
    ax.plot(df.displacement, df.measured_force, color=f"C{idx}", alpha=0.5)

    # Show the ranges where the breakaway is expected to happen
    add_ranges_where_breakaway_should_happen(ax)
    add_plot_cosmetics(ax, "Displacement [um]", "Measured Force [N]", "Moving Positive")
    add_stiffness(ax, df)
    ax.legend(fontsize=9)


def plot_stiffness_negative(ax: plt.Axes, df: pd.DataFrame, idx: int) -> None:
    """
    Plot the measured forces versus the displacement for when moving the
    hardpoint in the positive (compression) direction. The slant of the curve
    near zero corresponds to the stiffness of the hardpoint in that direction.
    """
    # Select relevant area
    df = select_region_where_forces_are_changing(df, "t_negative", "t_end")

    # Plot the displacement
    ax.plot(df.displacement[::-1], df.measured_force[::-1], color=f"C{idx}", alpha=0.5)

    # Show the ranges where the breakaway is expected to happen
    add_plot_cosmetics(
        ax,
        "Displacement [um]",
        "Measured Force [N]",
        "Moving Negative",
        right_axis=True,
    )
    add_ranges_where_breakaway_should_happen(ax)
    add_stiffness(ax, df)
    ax.legend(fontsize=9)


def get_stiffness_polynomial(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Fit a polynomial to the data near the minimum measured forces"""
    if df.measured_force.abs().min() > MEASURED_FORCE_MAXIMUM_TOLERANCE:
        reason = df.end_reason.iloc[0]
        hp = df.hardpoint.iloc[0]
        print(f"Warning - Not fitting stiffness for HP{hp}")
        return None

    df = df[df.displacement.abs() <= DISPLACEMENT_CROP_RANGE_FOR_FIT]
    poly = np.polyfit(df.displacement, df.measured_force, deg=1)

    return poly


def get_timestamp_from_dataframe(df: pd.DataFrame, column_name: str) -> pd.Timestamp:
    """
    Retrieve timestamps from a dataframe.
    The `column_name` is expected to be a column that contains timestamps.
    """
    return round_seconds(df[column_name].iloc[0])


def round_seconds(obj: pd.Timestamp) -> pd.Timestamp:
    """
    Round the timestamps to seconds.
    """
    if obj.microsecond >= 500000:
        obj += pd.Timedelta(seconds=1)
    return obj.replace(microsecond=0)


def select_region_where_forces_are_changing(
    df: pd.DataFrame, column_start: str, column_end: str
) -> pd.DataFrame:
    """Filter out the regions where the forces are changing in a linear fashion"""
    # Get the timestamps near the interesting region
    time_start = get_timestamp_from_dataframe(df, column_start)
    time_end = get_timestamp_from_dataframe(df, column_end)

    # Filter out our dataframe considering the time
    df = df[(df.index >= time_start) * (df.index <= time_end)].copy()

    # Find force intersection nearest to zero
    minimum_measured_force_index = df.measured_force.abs().idxmin()
    displacement_where_force_is_minimum = df.displacement.loc[
        minimum_measured_force_index
    ]

    if df.measured_force.abs().min() > MEASURED_FORCE_MAXIMUM_TOLERANCE:
        reason = df.end_reason.iloc[0]
        hp = df.hardpoint.iloc[0]
        print(
            f"Warning - Could not find valid test data "
            f"when {HardpointTest(reason).name} for HP{hp} "
            f"found minimum absolute measured forces {df.measured_force.abs().min()}."
        )

    # Put the zero of the displacement near the minimum measured force
    df.displacement = df.displacement - displacement_where_force_is_minimum

    # Filter out based on the displacement values
    df = df[df.displacement.abs() <= DISPLACEMENT_CROP_RANGE]

    return df
