import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from gridpoint_engine import GridPointOptimizer
from geo_utils import latlon_to_xy, xy_to_latlon

# ============================================================
# PAGE CONFIGURATION

st.set_page_config(
    page_title="GRIDPOINT",
    page_icon="📍",
    layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 28px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE

st.title("📍 GRIDPOINT")

st.subheader(
    "Mathematical Warehouse Location Optimization"
)

st.write(
    """
    GRIDPOINT determines warehouse locations and neighborhood
    assignments by minimizing demand-weighted delivery distance
    while respecting operational constraints.
    """
)

# ============================================================
# LOAD GEOGRAPHICAL DATA

st.sidebar.header("📂 Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload neighborhood CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(
        uploaded_file
    )

else:

    data = pd.read_csv(
        "data/sample_neighborhoods.csv"
    )


required_columns = {
    "neighborhood",
    "latitude",
    "longitude",
    "daily_orders"
}

missing_columns = (
    required_columns
    - set(data.columns)
)

if missing_columns:

    st.error(
        f"Missing columns: {missing_columns}"
    )

    st.stop()

points, reference_lat, reference_lon = (
    latlon_to_xy(
        data["latitude"].values,
        data["longitude"].values
    )
)

demands = data[
    "daily_orders"
].values.astype(float)

# ============================================================
# SIDEBAR

st.sidebar.header("⚙️ Optimization Settings")
number_of_warehouses = st.sidebar.slider(
    "Number of Warehouses",
    min_value=1,
    max_value=5,
    value=3
)

warehouse_capacity = st.sidebar.number_input(
    "Warehouse Capacity",
    min_value=100,
    max_value=10000,
    value=2500,
    step=100
)

maximum_radius = st.sidebar.number_input(
    "Maximum Service Radius",
    min_value=1.0,
    max_value=100.0,
    value=10.0,
    step=1.0
)

cost_per_order_km = st.sidebar.number_input(
    "Delivery Cost (₹ / order-km)",
    min_value=1.0,
    max_value=500.0,
    value=20.0,
    step=1.0
)

warehouse_operating_cost = st.sidebar.number_input(
    "Warehouse Operating Cost (₹ / day)",
    min_value=0.0,
    max_value=1000000.0,
    value=10000.0,
    step=1000.0
)

optimize_button = st.sidebar.button(
    "🚀 OPTIMIZE",
    type="primary"
)

analysis_button = st.sidebar.button(
    "📊 Analyze Warehouse Count"
)


# ============================================================
# DISPLAY INPUT DATA

st.header("📊 Neighborhood Data")

display_data = data[
    [
        "neighborhood",
        "latitude",
        "longitude",
        "daily_orders"
    ]
].copy()

display_data.columns = [
    "Neighborhood",
    "Latitude",
    "Longitude",
    "Daily Orders"
]

st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# RUN OPTIMIZATION

if optimize_button:

    optimizer = GridPointOptimizer(
        n_warehouses=number_of_warehouses,
        max_iterations=100,
        tolerance=1e-5,
        restarts=10,
        random_state=42
    )

    # --------------------------------------------------------
    # Unconstrained optimization

    with st.spinner(
        "Running mathematical optimization..."
    ):

        result = optimizer.fit(
            points,
            demands
        )
    st.session_state["optimization_result"] = result

    warehouses = result["warehouses"]

    warehouse_latitudes = []
    warehouse_longitudes = []

    for warehouse in warehouses:

        lat, lon = xy_to_latlon(
            warehouse[0],
            warehouse[1],
            reference_lat,
            reference_lon
        )

        warehouse_latitudes.append(lat)
        warehouse_longitudes.append(lon)
    warehouse_lat = np.array(warehouse_latitudes)
    warehouse_lon = np.array(warehouse_longitudes)
    st.session_state["warehouse_latitudes"] = warehouse_latitudes
    st.session_state["warehouse_longitudes"] = warehouse_longitudes

    # --------------------------------------------------------
    # Baseline

    baseline_warehouses = np.array([
        points.mean(axis=0)
        for _ in range(number_of_warehouses)
    ])

    baseline = optimizer.evaluate_solution(
        points,
        demands,
        baseline_warehouses
    )

    # --------------------------------------------------------
    # Constrained assignment

    capacities = np.full(
        number_of_warehouses,
        warehouse_capacity
    )

    try:

        constrained = (
            optimizer
            .evaluate_constrained_solution(
                points,
                demands,
                warehouses,
                capacities,
                maximum_radius
            )
        )
        st.session_state["constrained_solution"] = constrained

        feasible = True

    except ValueError:

        constrained = None

        feasible = False


    # ========================================================
    # RESULTS

    st.header("📈 Optimization Results")

    if not feasible:
        st.error(
            """
            No feasible solution exists for the selected
            capacity and service-radius constraints.

            Try increasing warehouse capacity or service radius.
            """
        )
    else:
        baseline_cost = (
            baseline[
                "total_weighted_distance"
            ]
        )
        optimized_cost = (
            constrained[
                "total_cost"
            ]
        )
        delivery_cost = (
            optimized_cost
            * cost_per_order_km
        )
        improvement = (
            (
                baseline_cost
                - optimized_cost
            )
            / baseline_cost
        ) * 100
        # ----------------------------------------------------
        # METRICS

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Baseline Demand-Weighted Distance",
            f"{baseline_cost:.2f} order-km"
        )
        col2.metric(
            "Optimized Demand-Weighted Distance",
            f"{optimized_cost:.2f} order-km"
        )
        col3.metric(
            "Improvement",
            f"{improvement:.2f}%"
        )
        col4.metric(
            "Unassigned Demand",
            f"{constrained['unassigned_demand']:.0f}"
        )
        # ====================================================
        # WAREHOUSE TABLE

        st.subheader(
            "🏭 Optimized Warehouse Locations"
        )
        warehouse_data = []

        for i, warehouse in enumerate(
            warehouses
        ):

            used = (
                warehouse_capacity
                - constrained[
                    "remaining_capacity"
                ][i]
            )

            utilization = (
                used /
                warehouse_capacity
            ) * 100

            warehouse_data.append({

                "Warehouse":
                    f"W{i + 1}",

                "X":
                    round(
                        warehouse[0],
                        2
                    ),

                "Y":
                    round(
                        warehouse[1],
                        2
                    ),

                "Orders Served":
                    int(used),

                "Capacity":
                    int(
                        warehouse_capacity
                    ),

                "Utilization":
                    f"{utilization:.1f}%"
            })


        warehouse_df = pd.DataFrame(
            warehouse_data
        )

        st.dataframe(
            warehouse_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # ASSIGNMENTS

        st.subheader(
            "📦 Neighborhood Assignments"
        )

        assignment_data = []

        for i, warehouse_id in enumerate(
            constrained["assignments"]
        ):

            assignment_data.append({

                "Neighborhood":
                    f"N{i + 1}",

                "Daily Orders":
                    int(demands[i]),

                "Assigned Warehouse":
                    f"W{warehouse_id + 1}"
            })


        assignment_df = pd.DataFrame(
            assignment_data
        )

        st.dataframe(
            assignment_df,
            use_container_width=True,
            hide_index=True
        )



# ============================================================
# GRIDPOINT NETWORK MAP

result = st.session_state.get("optimization_result")
constrained_result = st.session_state.get("constrained_solution")

if result is not None and constrained_result is not None:

    st.header("🗺️ GRIDPOINT Network Map")

    assignments = np.asarray(
        constrained_result["assignments"],
        dtype=int,
    )

    # ============================================================
    # NEIGHBORHOOD DATA

    neighborhood_map = pd.DataFrame({
        "name": data["neighborhood"].values,
        "latitude": data["latitude"].values,
        "longitude": data["longitude"].values,
        "orders": data["daily_orders"].values,
        "type": "Neighborhood",
    })

    # ============================================================
    # WAREHOUSE DATA

    warehouse_latitudes = np.asarray(
        st.session_state.get("warehouse_latitudes", []),
        dtype=float,
    )

    warehouse_longitudes = np.asarray(
        st.session_state.get("warehouse_longitudes", []),
        dtype=float,
    )

    warehouse_map = pd.DataFrame({
        "warehouse": [
            f"Warehouse {i + 1}"
            for i in range(len(warehouse_latitudes))
        ],
        "latitude": warehouse_latitudes,
        "longitude": warehouse_longitudes,
        "type": "Warehouse",
    })

    # ============================================================
    # ASSIGNMENT CONNECTIONS

    assignment_rows = []

    for i, warehouse_id in enumerate(assignments):

        if warehouse_id < 0:
            continue

        if warehouse_id >= len(warehouse_map):
            continue

        assignment_rows.append({
            "source_lat": neighborhood_map.loc[i, "latitude"],
            "source_lon": neighborhood_map.loc[i, "longitude"],

            "target_lat": warehouse_map.loc[
                warehouse_id,
                "latitude"
            ],

            "target_lon": warehouse_map.loc[
                warehouse_id,
                "longitude"
            ],

            "orders": neighborhood_map.loc[i, "orders"],

            "neighborhood": neighborhood_map.loc[
                i,
                "name"
            ],

            "warehouse": warehouse_map.loc[
                warehouse_id,
                "warehouse"
            ],
        })

    assignment_map = pd.DataFrame(assignment_rows)
    # ============================================================
    # NEIGHBORHOOD LAYER

    neighborhood_layer = pdk.Layer(
        "ScatterplotLayer",

        data=neighborhood_map,

        get_position=[
            "longitude",
            "latitude"
        ],

        # Small, clean neighborhood markers
        get_radius=180,

        # Blue
        get_fill_color=[40, 140, 255],

        get_line_color=[255, 255, 255],

        get_line_width=1,

        pickable=True,

        opacity=0.85,

        stroked=True,

        filled=True,
    )


    # ============================================================
    # WAREHOUSE LAYER

    warehouse_layer = pdk.Layer(
        "ScatterplotLayer",

        data=warehouse_map,

        get_position=[
            "longitude",
            "latitude"
        ],

        # Larger than neighborhoods
        get_radius=500,

        # Orange
        get_fill_color=[255, 140, 0],

        get_line_color=[255, 255, 255],

        get_line_width=3,

        pickable=True,

        opacity=1.0,

        stroked=True,

        filled=True,
    )

    # ============================================================
    # ASSIGNMENT CONNECTIONS

    layers = []

    if not assignment_map.empty:

        assignment_layer = pdk.Layer(
            "ArcLayer",

            data=assignment_map,

            get_source_position=[
                "source_lon",
                "source_lat"
            ],

            get_target_position=[
                "target_lon",
                "target_lat"
            ],

            # Thin connection lines
            get_width=1.5,

            # Subtle blue → orange connection
            get_source_color=[100, 180, 255],

            get_target_color=[255, 170, 80],

            get_tilt=0,

            pickable=True,

            opacity=0.35,
        )

        layers.append(assignment_layer)


    layers.append(neighborhood_layer)
    layers.append(warehouse_layer)
    # ============================================================
    # CREATE MAP

    deck = pdk.Deck(
        layers=layers,

        initial_view_state=pdk.ViewState(
            latitude=float(data["latitude"].mean()),
            longitude=float(data["longitude"].mean()),
            zoom=10.5,
            pitch=35,
        ),

        tooltip={
            "html": """
                <b>{name}</b>
                <br/>
                Orders: {orders}
                <br/>
                Warehouse: {warehouse}
            """,

            "style": {
                "backgroundColor": "black",
                "color": "white",
                "fontSize": "14px",
                "padding": "10px",
            },
        },
    )


    # ============================================================
    # DISPLAY MAP
    st.pydeck_chart(
        deck,
        use_container_width=True
    )

    # ============================================================
    # LEGEND

    st.markdown(
        """
        **Map Legend**

        🔵 **Blue points** → Neighborhoods  
        🟠 **Orange points** → Optimized warehouses  
        ── **Connection lines** → Neighborhood → Assigned warehouse
        """
    )


# ============================================================
# MATHEMATICAL MODEL

if result is not None and constrained_result is not None:
    st.subheader("🧮 Mathematical Model")

    st.markdown(
        r"""
### Objective

The constrained assignment minimizes demand-weighted delivery distance:

$$
\min
\sum_{i=1}^{n}
\sum_{j=1}^{k}
d_i D_{ij} x_{ij}
$$

where:

- $d_i$ = daily demand at neighborhood $i$
- $D_{ij}$ = distance from neighborhood $i$ to warehouse $j$
- $x_{ij}$ = binary assignment variable

### Constraints

**Every neighborhood is assigned exactly once:**

$$
\sum_j x_{ij}=1
$$

**Warehouse capacity:**

$$
\sum_i d_i x_{ij}\leq C_j
$$

**Maximum service radius:**

$$
D_{ij}\leq R
$$

**Binary assignment:**

$$
x_{ij}\in\{0,1\}
$$

The warehouse locations are obtained using nearest-warehouse assignment,
followed by demand-weighted geometric-median updates with the Weiszfeld
algorithm. The final constrained neighborhood assignment is solved using
Mixed-Integer Linear Programming (MILP).
"""
    )


# ============================================================
# WAREHOUSE COUNT ANALYSIS

if analysis_button:
    st.header("📊 Warehouse Count Analysis")

    analysis_optimizer = GridPointOptimizer(
        n_warehouses=number_of_warehouses,
        max_iterations=100,
        tolerance=1e-5,
        restarts=10,
        random_state=42,
    )

    with st.spinner("Evaluating warehouse configurations..."):
        scenario_results = analysis_optimizer.analyze_warehouse_counts(
            points,
            demands,
            max_warehouses=6,
            warehouse_capacity=warehouse_capacity,
            max_radius=maximum_radius,
            delivery_cost_per_order_km=cost_per_order_km,
            warehouse_operating_cost=warehouse_operating_cost,
        )

    scenario_df = pd.DataFrame(scenario_results)

    feasible_df = scenario_df[
        scenario_df["feasible"] == True
    ].copy()

    if feasible_df.empty:
        st.error("No feasible warehouse configuration was found.")
    else:
        display_scenarios = feasible_df[
            [
                "warehouses",
                "delivery_distance",
                "delivery_cost",
                "infrastructure_cost",
                "total_cost",
            ]
        ].copy()

        display_scenarios.columns = [
            "Warehouses",
            "Demand-Weighted Distance",
            "Delivery Cost",
            "Infrastructure Cost",
            "Total Cost",
        ]

        st.dataframe(
            display_scenarios,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Total Cost vs Number of Warehouses")

        chart_data = (
            feasible_df[
                [
                    "warehouses",
                    "delivery_cost",
                    "infrastructure_cost",
                    "total_cost",
                ]
            ]
            .set_index("warehouses")
        )

        st.line_chart(chart_data)


# ============================================================
# INITIAL STATE MESSAGE

if result is None and not analysis_button:
    st.info(
        "Configure the parameters in the sidebar and click "
        "🚀 OPTIMIZE to run GRIDPOINT."
    )
