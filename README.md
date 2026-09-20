# GRIDPOINT 📍
### Mathematical Warehouse Location & Distribution Optimization

GRIDPOINT is a mathematical optimization system that determines **warehouse locations and neighborhood assignments** while considering demand, warehouse capacity, delivery distance, and maximum service radius.

It converts a real-world logistics problem into a **constrained optimization problem** and provides an interactive interface for analyzing and visualizing the solution.

## 🧮 Mathematical Model

### Objective

\[
\min \sum_{i=1}^{n}\sum_{j=1}^{k} d_iD_{ij}x_{ij}
\]

where:

- \(d_i\) = demand at neighborhood \(i\)
- \(D_{ij}\) = distance between neighborhood \(i\) and warehouse \(j\)
- \(x_{ij}\) = binary assignment variable

### Constraints

**Every neighborhood is assigned:**

\[
\sum_j x_{ij}=1
\]

**Warehouse capacity:**

\[
\sum_i d_ix_{ij}\leq C_j
\]

**Maximum service radius:**

\[
D_{ij}\leq R
\]

**Binary assignment:**

\[
x_{ij}\in\{0,1\}
\]

Warehouse locations are optimized using a **demand-weighted geometric median / Weiszfeld approach**, followed by constrained assignment evaluation.

## ✨ Features

- Demand-weighted warehouse location optimization
- Weiszfeld geometric-median optimization
- Capacity-constrained assignment
- Maximum service-radius constraint
- Feasibility detection
- Multiple optimization restarts
- Baseline vs optimized comparison
- Warehouse-count analysis
- Delivery and infrastructure cost analysis
- Interactive PyDeck network map
- Streamlit interface

## 📊 Demonstration

For the demonstration dataset:

| Metric | Result |
|---|---:|
| Daily demand | 6,500 orders |
| Warehouse capacity | 2,500 orders |
| 2 warehouses | Infeasible |
| 3 warehouses | Feasible |
| Baseline distance | 38,847.42 |
| Optimized distance | 26,721.21 |
| Reduction | **31.21%** |
| Unassigned demand | **0** |

*Results depend on the input dataset and selected parameters.*

## 🛠️ Tech Stack

- **Python** — Core implementation
- **NumPy** — Numerical computation
- **Pandas** — Data processing
- **Streamlit** — Interactive application
- **PyDeck** — Geospatial visualization

## 📁 Structure

```text
GRIDPOINT/
├── app.py
├── gridpoint_engine.py
├── geo_utils.py
├── data/
│   └── sample_neighborhoods.csv
└── README.md
```

## ▶️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/GRIDPOINT.git
cd GRIDPOINT

python -m venv venv
venv\Scripts\activate

pip install streamlit pandas numpy pydeck
streamlit run app.py
```

## 📚 Open-Source & Reference Credits

GRIDPOINT uses and acknowledges the following open-source projects and mathematical references:

- **Streamlit** — application framework and interactive UI  
  https://streamlit.io/
- **NumPy** — numerical computing  
  https://numpy.org/
- **Pandas** — data processing and analysis  
  https://pandas.pydata.org/
- **PyDeck / deck.gl** — geospatial visualization  
  https://pydeck.gl/
- **Weiszfeld Algorithm** — mathematical basis for geometric-median location optimization.

All third-party software remains under its respective licenses. This project does not claim ownership of the referenced libraries or mathematical methods.

## 👨‍💻 Project

**GRIDPOINT** was developed as an individual hackathon project exploring the application of mathematical optimization to real-world warehouse and distribution planning.
