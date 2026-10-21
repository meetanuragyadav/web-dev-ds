"""Plotly chart builders. Each function takes a (filtered) DataFrame and returns a Figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_white"
FREQUENCIES = {"Week": "W", "Month": "MS", "Quarter": "QS"}


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return not df.empty and all(c in df.columns for c in cols)


def sales_profit_trend(df: pd.DataFrame, freq_label: str = "Month") -> go.Figure:
    """Sales (bars) and profit (line) over time."""
    ts = (df.groupby(pd.Grouper(key="Order Date", freq=FREQUENCIES[freq_label]))[["Sales", "Profit"]]
          .sum().reset_index())
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=ts["Order Date"], y=ts["Sales"], name="Sales", opacity=0.75)
    fig.add_scatter(x=ts["Order Date"], y=ts["Profit"], name="Profit", mode="lines+markers",
                    line=dict(width=3, color="#d62728"), secondary_y=True)
    fig.update_layout(template=TEMPLATE, title=f"Sales and Profit by {freq_label}", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=60, b=10))
    fig.update_yaxes(title_text="Sales ($)", secondary_y=False)
    fig.update_yaxes(title_text="Profit ($)", secondary_y=True)
    return fig


def category_bar(df: pd.DataFrame, dimension: str = "Category", metric: str = "Sales") -> go.Figure:
    """Metric by a chosen dimension, coloured by profit margin."""
    g = df.groupby(dimension).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                                  Quantity=("Quantity", "sum")).reset_index()
    g["Profit Margin (%)"] = (100 * g["Profit"] / g["Sales"].replace(0, np.nan)).round(2)
    g = g.sort_values(metric)
    fig = px.bar(g, x=metric, y=dimension, orientation="h", color="Profit Margin (%)",
                 color_continuous_scale="RdYlGn", color_continuous_midpoint=0, text_auto=".3s",
                 hover_data=["Sales", "Profit", "Quantity"], template=TEMPLATE,
                 title=f"{metric} by {dimension} (colour = profit margin)")
    fig.update_layout(margin=dict(t=60, b=10), yaxis_title=None)
    return fig


def region_treemap(df: pd.DataFrame) -> go.Figure:
    """Sales share by Region > Category; colour shows profit margin."""
    g = df.groupby(["Region", "Category"]).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    g = g[g["Sales"] > 0]
    g["Profit Margin (%)"] = (100 * g["Profit"] / g["Sales"]).round(2)
    fig = px.treemap(g, path=[px.Constant("All regions"), "Region", "Category"], values="Sales",
                     color="Profit Margin (%)", color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=0, template=TEMPLATE, title="Sales Share: Region > Category")
    fig.update_layout(margin=dict(t=60, b=10))
    return fig


def discount_profit_scatter(df: pd.DataFrame) -> go.Figure:
    """Discount vs profit margin per order line."""
    d = df[df["Sales"] > 0].copy()
    d["Profit Margin (%)"] = 100 * d["Profit"] / d["Sales"]
    color = "Category" if "Category" in d else None
    fig = px.strip(d, x="Discount", y="Profit Margin (%)", color=color, template=TEMPLATE,
                   hover_data=[c for c in ["Order ID", "Product Name", "Sales", "Quantity"] if c in d],
                   title="Profit Margin by Discount Level")
    fig.update_xaxes(type="category", categoryorder="category ascending", tickformat=".0%")
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(margin=dict(t=60, b=10))
    return fig


def margin_heatmap(df: pd.DataFrame) -> go.Figure:
    """Profit margin (%) for Region x Category."""
    p = df.pivot_table(index="Region", columns="Category", values=["Sales", "Profit"], aggfunc="sum")
    margin = (100 * p["Profit"] / p["Sales"].replace(0, np.nan)).round(1)
    fig = px.imshow(margin, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0, template=TEMPLATE,
                    title="Profit Margin (%) by Region and Category")
    fig.update_layout(margin=dict(t=60, b=10))
    return fig


def segment_shipmode_bar(df: pd.DataFrame) -> go.Figure:
    """Orders by segment, split by ship mode."""
    g = df.groupby(["Segment", "Ship Mode"])["Order ID"].nunique().reset_index(name="Orders")
    fig = px.bar(g, x="Segment", y="Orders", color="Ship Mode", barmode="group", template=TEMPLATE,
                 title="Unique Orders by Segment and Ship Mode")
    fig.update_layout(margin=dict(t=60, b=10))
    return fig


def correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    corr = df[cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    template=TEMPLATE, title="Correlation Matrix")
    fig.update_layout(margin=dict(t=60, b=10))
    return fig


def distribution_box(df: pd.DataFrame, column: str, by: str | None = None) -> go.Figure:
    fig = px.box(df, y=column, x=by, points="outliers", template=TEMPLATE,
                 title=f"Distribution of {column}" + (f" by {by}" if by else ""))
    fig.update_layout(margin=dict(t=60, b=10))
    return fig
