import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pickle

def load_stats(file_path):
    with open(file_path, "rb") as f:
        stats = pickle.load(f)
    return stats

def plot_stats(stats, title):
    df = pd.DataFrame(stats)
    df["average_over_100"] = df["reward"].rolling(window=100).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["episode"],
            y=df["reward"],
            name="Total Reward",
            yaxis="y1"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["episode"],
            y=df["average_over_100"],
            name="Average Reward (100 eps)",
            yaxis="y1"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["episode"],
            y=df["epsilon"],
            name="Epsilon",
            yaxis="y2"
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Episode"),
        yaxis=dict(
            title="Reward",
            rangemode="tozero"
        ),
        yaxis2=dict(
            title="Epsilon",
            overlaying="y",
            side="right",
            rangemode="tozero"
        ),
        legend=dict(x=0.01, y=0.99)
    )

    return fig 

stats_0_05 = load_stats("training_stats.pkl")
fig = plot_stats(stats_0_05, "Training Stats for DQN Agent with discrete set of actions lr = 0.001, eps_min = 0.05")
fig.show()