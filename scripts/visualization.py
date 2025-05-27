import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px



def plot_time_series(df, x_axis, y_axis, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
    ax.set_title(f"{y_axis} over Time")
    plt.grid()
    return fig  




def plot_time_series_double(df, col1, col2, x_axis='year', figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=df, x=x_axis, y=col1, ax=ax, label=col1)
    sns.lineplot(data=df, x=x_axis, y=col2, ax=ax, label=col2)
    ax.set_title(f"{col1} and {col2} over Time")
    return fig


def plot_correlation_heatmap(data):
    fig, ax = plt.subplots(figsize = (12,8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    return fig





def other_visualize(
    df: pd.DataFrame,
    plot_type: str,
    x: str = None,
    y: str = None,
    hue: str = None,
    group_by: str = None,
    cols: list = None,
    title: str = '',
    sample_frac: float = 1.0,
    **kwargs
):
    """
    Modular visualization function for different EDA plots.
    """
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac)

    fig = None

    if plot_type == 'box':
        fig = px.box(df, x=x, y=y, color=hue, title=title)

    elif plot_type == 'line':
        fig = px.line(df, x=x, y=y, color=hue, title=title)

    elif plot_type == 'dist':
        fig = px.histogram(df, x=x, color=hue, marginal="box", nbins=kwargs.get("bins", 50), title=title)

    elif plot_type == 'scatter':
        fig = px.scatter(df, x=x, y=y, color=hue, title=title)

    elif plot_type == 'bar':
        grouped = df.groupby(group_by)[y].mean().reset_index()
        fig = px.bar(grouped, x=group_by, y=y, color=group_by, title=title)

    elif plot_type == 'pairplot':
        subset = df[cols].dropna()
        sns.pairplot(subset.sample(frac=sample_frac), hue=hue)
        fig = plt.gcf()

    elif plot_type == 'heatmap_corr':

        corr = df[cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(title)
        fig = plt.gcf()

    return fig



def plot_actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    return fig

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=30, edgecolor='black')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title('Residuals Distribution')
    return fig