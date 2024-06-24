import seaborn as sns


def setup_plot_style():
    """
    Util to have a consistent style for all matplotlib plots.
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=2)
