import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


def set_plot_style():
    # Use a qualitative colormap which is beneficial for categorical data
    colors = cycler(color=plt.get_cmap("Set3").colors)

    mpl.style.use("ggplot")

    # Larger figure size to accommodate more complex plots
    mpl.rcParams["figure.figsize"] = (25, 10)

    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams[
        "axes.edgecolor"
    ] = "gray"  # Edge color to highlight the boundary of the plot area
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.color"] = "lightgray"
    mpl.rcParams["axes.prop_cycle"] = colors

    # Increased linewidth for axes for better visibility
    mpl.rcParams["axes.linewidth"] = 1.5

    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"

    # Slightly larger font size for better readability
    mpl.rcParams["font.size"] = 14

    # Adjusting title size to be proportionate with the figure size
    mpl.rcParams["axes.titlesize"] = 20
    mpl.rcParams["figure.titlesize"] = 25

    # High-resolution figures for clearer visuals
    mpl.rcParams["figure.dpi"] = 150
