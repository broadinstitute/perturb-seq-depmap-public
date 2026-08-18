import os

import matplotlib.pyplot as plt
import seaborn as sns

METADATA_DIR = os.path.join("..", "metadata")
DATA_DIR = os.path.join("..", "data")
DOWNLOADED_DIR = os.path.join("..", "downloaded")
PROCESSED_DIR = os.path.join("..", "processed")
FIGURE_DIR = os.path.join("..", "figures")
SCRATCH_DIR = os.path.join(FIGURE_DIR, "scratch")

# conversions from inches
cm = 1 / 2.54
mm = cm / 10
pt = 1 / 72

# font sizes (in cases where we manually override the stylesheet)
TITLE_SIZE = 7
LABEL_SIZE = 6
TICK_SIZE = 5
ANNOT_SIZE = 5
POINT_SIZE = 8

# annotation arrow styles
annot_arrow_props = dict(
    arrowstyle="-", color="black", alpha=1, linewidth=0.25, shrinkA=1, shrinkB=1
)

highlight_color = "#f87060"
primary_colors = ["tab:blue", "tab:olive", "tab:red"]


def blend_colors(colors):
    if "tab:blue" in colors and "tab:olive" in colors and "tab:red" in colors:
        return "tab:brown"
    if "tab:blue" in colors and "tab:olive" in colors:
        return "tab:green"
    if "tab:blue" in colors and "tab:red" in colors:
        return "tab:purple"
    if "tab:red" in colors and "tab:olive" in colors:
        return "tab:orange"
    if "tab:blue" in colors:
        return "tab:blue"
    if "tab:red" in colors:
        return "tab:red"
    if "tab:olive" in colors:
        return "tab:olive"
    else:
        return "black"


gene_class_palette = {
    "Common Essential": "tab:blue",
    "Selective": "tab:red",
    "High Variance": "tab:orange",
    "Olfactory Control": "tab:green",
}

lineage_palette = {
    "Bone": "#8dd3c7",
    "Bowel": "#ffffb3",
    "CNS/Brain": "#bebada",
    "Cervix": "#fb8072",
    "Esophagus/Stomach": "#80b1d3",
    "Head and Neck": "#fdb462",
    "Kidney": "#b3de69",
    "Ovary/Fallopian Tube": "#fccde5",
    "Pleura": "#ffed6f",
    "Skin": "#bc80bd",
    "Biliary Tract": "#ccebc5",
    "Other": "#d9d9d9",
}

truncation_cell_group_palette = {
    "Normal KO": "tab:green",
    "Passing\nsinglet": "tab:green",
    "Arm loss KO": "tab:blue",
    "Singlet with\narm loss": "tab:blue",
    "Arm gain KO": "tab:red",
    "Singlet with\narm gain": "tab:red",
    "Other control KO": "tab:purple",
    "Unperturbed": "tab:orange",
}

oncoprint_palette = {
    "Amplification": plt.cm.coolwarm(256),  #'tab:red',
    "Deletion": plt.cm.coolwarm(0),  #'tab:blue',
    "Neutral Copy": "white",
}

sns.color_palette("Set3", 12)
