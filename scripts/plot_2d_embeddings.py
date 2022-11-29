import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binned_statistic_2d

repo_dir = Path(__file__).parents[1]


def plot_entity_density():
    # Load UMAP embeddings
    df = pd.read_parquet(repo_dir / f"assets/data/emb_umap_nn=120.parquet")
    x, y = df.pop("X0"), df.pop("X1")
    # Compute 2D histogram and get the bin number for each entity
    n_pixels = 2000
    heatmap, xedges, yedges, binnumber = binned_statistic_2d(
        x, y, None, statistic="count", bins=n_pixels, expand_binnumbers=True
    )
    # Plot
    image = (heatmap + 1).T
    plt.figure(figsize=(1, 1), dpi=n_pixels)
    plt.axis("off")
    plt.figimage(image, origin="lower", norm=LogNorm(), cmap="hot")
    plt.savefig(
        repo_dir / f"assets/figures/entity_density.png",
        pad_inches=0.0,
    )
    return


def plot_entity_types():
    # Load UMAP embeddings
    df = pd.read_parquet(repo_dir / f"assets/data/emb_umap_nn=120.parquet")
    x, y = df.pop("X0"), df.pop("X1")
    # Compute 2D histogram and get the bin number for each entity
    n_pixels = 2000
    heatmap, xedges, yedges, binnumber = binned_statistic_2d(
        x, y, None, statistic="count", bins=n_pixels, expand_binnumbers=True
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    df[["BN1", "BN2"]] = binnumber.T - 1
    # Join most common entity types to df
    df_types = pd.read_parquet(repo_dir / "assets/data/entity_types.parquet")
    df_types["Type"] = df_types["Type"].str.split("_").str[1:-1].str.join(" ")
    types_color = {
        "administrative district": (1, 0.5, 0),  # Orange
        "person": (0, 0, 1),  # Blue
        "artist": (0, 1, 1),  # Cyan
        "album": (1, 1, 0),  # Yellow
        "movie": (0, 1, 0),  # Green
        "company": (1, 0, 0),  # Red
    }
    mask = df_types["Type"].isin(list(types_color.keys()))
    df_types = df_types.loc[mask]
    df = df.merge(df_types, on="Entity")
    df = pd.get_dummies(df, columns=["Type"], prefix="", prefix_sep="")
    # Group by bin number and take the most frequent category (considering imbalance)
    del df["Entity"]
    df = df.groupby(["BN1", "BN2"]).sum()
    df = df / df.sum()  # Normalize for imbalance
    # # Filter bins with too few entities # Not really useful
    # threshold = 0.9
    # mask = np.zeros(len(df), dtype=np.bool)
    # for col in df.columns:
    #     perc = np.sort(df[col])
    #     cumsum = perc.cumsum()
    #     idx = np.searchsorted(cumsum, 1 - threshold)
    #     th_perc = perc[idx]
    #     mask += (df[col] >= th_perc).to_numpy()
    # df = df.loc[mask]
    df = df.idxmax(axis=1)
    ### Build image
    # Init image with gray pixels where there are entities
    image = np.zeros((n_pixels, n_pixels, 3))
    mask = heatmap > 0
    image[mask] = (0.2, 0.2, 0.2)  # Light gray
    # Color each bin according to its type
    for (ridx, cidx), ent_type in df.iteritems():
        image[ridx, cidx] = types_color[ent_type]
    # Transpose image
    image = np.swapaxes(image, 0, 1)
    ### Plot
    fig = plt.figure(figsize=(1, 1), dpi=n_pixels)
    plt.axis("off")
    plt.figimage(image, origin="lower")
    # Add legend
    patches = [
        mpatches.Patch(color=c, label=l.replace(" ", " "))
        for l, c in types_color.items()
    ]
    fig.legend(
        handles=patches[::-1],
        loc=(0, 0),
        prop={"size": 1.3},
        labelcolor="white",
        framealpha=0,
    )
    plt.savefig(
        repo_dir / f"assets/figures/entity_types.png",
        pad_inches=0.0,
    )
    return


# def plot_one_type(n_neighbors: int, type: str):
#     # Load UMAP embeddings
#     df = pd.read_parquet(repo_dir / f"assets/data/emb_umap_nn={n_neighbors}.parquet")
#     x, y = df.pop("X0"), df.pop("X1")
#     # Compute 2D histogram and get the bin number for each entity
#     n_pixels = 2000
#     heatmap, xedges, yedges, binnumber = binned_statistic_2d(
#         x, y, None, statistic="count", bins=n_pixels, expand_binnumbers=True
#     )
#     binnumber = binnumber.T - 1
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#     # Join most common entity types to df
#     df_types = pd.read_parquet(repo_dir / "assets/data/entity_types.parquet")
#     df_types["Type"] = df_types["Type"].str.split("_").str[1:-1].str.join(" ")
#     entities_to_keep = df_types["Entity"][df_types["Type"] == type]
#     mask = df["Entity"].isin(entities_to_keep)
#     bins_to_keep = binnumber[mask]
#     # Build heatmap
#     heatmap = np.zeros((n_pixels, n_pixels))
#     for ridx, cidx in bins_to_keep:
#         heatmap[ridx, cidx] += 1
#     # Plot
#     dpi = n_pixels / 3
#     plt.figure(dpi=dpi)
#     plt.axis("off")
#     plt.savefig(
#         repo_dir / f"assets/visualization/umap_{type}_nn={n_neighbors}.png",
#         bbox_inches="tight",
#         pad_inches=0,
#         dpi=dpi,
#     )
#     return


# def plot_entity_types2(n_neighbors):
#     # Load UMAP embeddings
#     df = pd.read_parquet(repo_dir / f"assets/data/emb_umap_nn={n_neighbors}.parquet")
#     x, y = df.pop("X0"), df.pop("X1")
#     # Compute 2D histogram and get the bin number for each entity
#     n_pixels = 2000
#     heatmap, xedges, yedges, binnumber = binned_statistic_2d(
#         x, y, None, statistic="count", bins=n_pixels, expand_binnumbers=True
#     )
#     binnumber = binnumber.T - 1
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#     # Join most common entity types to df
#     df_types = pd.read_parquet(repo_dir / "assets/data/entity_types.parquet")
#     df_types["Type"] = df_types["Type"].str.split("_").str[1:-1].str.join(" ")
#     types = [
#         # "administrative district",
#         "person",
#         # "artist",
#         # "album",
#         # "movie",
#         # "company",
#     ]
#     # Update heatmap with type "other"
#     mask = heatmap > 0
#     heatmap[mask] = 1
#     for k, type in enumerate(types):
#         entities_to_keep = df_types["Entity"][df_types["Type"] == type]
#         mask = df["Entity"].isin(entities_to_keep)
#         bins_to_keep = binnumber[mask]
#         # Update heatmap
#         for ridx, cidx in bins_to_keep:
#             heatmap[ridx, cidx] = k + 2
#     # Plot
#     cmap = plt.get_cmap("tab10", len(types) + 2)
#     colors = [cmap(k) for k in range(len(types) + 2)]
#     colors[0] = (0, 0, 0, 1)
#     colors[1] = (0.2, 0.2, 0.2, 1)
#     cmap = LinearSegmentedColormap.from_list("", colors, len(types) + 2)
#     plt.figure(figsize=(1, 1), dpi=n_pixels)
#     plt.axis("off")
#     plt.figimage(
#         heatmap.T,
#         origin="lower",
#         cmap=cmap,
#     )
#     # Add legend
#     # color_labels = ["unkwown"] + types
#     # patches = [
#     #     mpatches.Patch(color=colors[i + 1], label=color_labels[i])
#     #     for i in range(len(color_labels))
#     # ]
#     # plt.legend(
#     #     handles=patches,
#     #     bbox_to_anchor=(0, 0),
#     #     loc=2,
#     #     borderaxespad=0.0,
#     #     ncol=7,
#     #     prop={"size": 3.21},
#     # )
#     plt.savefig(
#         repo_dir / f"assets/visualization/umap_types_nn={n_neighbors}.png",
#         # pad_inches=0,
#     )
