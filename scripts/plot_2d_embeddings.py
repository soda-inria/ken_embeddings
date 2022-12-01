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


def plot_entity_types(add_entities: bool):
    # Load UMAP embeddings and assign entities to bins
    df = pd.read_parquet(repo_dir / f"assets/data/emb_umap_nn=120.parquet")
    x, y = df.pop("X0"), df.pop("X1")
    # Compute 2D histogram and get the bin number for each entity
    n_pixels = 2000
    heatmap, xedges, yedges, binnumber = binned_statistic_2d(
        x, y, None, statistic="count", bins=n_pixels, expand_binnumbers=True
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    df[["BN1", "BN2"]] = binnumber.T - 1
    df_bins = df.copy()

    ### Join most common entity types to df
    df_types = pd.read_parquet(repo_dir / "assets/data/entity_types.parquet")
    # Filter only the types we want
    clean_types = {
        "<wordnet_person_100007846>": "person",
        "<yagoGeoEntity>": "location",
        "<wordnet_album_106591815>": "album",
        "<wordnet_movie_106613686>": "movie",
        "<wordnet_company_108058098>": "company",
        "<wordnet_school_108276720>": "school",
    }
    mask = df_types["Type"].isin(clean_types.keys())
    df_types = df_types[mask].copy()
    df_types["Type"] = df_types["Type"].map(clean_types)
    
    # Include subtypes into more common types
    type_colors = {
        "location": (1, 0.5, 0),  # Orange
        "person": (0.1, 0.4, 1),  # Blue
        "album": (0.9, 0.9, 0),  # Yellow
        "movie": (0, 1, 0),  # Green
        "company": (1, 0, 0),  # Red
        "school": (0.5, 0, 1), # Purple
    }
    mask = df_types["Type"].isin(list(type_colors.keys()))
    df_types = df_types.loc[mask].copy()
    df = df.merge(df_types, on="Entity")
    df = pd.get_dummies(df, columns=["Type"], prefix="", prefix_sep="")

    # Group by bin number and take the most frequent category (considering imbalance)
    del df["Entity"]
    df = df.groupby(["BN1", "BN2"]).sum()
    df = df / df.sum()  # Normalize for imbalance
    df = df.idxmax(axis=1)

    # Init image with gray pixels where there are entities
    image = np.zeros((n_pixels, n_pixels, 3))
    mask = heatmap > 0
    image[mask] = (0.2, 0.2, 0.2)  # Light gray
    # Color each bin according to its type
    for (ridx, cidx), ent_type in df.iteritems():
        image[ridx, cidx] = type_colors[ent_type]
    # Transpose image
    image = np.swapaxes(image, 0, 1)

    ### Plot
    fig = plt.figure(figsize=(1, 1), dpi=n_pixels)
    ax = plt.Axes(fig, (0, 0, 1, 1), xticks=[], yticks=[], frameon=False)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    fig.add_axes(ax)
    ax.imshow(image, origin="lower")
    # Add legend
    patches = [
        mpatches.Patch(color=c, label=l.replace(" ", " "))
        for l, c in type_colors.items()
    ]
    ax.legend(
        handles=patches,
        loc=(0, 0),
        prop={"size": 1.3},
        labelcolor="white",
        framealpha=0,
    )
    # Add some entities
    entity_shifts = {
        # Locations
        "<Paris>": (-5, -30),
        "<France>": (-5, 8),
        "<Japan>": (0, -30),
        "<United_States>": (-190, -25),
        "<Berlin>": (-10, -30),
        "<Germany>": (5, 0),
        "<Rome>": (5, -20),
        "<Italy>": (-5, 10),
        "<United_Kingdom>": (5, 0),
        "<India>": (8, -20),
        "<Brazil>": (8, -3),
        "<Argentina>": (8, -7),
        "<China>": (5, 0),
        "<Russia>": (8, -10),
        # Politicians
        "<Joe_Biden>": (-10, 10),
        "<Donald_Trump>": (5, -20),
        # Football players
        "<Lionel_Messi>": (5, 0),
        "<Cristiano_Ronaldo>": (5, 0),
        # Artists
        "<Freddie_Mercury>": (-20, -30),
        "<Michael_Jackson>": (-5, 8),
        # Movies
        "<Avatar_(2009_film)>": (5, 18),
        "<Titanic_(1997_film)>": (15, -15),
        # Albums
        "<Thriller_(Michael_Jackson_album)>": (-95, -30),
        "<Abbey_Road>": (5, -20),
        # Companies
        "<Google>": (5, 5),
        "<Apple_Inc.>": (5, -25),
        "<Toyota>": (-5, 10),
        # Scientists
        "<Isaac_Newton>": (5, 0),
        "<Galileo_Galilei>": (5, 0),
        # Universities
        "<University_of_Oxford>": (5, 0),
        "<Harvard_University>": (5, 0),
        "<Peking_University>": (-10, -30),
        
    }
    if add_entities:
        mask = df_bins["Entity"].isin(entity_shifts.keys())
        for _, ent, x, y in df_bins.loc[mask].itertuples():
            sx, sy = entity_shifts[ent]
            clean_ent = ent[1:-1].replace("_", " ")
            if any(s in clean_ent for s in ["film", "album", "Inc."]):
                clean_ent = clean_ent.split(" ")[0]
            ax.scatter(x, y, s=0.07, c="white", edgecolors="none")
            ax.text(x + sx, y + sy, clean_ent, c="white", size=0.5)
        # Save figure
        plt.savefig(repo_dir / f"assets/figures/entity_types_with_names.png")
    else:
        plt.savefig(repo_dir / f"assets/figures/entity_types.png")
    plt.close()
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
