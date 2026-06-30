import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from vinecopulas.marginals import best_fit_distribution
from vinecopulas.vinecopula import fit_vinecop


# Candidate copula families and their numeric IDs used by vinecopulas
copula_names = {
    1: "Gaussian", 2: "Gumbel0", 3: "Gumbel90", 4: "Gumbel180", 5: "Gumbel270",
    6: "Clayton0", 7: "Clayton90", 8: "Clayton180", 9: "Clayton270",
    10: "Frank", 11: "Joe0", 12: "Joe90", 13: "Joe180", 14: "Joe270", 15: "Student",
}

def load_dataset(datapath, drop_cols=("level_0", "index", "Unnamed: 0", "person_id"),
                 short_items=None):
    """
    Load a CSV dataset and impute missing values.

    Note:
        Fill remaining NaNs with per-column medians.

    Arguments:
        datapath (str):   Path to the CSV file.
        drop_cols (tuple): Housekeeping/ID columns to drop if present.
        short_items (dict | None): Optional column-rename mapping
                                   (long name -> short name).

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    df = pd.read_csv(datapath)
    df = df.drop(columns=list(drop_cols), errors="ignore")
    print(f"{datapath} | missing rate: {round(df.isna().sum().sum() / df.size, 4)}")

    # Impute missing values with the median of each numeric column
    df = df.fillna(df.median(numeric_only=True))

    if short_items is not None:
        df = df.rename(columns=short_items)

    print(f"{datapath} | shape:", df.shape)
    return df


def split_by_class(df, class_col):
    """
    Split a dataframe into positive (class 1) and negative (class 0) cohorts.

    Arguments:
        df (pd.DataFrame): Cleaned feature matrix including the class column.
        class_col (str):   Name of the binary class column (0/1).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_class1, df_class0), each
        without the class column.
    """
    df_1 = df[df[class_col] == 1].drop(columns=class_col)
    df_0 = df[df[class_col] == 0].drop(columns=class_col)
    print(f"class 1: {df_1.shape} | class 0: {df_0.shape}")
    return df_1, df_0


def get_u(df):
    x = np.array(df, dtype=float)
    u = x.copy()

    for i in range(len(df.columns)):
        dist = best_fit_distribution(x[:, i])
        u[:, i] = dist[0].cdf(u[:, i], *dist[1])
    return u


def fit_and_save_vine(u, cohort_name, out_dir="vine_outputs",
                      cops=None, vine=None, extra_meta=None):
    """
    Fit a vine copula model to uniform data and persist the result to disk.

    The fitted model (structure matrix M, parameter matrix P, copula-family
    matrix C) is bundled together with metadata and serialised as a pickle
    file named `vine_{cohort_name}_{vine}.pkl` inside `out_dir`.

    Arguments:
        u (np.ndarray):       Uniform pseudo-observations, shape (n, d).
        cohort_name (str):    Label for this run (used in the filename).
        out_dir (str):        Directory where the pickle file is saved.
                              Created automatically if it does not exist.
        cops (list[int] | None): Candidate copula family IDs to consider.
                              Defaults to all 15 families (1-15).
        vine (str | None):    Vine type passed to `fit_vinecop`
                              (e.g. 'C' for C-vine, 'R' for R-vine, 'D' for D-vine).
        extra_meta (dict | None): Additional key-value pairs merged into
                              the saved result dict (e.g. datapath).

    Returns:
        tuple: (M, P, C, save_path)
    """
    if cops is None:
        cops = list(range(1, 16))  # use all 15 candidate families by default

    os.makedirs(out_dir, exist_ok=True)

    # Fit the vine copula; returns structure (M), parameters (P), families (C)
    M, P, C = fit_vinecop(u, cops, vine=vine)

    # Bundle model arrays with provenance metadata for reproducibility
    result = {
        "cohort": cohort_name,
        "vine": vine,
        "copulas": cops,
        "M": M,
        "P": P,
        "C": C,
        "n": u.shape[0],  # number of observations
        "d": u.shape[1],  # number of dimensions / variables
    }
    if extra_meta:
        result.update(extra_meta)

    save_path = os.path.join(out_dir, f"vine_{cohort_name}_{vine}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved: {save_path}")
    return M, P, C, save_path


def run_dataset(datapath, class_col=None, dataset_tag="dataset", out_dir="vine_outputs",
                vine=None, cops=None, short_items=None):
    """
    End-to-end pipeline: load data, transform to uniforms, fit vine copula(s),
    and save the results.

    Behaviour depends on `class_col`:
      - class_col is None:
            Treat the entire dataset as a single cohort. Fit one vine copula
            and return a dict with key 'combined'.
      - class_col is given:
            Split the dataset into two cohorts (class 1 and class 0). Fit a
            separate vine copula for each and return a dict with keys
            'class1' and 'class0'.

    Arguments:
        datapath (str):       Path to the input CSV file.
        class_col (str | None): Column name used for binary splitting. When
                              None, the dataset is treated as a single cohort.
        dataset_tag (str):    Short identifier for this run; used in filenames.
        out_dir (str):        Directory for saved vine pickle files.
        vine (str | None):    Vine type ('C', 'R', etc.).
        cops (list[int] | None): Candidate copula family IDs.
        short_items (dict | None): Optional column-rename mapping.

    Returns:
        dict: Combined path: {'combined': {'M', 'P', 'C', 'file'}}.
              Split path:    {'class1': {...}, 'class0': {...}}.
    """
    if class_col is None:
        # ── Combined cohort path ────────────────────────────────────────────
        df = load_dataset(datapath, short_items=short_items)
        u = get_u(df)

        M, P, C, p = fit_and_save_vine(
            u, cohort_name=dataset_tag,
            out_dir=out_dir, vine=vine, cops=cops,
            extra_meta={"datapath": datapath})

        return {"combined": {"M": M, "P": P, "C": C, "file": p}}

    # ── Split-cohort path (class 1 vs class 0) ──────────────────────────────
    df = load_dataset(datapath, short_items=short_items)
    df_1, df_0 = split_by_class(df, class_col)

    u_1 = get_u(df_1)
    u_0 = get_u(df_0)

    M1, P1, C1, p1 = fit_and_save_vine(
        u_1, cohort_name=f"{dataset_tag}_class1",
        out_dir=out_dir, vine=vine, cops=cops,
        extra_meta={"datapath": datapath, "class_value": 1})
    M0, P0, C0, p0 = fit_and_save_vine(
        u_0, cohort_name=f"{dataset_tag}_class0",
        out_dir=out_dir, vine=vine, cops=cops,
        extra_meta={"datapath": datapath, "class_value": 0})

    return {
        "class1": {"M": M1, "P": P1, "C": C1, "file": p1},
        "class0": {"M": M0, "P": P0, "C": C0, "file": p0},
    }


def print_last_vine_tree(a, p, c):
    """
    Print a summary of the deepest (last) tree in a fitted vine copula.

    Arguments:
        a (np.ndarray): Triangular vine structure matrix (d x d).
        p (np.ndarray): Parameter matrix (d-1 x d-1).
        c (np.ndarray): Copula-family index matrix (d-1 x d-1).
    """
    dimen = a.shape[0]
    i = dimen - 2

    print(f"** Tree:  {i + 1}")
    for k in range(dimen - 1 - i):
        ak = a[:, k]
        akn = [int(ak[-1 - k]), int(ak[i])]

        conditioning_set = list((ak.astype(int)[:i])[::-1])
        cond_str = ",".join(map(str, conditioning_set))
        node_str = f"{akn[0]},{akn[1]} | {cond_str}"

        cop_id = c[i, k]
        name = copula_names.get(cop_id, f"Copula_{cop_id}")
        params = p[i, k]

        print(f"{node_str: <15} --->  {name} : parameters =  {params}")


def print_vine_structure(a, p, c):
    """
    Print a human-readable summary of the complete vine copula structure.

    Arguments:
        a (np.ndarray): Vine structure matrix (d x d).
        p (np.ndarray): Copula parameter matrix (d-1 x d-1).
        c (np.ndarray): Copula family index matrix (d-1 x d-1).
    """
    dimen = a.shape[0]  # number of variables
    for i in range(dimen - 1):  # iterate over each tree level
        print(f"** Tree:  {i + 1}")
        for k in range(dimen - 1 - i):  # iterate over edges in this tree
            ak = a[:, k]  # column k encodes the variable ordering for edge k
            akn = [int(ak[-1 - k]), int(ak[i])]  # conditioned variable pair

            if i == 0:
                # Tree 1: unconditional bivariate copula, no conditioning set
                node_str = f"{akn[0]},{akn[1]}"
            else:
                # Higher trees: include the conditioning set
                conditioning_set = list((ak.astype(int)[:i])[::-1])
                cond_str = ",".join(map(str, conditioning_set))
                node_str = f"{akn[0]},{akn[1]} | {cond_str}"

            cop_id = c[i, k]
            name = copula_names.get(cop_id, f"Copula_{cop_id}")
            params = p[i, k]
            print(f"{node_str: <15} --->  {name} : parameters =  {params}")
        print("")  # blank line between trees for readability


def plotvine_selected(a, plottitle=None, variables=None, savepath=None, tree_to_plot=None):
    """
    Plot the vine structure.

    Arguments:
        a (np.ndarray): The vine tree structure provided as a triangular matrix.
        plottitle (str | None): Title of the plot.
        variables (list | None): List of variable names.
        savepath (str | None):   Path to save the plot.
        tree_to_plot (int | None): The specific tree number to plot (starts at 1).
                                   If None, plots all trees.
    """
    dimen = a.shape[0]
    order = pd.DataFrame(columns=["node", "l", "r", "tree"])

    s = 0
    for i in list(range(dimen - 1)):
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)
            if i == 0:
                single_row_values = {
                    "node": list(akn),
                    "l": akn[0],
                    "r": akn[1],
                    "tree": i,
                }
            else:
                single_row_values = {
                    "node": list(akn) + ["|"] + list((ak.astype(int)[:i])[::-1]),
                    "l": list(akn),
                    "r": list((ak.astype(int)[:i])[::-1]),
                    "tree": i,
                }

            order.loc[s] = single_row_values
            s = s + 1

    for t in list(range(dimen - 1)):
        orderk = order[order.tree == t].reset_index(drop=True)
        if t == 0:
            orderk["v1"] = orderk.l
            orderk["v2"] = orderk.r
            for j in range(len(orderk)):
                int(orderk.v1[j])
                int(orderk.v2[j])
            locals()["order" + str(t + 1)] = orderk
        else:
            v1k = []
            v2k = []
            for j in range(len(orderk)):
                orderk2 = order[order.tree == t - 1].reset_index(drop=True)
                l = orderk.l[j] + orderk.r[j]
                subnodes = []
                for k in range(len(orderk2)):
                    subnodes.append(sum(1 for item in orderk2.node[k] if item in l))
                subnodes = np.array(subnodes) == len(l) - 1
                orderk2 = orderk2[subnodes].reset_index(drop=True)
                v1k.append(orderk2.node[0])
                v2k.append(orderk2.node[1])
            orderk["v1"] = v1k
            orderk["v2"] = v2k
            locals()["order" + str(t + 1)] = orderk

    n = dimen - 1

    if tree_to_plot is not None:
        fig, ax = plt.subplots(figsize=((dimen - 1) * 2, (dimen - 1) * 2))
        plot_iterator = [(tree_to_plot, ax)]
    else:
        fig, axes = plt.subplots(dimen - 1, 1, figsize=(n * 2, n * 3))
        plot_iterator = zip(range(1, dimen), axes.flat)

    leg_labels = {}
    if variables is not None:
        for i in range(len(variables)):
            leg_labels.update({i: variables[i]})

    for t, ax in plot_iterator:
        if t == 1:
            orderk = locals()["order" + str(t)]
            edges = list(orderk.node)
            edges = [tuple(sublist) for sublist in edges]
            edge_labels = {
                edge: ",".join(map(str, orderk.node[i])) for i, edge in enumerate(edges)
            }
        elif t == 2:
            orderk = locals()["order" + str(t)]
            edges = [
                (",".join(map(str, orderk.v1[i])), ",".join(map(str, orderk.v2[i])))
                for i in range(len(orderk))
            ]
            edge_labels = {
                edge: ",".join(map(str, orderk.node[i])).replace(",|,", "|")
                for i, edge in enumerate(edges)
            }
        else:
            orderk = locals()["order" + str(t)]
            edges = [
                (
                    ",".join(map(str, orderk.v1[i])).replace(",|,", "|"),
                    ",".join(map(str, orderk.v2[i])).replace(",|,", "|"),
                )
                for i in range(len(orderk))
            ]
            edge_labels = {
                edge: ",".join(map(str, orderk.node[i])).replace(",|,", "|")
                for i, edge in enumerate(edges)
            }

        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)

        try:
            len(edges[0][0]) * 400
        except TypeError:
            len(edges[0]) * 200

        nx.draw_networkx_labels(
            G, pos, ax=ax, bbox=dict(facecolor="skyblue"), font_size=13
        )
        nx.draw_networkx_edges(G, pos, edge_color="black", ax=ax)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_color="black", ax=ax, font_size=12
        )

        ax.axis("off")
        ax.set_title(f"Tree {t}")

    if plottitle is not None:
        fig.suptitle(plottitle, fontsize=16, y=0.95)
    if variables is not None:
        plt.text(
            0.9,
            0.5,
            "\n".join([f"{key} : {value}" for key, value in leg_labels.items()]),
            transform=plt.gcf().transFigure,
            fontsize=15,
            verticalalignment="center",
        )

    if savepath is not None:
        plt.savefig(savepath, dpi=500, bbox_inches="tight")
    plt.show()


def load_vine(pkl_path):
    """
    Load a saved vine model pickle and return its (M, P, C) arrays.

    Arguments:
        pkl_path (str): Path to a pickle saved by `fit_and_save_vine`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (M, P, C).
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["M"], data["P"], data["C"]



if __name__ == "__main__":
    short_items = None    # Optional mapping of long column names to short display names.

    out_dir = "vine_outputs"
    vine_curve = "R"
    cop_4 = [1, 6, 10, 15]  # Gaussian, Clayton0, Frank, Student
    dataset_tag = "combined"
    datapath = "hf_icd_combined.csv"

    # --- Fit a vine copula on the whole dataset (single cohort) -------------
    res = run_dataset(
        datapath,
        class_col=None,            # set to a column name to split into class1/class0
        dataset_tag=dataset_tag,
        out_dir=out_dir,
        vine=vine_curve,
        cops=cop_4,
        short_items=short_items,
    )

    # --- Reload the saved model and visualise / summarise it ----------------
    pkl_path = res["combined"]["file"]
    M, P, C = load_vine(pkl_path)

    df = load_dataset(datapath, short_items=short_items)

    num_tree = 1
    plotvine_selected(
        M,
        plottitle=f"{vine_curve}-Vine",
        variables=list(df.columns),
        savepath=os.path.join(out_dir, f"{dataset_tag}_{vine_curve}_tree{num_tree}.pdf"),
        tree_to_plot=num_tree,
    ) # only plot the num_tree tree

    # print_vine_structure(M, P, C)  ##print the whole tree structure
    # print_last_vine_tree(M, P, C)  ##only print the last tree structure
