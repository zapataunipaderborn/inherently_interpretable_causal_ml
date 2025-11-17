import re
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
import math


def round_numbers_in_string(s, decimals=4):
    """
    Round all floating-point numbers in a string to a specified number of decimal places.

    This function uses regular expressions to find all substrings that match floating-point numbers
    (e.g., "1.234") and rounds them to the given number of decimals.

    Args:
        s (str): The input string containing numbers to round.
        decimals (int, optional): Number of decimal places to round to. Defaults to 4.

    Returns:
        str: The input string with all floating-point numbers rounded.
    """
    def round_match(match):
        return f"{float(match.group()):.{decimals}f}"
    return re.sub(r'\d+\.\d+', round_match, s)


def weighted_absolute_percentage_error(y_true, y_pred):
    """
    Compute Weighted Absolute Percentage Error (WAPE).

    WAPE is a measure of forecast accuracy that calculates the total absolute error as a percentage
    of the total absolute true values. Formula: sum(|y_true - y_pred|) / sum(|y_true|) * 100.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: Weighted Absolute Percentage Error (in percent).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def evaluate_metrics(y_true, y_pred, normalize=False):
    """
    Evaluate regression metrics: RMSE, WAPE, and MAE.

    Computes Root Mean Squared Error (RMSE), Weighted Absolute Percentage Error (WAPE),
    and Mean Absolute Error (MAE) between true and predicted values. If normalize is True,
    scales the values using StandardScaler before computing metrics.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        normalize (bool, optional): If True, normalize y_true and y_pred using StandardScaler.
                                    Defaults to False.

    Returns:
        tuple: (rmse, wape, mae) where:
            - rmse (float): Root Mean Squared Error.
            - wape (float): Weighted Absolute Percentage Error (in percent).
            - mae (float): Mean Absolute Error.
    """
    if normalize:
        scaler = StandardScaler()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_true_scaled = scaler.fit_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_scaled = scaler.transform(y_pred.reshape(-1, 1)).flatten()
        y_true = y_true_scaled
        y_pred = y_pred_scaled
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = weighted_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, wape, mae


def train_causal_models(train_df, G, model_class, model_params=None, param_grid=None, cv=4):
    """
    Train causal models for each node in the causal graph using their parent nodes as features.

    This function iterates over each node in the graph G, and for nodes with parents, trains a model
    using the parent variables as predictors. If a param_grid is provided, it performs grid search
    with cross-validation to find the best hyperparameters.

    Args:
        train_df (pd.DataFrame): Training data with columns corresponding to graph nodes.
        G (nx.DiGraph): NetworkX directed graph representing the causal structure.
        model_class (class): The machine learning model class to instantiate (e.g., LinearRegression).
        model_params (dict, optional): Parameters to pass to the model_class constructor. Defaults to {}.
        param_grid (dict, optional): Parameter grid for GridSearchCV. If provided, performs hyperparameter tuning.
        cv (int, optional): Number of cross-validation folds for grid search. Defaults to 4.

    Returns:
        dict: A dictionary where keys are node names and values are trained model instances.
              Nodes without parents are skipped.
    """
    model_params = model_params or {}
    param_grid = param_grid or {}
    parents = {node: list(G.predecessors(node)) for node in G.nodes()}
    causal_models = {}
    for node in G.nodes():
        par = parents[node]
        if not par:
            continue
        X = train_df[par]
        y = train_df[node]
        if param_grid:
            base_model = model_class(**model_params)
            grid_search = GridSearchCV(
                base_model, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='neg_mean_absolute_error'
            )
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            causal_models[node] = best_model
        else:
            model = model_class(**model_params)
            model.fit(X, y)
            causal_models[node] = model
            
    return causal_models

def predict_causal(test_df, G, causal_models, what_if=False):
    """
    Predict values for each node in the causal graph using trained causal models.

    This function generates predictions for all nodes in the graph. If what_if is False, it uses
    observed values from test_df for parent nodes. If what_if is True, it performs a full graph
    prediction where predictions for upstream nodes are used as inputs for downstream nodes,
    simulating interventions.

    Args:
        test_df (pd.DataFrame): Test data with columns corresponding to graph nodes.
        G (nx.DiGraph): NetworkX directed graph representing the causal structure.
        causal_models (dict): Dictionary of trained models for each node (from train_causal_models).
        what_if (bool, optional): If True, predict the entire graph using previous predictions
                                  (for interventions). If False, use observed parent values. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with predicted values for all nodes, indexed like test_df.
    """
    # get nodes
    parents = {node: list(G.predecessors(node)) for node in G.nodes()}
    # get root nodes
    root_nodes = [n for n in G.nodes() if len(parents[n]) == 0]

    # Create dataset for predictions
    causal_preds = pd.DataFrame(index=test_df.index)
    
    if what_if:
        # For what-if analysis: predict the whole graph using previous predictions
        # For root nodes only take the original node values
        for rn in root_nodes:
            causal_preds[rn] = test_df[rn]
        # Predict other nodes in topological order
        for node in nx.topological_sort(G):
            if node in root_nodes:
                continue
            model = causal_models[node]
            X_pred = causal_preds[parents[node]]
            # store the new predictions in the node
            causal_preds[node] = model.predict(X_pred)
    else:
        # For test set evaluation: use observed values from test_df for parents
        for node in G.nodes():
            if node in root_nodes:
                causal_preds[node] = test_df[node]
            else:
                model = causal_models[node]
                X_pred = test_df[parents[node]]  # Use observed parents
                causal_preds[node] = model.predict(X_pred)
    
    return causal_preds


def plot_causal_graph(G, title="Causal Graph", figsize=(5, 5)):
    """
    Plot a directed causal graph with customized styling.

    This function visualizes a NetworkX directed graph representing a causal structure.
    It uses a spring layout with increased repulsion for better node separation, draws
    nodes with a light blue color, and renders edges as fine arrows with small arrowheads.

    Args:
        G (nx.DiGraph): The NetworkX directed graph to plot.
        title (str, optional): Title of the plot. Defaults to "Causal Graph".
        figsize (tuple, optional): Figure size as (width, height). Defaults to (5, 5).

    Returns:
        None: Displays the plot.
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

    # Layout: increase k for stronger repulsion between nodes
    pos = nx.spring_layout(G, k=6.0, iterations=400, seed=42)
    node_radius = 0.05  # approximate visual node radius

    # --- Draw nodes ---
    nx.draw_networkx_nodes(
        G, pos,
        node_color="#c6dcef",
        node_size=2200,
        edgecolors="black",
        linewidths=0.8,
        ax=ax,
    )

    # --- Draw labels ---
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    # --- Draw fine arrows ---
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        shrink = node_radius / dist
        x0s = x0 + dx * shrink
        y0s = y0 + dy * shrink
        x1s = x1 - dx * shrink
        y1s = y1 - dy * shrink

        ax.annotate(
            "",
            xy=(x1s, y1s),
            xytext=(x0s, y0s),
            arrowprops=dict(
                arrowstyle="simple,head_length=1.5,head_width=1.0,tail_width=0.2",
                color="#023047",
                lw=0.6,
            ),
        )

    plt.tight_layout()
    plt.show()

def create_causal_graph(var_names, causal_links):
    G = nx.DiGraph()
    G.add_nodes_from(var_names)
    for target, preds in causal_links.items():
        for p in preds:
            G.add_edge(p, target)
    return G



