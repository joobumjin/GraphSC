####
### THIS CODE IS TAKEN FROM PYTORCH GEOMETRIC'S explanation.py
### THIS CODE WAS MODIFIED TO ALLOW FOR DYNAMIC MPL SIZING AND WANDB INTEGRATION
###
import copy
from typing import List, Optional

import wandb

import torch
from torch import Tensor

def visualize_feature_importance(
    explanation,
    path: Optional[str] = None,
    feat_labels: List[str] = None,
    top_k: Optional[int] = None,
    run = None
):
    r"""Creates a bar plot of the node feature importances by summing up
    the node mask across all nodes.

    Args:
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        feat_labels (List[str], optional): The labels of features.
            (default :obj:`None`)
        top_k (int, optional): Top k features to plot. If :obj:`None`
            plots all features. (default: :obj:`None`)
    """
    node_mask = explanation.get('node_mask')
    if node_mask is None:
        raise ValueError(f"The attribute 'node_mask' is not available "
                            f"in '{explanation.__class__.__name__}' "
                            f"(got {explanation.available_explanations})")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f"Cannot compute feature importance for "
                            f"object-level 'node_mask' "
                            f"(got shape {node_mask.size()})")
    
    assert feat_labels is not None

    score = node_mask.sum(dim=0)

    return _visualize_score(score, feat_labels, path, top_k, run)

def _visualize_score(
    score: torch.Tensor,
    labels: List[str],
    path: Optional[str] = None,
    top_k: Optional[int] = None,
    run = None
):
    import matplotlib.pyplot as plt
    import pandas as pd

    if len(labels) != score.numel():
        raise ValueError(f"The number of labels (got {len(labels)}) must "
                         f"match the number of scores (got {score.numel()})")

    score = score.cpu().numpy()

    # df = pd.DataFrame({'score': score}, index=labels)
    df = pd.DataFrame({'score': score, "Feature": labels})
    print(f"Vis Columns: {df.columns}")
    df = df.sort_values('score', ascending=False)
    df = df.round(decimals=3)

    if top_k is not None:
        df = df.head(top_k)
        title = f"Feature importance for top {len(df)} features"
    else:
        title = f"Feature importance for {len(df)} features"


    if run is not None:
        import plotly.express as px

        fig = px.bar(df, x = "score", y = "Feature", width=800, height=1500, title="Feature Importances")
        
        fig.update_layout(yaxis={"type":'category', 
                                "categoryorder": "total ascending"},
                          margin=dict(l=20, r=20, b=20, t=40))
        
        run.log({"feature importance": fig})

    if path is not None:
        ax = df.plot(
            kind='barh',
            figsize=(14, 16),
            ylabel='Feature label',
            xlim=[0, float(df['score'].max()) + 0.3],
            title=title,
            y="score",
            x="Feature",
            legend=False,
        )
        plt.gca().invert_yaxis()
        # ax.bar_label(container=ax.containers[0], label_type='edge')

        plt.tight_layout()
        plt.savefig(path)
    else:
        plt.show()

    plt.close()

    return df
