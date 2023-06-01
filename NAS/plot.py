#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objs as go
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.exceptions.core import UserInputError
from ax.plot.base import AxPlotConfig, AxPlotTypes, CI_OPACITY, DECIMALS
from ax.plot.color import COLORS, DISCRETE_COLOR_SCALE, rgba
from ax.plot.helper import _format_CI, _format_dict, extend_range
from ax.plot.pareto_utils import ParetoFrontierResults
from ax.utils.common.typeutils import checked_cast, not_none
from scipy.stats import norm

from ax.plot.pareto_frontier import _get_single_pareto_trace

DEFAULT_CI_LEVEL: float = 0.9
VALID_CONSTRAINT_OP_NAMES = {"GEQ", "LEQ"}


def plot_pareto_frontier(
    frontier: ParetoFrontierResults,
    CI_level: float = DEFAULT_CI_LEVEL,
    show_parameterization_on_hover: bool = True,
):
    """Plot a Pareto frontier from a ParetoFrontierResults object.

    Args:
        frontier (ParetoFrontierResults): The results of the Pareto frontier
            computation.
        CI_level (float, optional): The confidence level, i.e. 0.95 (95%)
        show_parameterization_on_hover (bool, optional): If True, show the
            parameterization of the points on the frontier on hover.

    Returns:
        AEPlotConfig: The resulting Plotly plot definition.

    """
    trace = _get_single_pareto_trace(
        frontier=frontier,
        CI_level=CI_level,
        show_parameterization_on_hover=show_parameterization_on_hover,
    )

    shapes = []
    primary_threshold = None
    secondary_threshold = None
    if frontier.objective_thresholds is not None:
        primary_threshold = frontier.objective_thresholds.get(
            frontier.primary_metric, None
        )
        secondary_threshold = frontier.objective_thresholds.get(
            frontier.secondary_metric, None
        )
    absolute_metrics = frontier.absolute_metrics
    rel_x = frontier.secondary_metric not in absolute_metrics
    rel_y = frontier.primary_metric not in absolute_metrics
    if primary_threshold is not None:
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "x0": 0.0,
                "x1": 1.0,
                "yref": "y",
                "y0": primary_threshold,
                "y1": primary_threshold,
                "line": {"color": rgba(COLORS.CORAL.value), "width": 3},
            }
        )
    if secondary_threshold is not None:
        shapes.append(
            {
                "type": "line",
                "yref": "paper",
                "y0": 0.0,
                "y1": 1.0,
                "xref": "x",
                "x0": secondary_threshold,
                "x1": secondary_threshold,
                "line": {"color": rgba(COLORS.CORAL.value), "width": 3},
            }
        )

    layout = go.Layout(
        title="Pareto Frontier",
        xaxis={
            "title": frontier.secondary_metric,
            "ticksuffix": "%" if rel_x else "",
            "zeroline": True,
        },
        yaxis={
            "title": frontier.primary_metric,
            "ticksuffix": "%" if rel_y else "",
            "zeroline": True,
        },
        hovermode="closest",
        legend={"orientation": "h"},
        width=750,
        height=500,
        margin=go.layout.Margin(pad=4, l=225, b=75, t=75),  # noqa E741
        shapes=shapes,
    )

    fig = go.Figure(data=[trace], layout=layout)
    return fig