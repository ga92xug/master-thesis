#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ax.utils.stats.statstools import marginal_effects
from plotly import subplots
from ax.core.observation import ObservationFeatures
from ax.modelbridge.base import ModelBridge
from ax.plot.base import AxPlotConfig, AxPlotTypes, PlotData, DECIMALS
from ax.plot.color import BLUE_SCALE, GREEN_PINK_SCALE, GREEN_SCALE
from ax.plot.helper import (
    axis_range,
    contour_config_to_trace,
    get_fixed_values,
    get_grid_for_parameter,
    get_plot_data,
    get_range_parameter,
    get_range_parameters,
    relativize_data,
    rgb,
    TNullableGeneratorRunsDict,
)

from ax.plot.contour import _get_contour_predictions, short_name


def interact_contour_plotly(
    model: ModelBridge,
    metric_name: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    relative: bool = False,
    density: int = 50,
    slice_values: Optional[Dict[str, Any]] = None,
    lower_is_better: bool = False,
    fixed_features: Optional[ObservationFeatures] = None,
    trial_index: Optional[int] = None,
) -> go.Figure:
    """Create interactive plot with predictions for a 2-d slice of the parameter
    space.

    Args:
        model: ModelBridge that contains model for predictions
        metric_name: Name of metric to plot
        generator_runs_dict: A dictionary {name: generator run} of generator runs
            whose arms will be plotted, if they lie in the slice.
        relative: Predictions relative to status quo
        density: Number of points along slice to evaluate predictions.
        slice_values: A dictionary {name: val} for the fixed values of the
            other parameters. If not provided, then the status quo values will
            be used if there is a status quo, otherwise the mean of numeric
            parameters or the mode of choice parameters.
        lower_is_better: Lower values for metric are better.
        fixed_features: An ObservationFeatures object containing the values of
            features (including non-parameter features like context) to be set
            in the slice.

    Returns:
        go.Figure: interactive plot of objective vs. parameters
    """

    # NOTE: This implements a hack to allow Plotly to specify two parameters
    # simultaneously. It is not possible within Plotly to specify a third,
    # so `metric_name` must be specified and cannot be selected via dropdown
    # by the user.

    if trial_index is not None:
        if slice_values is None:
            slice_values = {}
        slice_values["TRIAL_PARAM"] = str(trial_index)

    range_parameters = get_range_parameters(model, min_num_values=2)
    plot_data, _, _ = get_plot_data(
        model, generator_runs_dict or {}, {metric_name}, fixed_features=fixed_features
    )
    #print("plot_data", type(plot_data), plot_data)
    #for arm_name, plotInSampleArm in plot_data.in_sample.items():
    #    mapper = [0,1,2,4,8,16]
    #    plotInSampleArm.parameters["0_group"] = mapper[plotInSampleArm.parameters["0_group"]]
    #print("plot_data", type(plot_data), plot_data)

    # TODO T38563759: Sort parameters by feature importances
    param_names = [parameter.name for parameter in range_parameters]

    is_log_dict: Dict[str, bool] = {}
    grid_dict: Dict[str, np.ndarray] = {}
    for parameter in range_parameters:
        is_log_dict[parameter.name] = parameter.log_scale
        grid_dict[parameter.name] = get_grid_for_parameter(parameter, density)

    # Populate `f_dict` (the predicted expectation value of `metric_name`) and
    # `sd_dict` (the predicted SEM), each of which represents a 2D array of plots
    # where each parameter can be assigned to each of the x or y axes.

    # pyre-fixme[9]: f_dict has type `Dict[str, Dict[str, np.ndarray]]`; used as
    #  `Dict[str, Dict[str, typing.List[Variable[_T]]]]`.
    f_dict: Dict[str, Dict[str, np.ndarray]] = {
        param1: {param2: [] for param2 in param_names} for param1 in param_names
    }
    # pyre-fixme[9]: sd_dict has type `Dict[str, Dict[str, np.ndarray]]`; used as
    #  `Dict[str, Dict[str, typing.List[Variable[_T]]]]`.
    sd_dict: Dict[str, Dict[str, np.ndarray]] = {
        param1: {param2: [] for param2 in param_names} for param1 in param_names
    }

    for param1 in param_names:
        for param2 in param_names:
            _, f_plt, sd_plt, _, _, _ = _get_contour_predictions(
                model=model,
                x_param_name=param1,
                y_param_name=param2,
                metric=metric_name,
                generator_runs_dict=generator_runs_dict,
                density=density,
                slice_values=slice_values,
                fixed_features=fixed_features,
            )
            # print("f_plt", f_plt)
            f_dict[param1][param2] = f_plt
            sd_dict[param1][param2] = sd_plt

    # Set plotting defaults for all subplots

    config = {
        "arm_data": plot_data,
        "blue_scale": BLUE_SCALE,
        "density": density,
        "f_dict": f_dict,
        "green_scale": GREEN_SCALE,
        "green_pink_scale": GREEN_PINK_SCALE,
        "grid_dict": grid_dict,
        "lower_is_better": lower_is_better,
        "metric": metric_name,
        "rel": relative,
        "sd_dict": sd_dict,
        "is_log_dict": is_log_dict,
        "param_names": param_names,
    }

    config = AxPlotConfig(config, plot_type=AxPlotTypes.GENERIC).data

    arm_data = config["arm_data"]
    density = config["density"]
    grid_dict = config["grid_dict"]
    f_dict = config["f_dict"]
    lower_is_better = config["lower_is_better"]
    metric = config["metric"]
    rel = config["rel"]
    sd_dict = config["sd_dict"]
    is_log_dict = config["is_log_dict"]
    param_names = config["param_names"]

    green_scale = config["green_scale"]
    green_pink_scale = config["green_pink_scale"]
    blue_scale = config["blue_scale"]

    CONTOUR_CONFIG = {
        "autocolorscale": False,
        "autocontour": True,
        "contours": {"coloring": "heatmap"},
        "hoverinfo": "x+y+z",
        "ncontours": int(density / 2),
        "type": "contour",
    }

    if rel:
        f_scale = reversed(green_pink_scale) if lower_is_better else green_pink_scale
    else:
        f_scale = green_scale

    f_contour_trace_base = {
        "colorbar": {
            "len": 0.875,
            "x": 0.45,
            "y": 0.5,
            "ticksuffix": "%" if rel else "",
            "tickfont": {"size": 8},
        },
        "colorscale": [(i / (len(f_scale) - 1), rgb(v)) for i, v in enumerate(f_scale)],
        "xaxis": "x",
        "yaxis": "y",
        # zmax and zmin are ignored if zauto is true
        "zauto": not rel,
    }

    sd_contour_trace_base = {
        "colorbar": {
            "len": 0.875,
            "x": 1,
            "y": 0.5,
            "ticksuffix": "%" if rel else "",
            "tickfont": {"size": 8},
        },
        "colorscale": [
            (i / (len(blue_scale) - 1), rgb(v)) for i, v in enumerate(blue_scale)
        ],
        "xaxis": "x2",
        "yaxis": "y2",
    }

    # pyre-fixme[6]: Expected `Mapping[str, typing.Union[Dict[str,
    #  typing.Union[Dict[str, int], float, str]], typing.List[Tuple[float, str]], bool,
    #  str]]` for 1st param but got `Dict[str, typing.Union[Dict[str, str], int,
    #  str]]`.
    f_contour_trace_base.update(CONTOUR_CONFIG)
    # pyre-fixme[6]: Expected `Mapping[str, typing.Union[Dict[str,
    #  typing.Union[Dict[str, int], float, str]], typing.List[Tuple[float, str]],
    #  str]]` for 1st param but got `Dict[str, typing.Union[Dict[str, str], int,
    #  str]]`.
    sd_contour_trace_base.update(CONTOUR_CONFIG)

    # Format and add hovertext to contour plots.

    insample_param_values = {}
    for param_name in param_names:
        insample_param_values[param_name] = []
        for arm_name in arm_data["in_sample"].keys():
            insample_param_values[param_name].append(
                arm_data["in_sample"][arm_name]["parameters"][param_name]
            )

    insample_arm_text = []
    for arm_name in arm_data["in_sample"].keys():
        atext = f"Arm {arm_name}"
        params = arm_data["in_sample"][arm_name]["parameters"]
        ys = arm_data["in_sample"][arm_name]["y"]
        ses = arm_data["in_sample"][arm_name]["se"]
        for yname in ys.keys():
            sem_str = f"{ses[yname]}" if ses[yname] is None else f"{ses[yname]:.6g}"
            y_str = f"{ys[yname]}" if ys[yname] is None else f"{ys[yname]:.6g}"
            atext += f"<br>{yname}: {y_str} (SEM: {sem_str})"
        for pname in params.keys():
            pval = params[pname]
            pstr = f"{pval:.6g}" if isinstance(pval, float) else f"{pval}"
            atext += f"<br>{pname}: {pstr}"
        insample_arm_text.append(atext)

    out_of_sample_param_values = {}
    for param_name in param_names:
        out_of_sample_param_values[param_name] = {}
        for generator_run_name in arm_data["out_of_sample"].keys():
            out_of_sample_param_values[param_name][generator_run_name] = []
            for arm_name in arm_data["out_of_sample"][generator_run_name].keys():
                out_of_sample_param_values[param_name][generator_run_name].append(
                    arm_data["out_of_sample"][generator_run_name][arm_name][
                        "parameters"
                    ][param_name]
                )

    out_of_sample_arm_text = {}
    for generator_run_name in arm_data["out_of_sample"].keys():
        out_of_sample_arm_text[generator_run_name] = [
            "<em>Candidate " + arm_name + "</em>"
            for arm_name in arm_data["out_of_sample"][generator_run_name].keys()
        ]

    # Populate `xbuttons`, which allows the user to select 1D slices of `f_dict` and
    # `sd_dict`, corresponding to all plots that have a certain parameter on the x-axis.

    # Number of traces for each pair of parameters
    trace_cnt = 4 + (len(arm_data["out_of_sample"]) * 2)

    xbuttons = []
    ybuttons = []

    for xvar in param_names:
        xbutton_data_args = {"x": [], "y": [], "z": []}
        for yvar in param_names:
            res = relativize_data(
                f_dict[xvar][yvar], sd_dict[xvar][yvar], rel, arm_data, metric
            )
            f_final = res[0]
            sd_final = res[1]
            # transform to nested array
            f_plt = []
            for ind in range(0, len(f_final), density):
                f_plt.append(f_final[ind : ind + density])
            sd_plt = []
            for ind in range(0, len(sd_final), density):
                sd_plt.append(sd_final[ind : ind + density])

            # grid + in-sample
            xbutton_data_args["x"] += [
                grid_dict[xvar],
                grid_dict[xvar],
                insample_param_values[xvar],
                insample_param_values[xvar],
            ]
            xbutton_data_args["y"] += [
                grid_dict[yvar],
                grid_dict[yvar],
                insample_param_values[yvar],
                insample_param_values[yvar],
            ]
            xbutton_data_args["z"] += [f_plt, sd_plt, [], []]

            for generator_run_name in out_of_sample_param_values[xvar]:
                generator_run_x_vals = out_of_sample_param_values[xvar][
                    generator_run_name
                ]
                xbutton_data_args["x"] += [generator_run_x_vals] * 2
            for generator_run_name in out_of_sample_param_values[yvar]:
                generator_run_y_vals = out_of_sample_param_values[yvar][
                    generator_run_name
                ]
                xbutton_data_args["y"] += [generator_run_y_vals] * 2
                xbutton_data_args["z"] += [[]] * 2

        xbutton_args = [
            xbutton_data_args,
            {
                "xaxis.title": short_name(xvar),
                "xaxis2.title": short_name(xvar),
                "xaxis.range": axis_range(grid_dict[xvar], is_log_dict[xvar]),
                "xaxis2.range": axis_range(grid_dict[xvar], is_log_dict[xvar]),
                "xaxis.type": "log" if is_log_dict[xvar] else "linear",
                "xaxis2.type": "log" if is_log_dict[xvar] else "linear",
            },
        ]
        xbuttons.append({"args": xbutton_args, "label": xvar, "method": "update"})

    # Populate `ybuttons`, which uses the `visible` arg to mask the 1D slice of plots
    # produced by `xbuttons`, down to a single plot, so that only one element `f_dict`
    # and `sd_dict` remain.

    # No y button for first param so initial value is sane
    for y_idx in range(1, len(param_names)):
        visible = [False] * (len(param_names) * trace_cnt)
        for i in range(y_idx * trace_cnt, (y_idx + 1) * trace_cnt):
            visible[i] = True
        y_param = param_names[y_idx]
        ybuttons.append(
            {
                "args": [
                    {"visible": visible},
                    {
                        "yaxis.title": short_name(y_param),
                        "yaxis.range": axis_range(
                            grid_dict[y_param], is_log_dict[y_param]
                        ),
                        "yaxis2.range": axis_range(
                            grid_dict[y_param], is_log_dict[y_param]
                        ),
                        "yaxis.type": "log" if is_log_dict[y_param] else "linear",
                        "yaxis2.type": "log" if is_log_dict[y_param] else "linear",
                    },
                ],
                "label": param_names[y_idx],
                "method": "update",
            }
        )

    # calculate max of abs(outcome), used for colorscale
    # TODO(T37079623) Make this work for relative outcomes
    # let f_absmax = Math.max(Math.abs(Math.min(...f_final)), Math.max(...f_final))

    traces = []
    xvar = param_names[0]
    base_in_sample_arm_config = None

    # start symbol at 2 for out-of-sample candidate markers
    i = 2

    for yvar_idx, yvar in enumerate(param_names):
        cur_visible = yvar_idx == 1
        f_start = xbuttons[0]["args"][0]["z"][trace_cnt * yvar_idx]
        sd_start = xbuttons[0]["args"][0]["z"][trace_cnt * yvar_idx + 1]

        # create traces
        f_trace = {
            "x": grid_dict[xvar],
            "y": grid_dict[yvar],
            "z": f_start,
            "visible": cur_visible,
        }

        for key in f_contour_trace_base.keys():
            f_trace[key] = f_contour_trace_base[key]

        sd_trace = {
            "x": grid_dict[xvar],
            "y": grid_dict[yvar],
            "z": sd_start,
            "visible": cur_visible,
        }

        for key in sd_contour_trace_base.keys():
            sd_trace[key] = sd_contour_trace_base[key]

        f_in_sample_arm_trace = {"xaxis": "x", "yaxis": "y"}

        sd_in_sample_arm_trace = {"showlegend": False, "xaxis": "x2", "yaxis": "y2"}
        base_in_sample_arm_config = {
            "hoverinfo": "text",
            "legendgroup": "In-sample",
            "marker": {"color": "black", "symbol": 1, "opacity": 0.5},
            "mode": "markers",
            "name": "In-sample",
            "text": insample_arm_text,
            "type": "scatter",
            "visible": cur_visible,
            "x": insample_param_values[xvar],
            "y": insample_param_values[yvar],
        }

        for key in base_in_sample_arm_config.keys():
            f_in_sample_arm_trace[key] = base_in_sample_arm_config[key]
            sd_in_sample_arm_trace[key] = base_in_sample_arm_config[key]

        traces += [f_trace, sd_trace, f_in_sample_arm_trace, sd_in_sample_arm_trace]

        # iterate over out-of-sample arms
        for generator_run_name in arm_data["out_of_sample"].keys():
            traces.append(
                {
                    "hoverinfo": "text",
                    "legendgroup": generator_run_name,
                    "marker": {"color": "black", "symbol": i, "opacity": 0.5},
                    "mode": "markers",
                    "name": generator_run_name,
                    "text": out_of_sample_arm_text[generator_run_name],
                    "type": "scatter",
                    "xaxis": "x",
                    "x": out_of_sample_param_values[xvar][generator_run_name],
                    "yaxis": "y",
                    "y": out_of_sample_param_values[yvar][generator_run_name],
                    "visible": cur_visible,
                }
            )
            traces.append(
                {
                    "hoverinfo": "text",
                    "legendgroup": generator_run_name,
                    "marker": {"color": "black", "symbol": i, "opacity": 0.5},
                    "mode": "markers",
                    "name": "In-sample",
                    "showlegend": False,
                    "text": out_of_sample_arm_text[generator_run_name],
                    "type": "scatter",
                    "x": out_of_sample_param_values[xvar][generator_run_name],
                    "xaxis": "x2",
                    "y": out_of_sample_param_values[yvar][generator_run_name],
                    "yaxis": "y2",
                    "visible": cur_visible,
                }
            )
            i += 1

    # Initially visible yvar
    yvar = param_names[1]

    xrange = axis_range(grid_dict[xvar], is_log_dict[xvar])
    yrange = axis_range(grid_dict[yvar], is_log_dict[yvar])

    xtype = "log" if is_log_dict[xvar] else "linear"
    ytype = "log" if is_log_dict[yvar] else "linear"

    layout = {
        "title": metric_name,
        "annotations": [
            {
                "font": {"size": 14},
                "showarrow": False,
                "text": "Mean",
                "x": 0.25,
                "xanchor": "center",
                "xref": "paper",
                "y": 1,
                "yanchor": "bottom",
                "yref": "paper",
            },
            {
                "font": {"size": 14},
                "showarrow": False,
                "text": "Standard Error",
                "x": 0.8,
                "xanchor": "center",
                "xref": "paper",
                "y": 1,
                "yanchor": "bottom",
                "yref": "paper",
            },
            {
                "x": 0.26,
                "y": -0.26,
                "xref": "paper",
                "yref": "paper",
                "text": "x-param:",
                "showarrow": False,
                "yanchor": "top",
                "xanchor": "left",
            },
            {
                "x": 0.26,
                "y": -0.4,
                "xref": "paper",
                "yref": "paper",
                "text": "y-param:",
                "showarrow": False,
                "yanchor": "top",
                "xanchor": "left",
            },
        ],
        "updatemenus": [
            {
                "x": 0.35,
                "y": -0.29,
                "buttons": xbuttons,
                "xanchor": "left",
                "yanchor": "middle",
                "direction": "up",
            },
            {
                "x": 0.35,
                "y": -0.43,
                "buttons": ybuttons,
                "xanchor": "left",
                "yanchor": "middle",
                "direction": "up",
            },
        ],
        "autosize": False,
        "height": 450,
        "hovermode": "closest",
        "legend": {"orientation": "v", "x": 0, "y": -0.2, "yanchor": "top"},
        "margin": {"b": 100, "l": 35, "pad": 0, "r": 35, "t": 35},
        "width": 950,
        "xaxis": {
            "anchor": "y",
            "autorange": False,
            "domain": [0.05, 0.45],
            "exponentformat": "e",
            "range": xrange,
            "tickfont": {"size": 11},
            "tickmode": "auto",
            "title": short_name(xvar),
            "type": xtype,
        },
        "xaxis2": {
            "anchor": "y2",
            "autorange": False,
            "domain": [0.6, 1],
            "exponentformat": "e",
            "range": xrange,
            "tickfont": {"size": 11},
            "tickmode": "auto",
            "title": short_name(xvar),
            "type": xtype,
        },
        "yaxis": {
            "anchor": "x",
            "autorange": False,
            "domain": [0, 1],
            "exponentformat": "e",
            "range": yrange,
            "tickfont": {"size": 11},
            "tickmode": "auto",
            "title": short_name(yvar),
            "type": ytype,
        },
        "yaxis2": {
            "anchor": "x2",
            "autorange": False,
            "domain": [0, 1],
            "exponentformat": "e",
            "range": yrange,
            "tickfont": {"size": 11},
            "tickmode": "auto",
            "type": ytype,
        },
    }

    return go.Figure(data=traces, layout=layout)
    # return AxPlotConfig(config, plot_type=AxPlotTypes.INTERACT_CONTOUR)

def plot_marginal_effects(model: ModelBridge, metric: str, split_factor: int = 4):
    """
    Calculates and plots the marginal effects -- the effect of changing one
    factor away from the randomized distribution of the experiment and fixing it
    at a particular level.

    Args:
        model: Model to use for estimating effects
        metric: The metric for which to plot marginal effects.

    Returns:
        AxPlotConfig of the marginal effects
    """
    plot_data, _, _ = get_plot_data(model, {}, {metric})

    arm_dfs = []
    for arm in plot_data.in_sample.values():
        arm_df = pd.DataFrame(arm.parameters, index=[arm.name])
        arm_df["mean"] = arm.y_hat[metric]
        arm_df["sem"] = arm.se_hat[metric]
        arm_dfs.append(arm_df)
    effect_table = marginal_effects(pd.concat(arm_dfs, 0))

    varnames = effect_table["Name"].unique()
    #varnames = ['0_reflection' '0_group' '0_out_channels' '0_kernel_size' '1_reflection']
    # pyre-fixme[33]: Given annotation cannot contain `Any`.
    figures = []
    for varnames_split in np.array_split(varnames, split_factor):
        data: List[Any] = []
        for varname in varnames_split:
            var_df = effect_table[effect_table["Name"] == varname]
            data += [
                go.Bar(
                    x=var_df["Level"],
                    y=var_df["Beta"],
                    error_y={"type": "data", "array": var_df["SE"]},
                    name=varname,
                )
            ]
        fig = subplots.make_subplots(
            cols=len(varnames_split),
            rows=1,
            subplot_titles=list(varnames_split),
            print_grid=False,
            shared_yaxes=True,
        )
        for idx, item in enumerate(data):
            fig.append_trace(item, 1, idx + 1)
        fig.layout.showlegend = False
        # fig.layout.margin = go.layout.Margin(l=2, r=2)
        fig.layout.title = "Marginal Effects by Factor"
        fig.layout.yaxis = {
            "title": "% higher than experiment average",
            "hoverformat": ".{}f".format(DECIMALS),
        }
        figures.append(fig)
    return figures


from ax.service.ax_client import AxClient
from ax.modelbridge.factory import get_MOO_NEHVI 
import torch
def evaluate(
        ax_client: AxClient,
        filepath: str = None,
):
    assert ax_client is not None or filepath is not None, "Either ax_client or filepath must be provided"
    # load ax client
    if ax_client is None:
        ax_client = AxClient.load_from_json_file(filepath=filepath)

    experiment = ax_client.experiment
    data = experiment.fetch_data()


    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model = get_MOO_NEHVI(
        experiment=experiment, 
        data=data,
        device=device
    )
    valid_acc_interact_contour_plotly = interact_contour_plotly(model, metric_name="valid_acc", lower_is_better=False, density=2)


if __name__ == "__main__":
    evaluate(
        ax_client=None,
        filepath="NAS/data/cifar10_3/ax_client.json"
        )