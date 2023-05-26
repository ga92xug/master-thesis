from ax.service.utils.report_utils import exp_to_df
from ax.service.utils.report_utils import _pareto_frontier_scatter_2d_plotly


def evaluate(experiment, verbose=1):
    df = exp_to_df(experiment)
    if verbose >= 1:
        print(df)


    ######################################################################
    # We can also visualize the Pareto frontier of tradeoffs between the
    # validation accuracy and the number of model parameters.
    #
    # .. tip::
    #   Ax uses Plotly to produce interactive plots, which allow you to
    #   do things like zoom, crop, or hover in order to view details
    #   of components of the plot. Try it out, and take a look at the
    #   `visualization tutorial <https://ax.dev/tutorials/visualizations.html>`__
    #   if you'd like to learn more).
    #
    # The final optimization results are shown in the figure below where
    # the color corresponds to the iteration number for each trial.
    # We see that our method was able to successfully explore the
    # trade-offs and found both large models with high validation
    # accuracy as well as small models with comparatively lower
    # validation accuracy.
    #

    
    _pareto_frontier_scatter_2d_plotly(experiment)


"""
    ######################################################################
    # To better understand what our surrogate models have learned about
    # the black box objectives, we can take a look at the leave-one-out
    # cross validation results. Since our models are Gaussian Processes,
    # they not only provide point predictions but also uncertainty estimates
    # about these predictions. A good model means that the predicted means
    # (the points in the figure) are close to the 45 degree line and that the
    # confidence intervals cover the 45 degree line with the expected frequency
    # (here we use 95% confidence intervals, so we would expect them to contain
    # the true observation 95% of the time).
    #
    # As the figures below show, the model size (``num_params``) metric is
    # much easier to model than the validation accuracy (``val_acc``) metric.
    #

    from ax.modelbridge.cross_validation import compute_diagnostics, cross_validate
    from ax.plot.diagnostic import interact_cross_validation_plotly
    from ax.utils.notebook.plotting import init_notebook_plotting, render

    cv = cross_validate(model=gs.model)  # The surrogate model is stored on the ``GenerationStrategy``
    compute_diagnostics(cv)

    interact_cross_validation_plotly(cv)


    ######################################################################
    # We can also make contour plots to better understand how the different
    # objectives depend on two of the input parameters. In the figure below,
    # we show the validation accuracy predicted by the model as a function
    # of the two hidden sizes. The validation accuracy clearly increases
    # as the hidden sizes increase.
    #

    from ax.plot.contour import interact_contour_plotly

    interact_contour_plotly(model=gs.model, metric_name="val_acc")


    ######################################################################
    # Similarly, we show the number of model parameters as a function of
    # the hidden sizes in the figure below and see that it also increases
    # as a function of the hidden sizes (the dependency on ``hidden_size_1``
    # is much larger).

    interact_contour_plotly(model=gs.model, metric_name="num_params")

"""