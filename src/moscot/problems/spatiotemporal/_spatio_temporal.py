import types
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

from ott.geometry import epsilon_scheduler

from anndata import AnnData
import numpy as np

from moscot import _constants
from moscot._types import (
    ArrayLike,
    CostKwargs_t,
    Numeric_t,
    OttCostFnMap_t,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
)
from moscot.base.problems.birth_death import BirthDeathMixin, BirthDeathProblem
from moscot.base.problems.compound_problem import B, Callback_t, K
from moscot.problems.space import AlignmentProblem, SpatialAlignmentMixin
from moscot.problems.time import TemporalMixin

__all__ = ["SpatioTemporalProblem"]


class SpatioTemporalProblem(  # type: ignore[misc]
    TemporalMixin[Numeric_t, BirthDeathProblem],
    BirthDeathMixin,
    AlignmentProblem[Numeric_t, BirthDeathProblem],
    SpatialAlignmentMixin[Numeric_t, BirthDeathProblem],
):
    """Class for analyzing time series spatial single-cell data.

    Parameters
    ----------
    adata
        Annotated data object.
    kwargs
        Keyword arguments for :class:`~moscot.base.problems.CompoundProblem`.
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    def prepare(
        self,
        time_key: str,
        spatial_key: str = "spatial",
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        normalize_spatial: bool = True,
        policy: Literal["sequential", "triu", "tril", "explicit"] = "sequential",
        cost: OttCostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[Union[bool, str]] = None,
        b: Optional[Union[bool, str]] = None,
        marginal_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        subset: Optional[Sequence[Tuple[Numeric_t, Numeric_t]]] = None,
        xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        x_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        y_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        xy_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        x_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        y_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
    ) -> "SpatioTemporalProblem":
        """Prepare the spatiotemporal problem problem.

        .. seealso::
            - See :doc:`../../notebooks/tutorials/500_spatiotemporal` on how to
              prepare and solve the :class:`~moscot.problems.spatiotemporal.SpatioTemporalProblem`.
            - See :doc:`../../notebooks/examples/problems/TemporalProblem/800_score_genes_for_marginals` on how to
              :meth:`score genes for proliferation and apoptosis <score_genes_for_marginals>`.

        Parameters
        ----------
        time_key
            Key in :attr:`~anndata.AnnData.obs` where the time points are stored.
        spatial_key
            Key in :attr:`~anndata.AnnData.obsm` where the spatial coordinates are stored.
        joint_attr
            How to get the data for the :term:`linear term` in the :term:`fused <fused Gromov-Wasserstein>` case:

            - :obj:`None` - `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
              on :attr:`~anndata.AnnData.X` is computed.
            - :class:`str` - key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        normalize_spatial
            Whether to normalize the spatial coordinates. If :obj:`True`, the coordinates are normalized
            by standardizing them.
        policy
            Rule which defines how to construct the subproblems using :attr:`obs['{time_key}'] <anndata.AnnData.obs>`.
            Valid options are:

            - ``'sequential'`` - align subsequent time points ``[(t0, t1), (t1, t2), ...]``.
            - ``'triu'`` - upper triangular matrix ``[(t0, t1), (t0, t2), ..., (t1, t2), ...]``.
            - ``'tril'`` - lower triangular matrix ``[(t_n, t_n-1), (t_n, t0), ..., (t_n-1, t_n-2), ...]``.
            - ``'explicit'`` - explicit sequence of subsets passed via ``subset = [(b3, b0), ...]``.
        cost
            Cost function to use. Valid options are:

            - :class:`str` - name of the cost function for all terms, see :func:`~moscot.costs.get_available_costs`.
            - :class:`dict` - a dictionary with the following keys and values:

              - ``'xy'`` - cost function for the :term:`linear term`.
              - ``'x'`` - cost function for the source :term:`quadratic term`.
              - ``'y'`` - cost function for the target :term:`quadratic term`.
        cost_kwargs
            Keyword arguments for the :class:`~moscot.base.cost.BaseCost` or any backend-specific cost.
        a
            Source :term:`marginals`. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where the source marginals are stored.
            - :class:`bool` - if :obj:`True`,
              :meth:`estimate the marginals <moscot.base.problems.BirthDeathProblem.estimate_marginals>`,
              otherwise use uniform marginals.
            - :obj:`None` - set to :obj:`True` if :attr:`proliferation_key` or :attr:`apoptosis_key` is not :obj:`None`.
        b
            Target :term:`marginals`. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where the target marginals are stored.
            - :class:`bool` - if :obj:`True`,
              :meth:`estimate the marginals <moscot.base.problems.BirthDeathProblem.estimate_marginals>`,
              otherwise use uniform marginals.
            - :obj:`None` - set to :obj:`True` if :attr:`proliferation_key` or :attr:`apoptosis_key` is not :obj:`None`.
        marginal_kwargs
            Keyword arguments for :meth:`~moscot.base.problems.BirthDeathProblem.estimate_marginals`.
            It always contains :attr:`proliferation_key` and :attr:`apoptosis_key`,
            see :meth:`score_genes_for_marginals` for more information.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`problems` - the prepared subproblems.
        - :attr:`solutions` - set to an empty :class:`dict`.
        - :attr:`spatial_key` - key in :attr:`~anndata.AnnData.obsm` where spatial coordinates are stored.
        - :attr:`temporal_key` - key in :attr:`~anndata.AnnData.obs` where time points are stored.
        - :attr:`stage` - set to ``'prepared'``.
        - :attr:`problem_kind` - set to ``'quadratic'``.
        """
        # spatial key set in AlignmentProblem
        # handle_joint_attr and handle_cost in AlignmentProblem
        self.temporal_key = time_key
        marginal_kwargs = dict(marginal_kwargs)

        estimate_marginals = self.proliferation_key is not None or self.apoptosis_key is not None
        a = estimate_marginals if a is None else a
        b = estimate_marginals if b is None else b
        if self.apoptosis_key is not None:
            marginal_kwargs["apoptosis_key"] = self.apoptosis_key
        if self.proliferation_key is not None:
            marginal_kwargs["proliferation_key"] = self.proliferation_key

        return super().prepare(  # type: ignore[return-value]
            spatial_key=spatial_key,
            batch_key=time_key,
            joint_attr=joint_attr,
            normalize_spatial=normalize_spatial,
            policy=policy,  # type: ignore[arg-type]
            reference=None,
            cost=cost,
            cost_kwargs=cost_kwargs,
            a=a,
            b=b,
            marginal_kwargs=marginal_kwargs,
            subset=subset,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback=xy_callback,
            x_callback_kwargs=x_callback_kwargs,
            y_callback_kwargs=y_callback_kwargs,
            xy_callback_kwargs=xy_callback_kwargs,
        )

    def solve(
        self,
        alpha: float = 0.5,
        epsilon: Union[float, epsilon_scheduler.Epsilon] = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        jit: bool = True,
        min_iterations: Optional[int] = None,
        max_iterations: Optional[int] = None,
        threshold: float = 1e-3,
        linear_solver_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "SpatioTemporalProblem":
        r"""Solve the spatiotemporal problem.

        .. seealso::
            - See :doc:`../../notebooks/tutorials/500_spatiotemporal` on how to
              prepare and solve the :class:`~moscot.problems.spatiotemporal.SpatioTemporalProblem`.

        Parameters
        ----------
        alpha
            Parameter in :math:`(0, 1]` that interpolates between the :term:`quadratic term` and
            the :term:`linear term`. :math:`\alpha = 1` corresponds to the pure :term:`Gromov-Wasserstein` problem while
            :math:`\alpha \to 0` corresponds to the pure :term:`linear problem`.
        epsilon
            :term:`Entropic regularization`.
        tau_a
            Parameter in :math:`(0, 1]` that defines how much :term:`unbalanced <unbalanced OT problem>` is the problem
            on the source :term:`marginals`. If :math:`1`, the problem is :term:`balanced <balanced OT problem>`.
        tau_b
            Parameter in :math:`(0, 1]` that defines how much :term:`unbalanced <unbalanced OT problem>` is the problem
            on the target :term:`marginals`. If :math:`1`, the problem is :term:`balanced <balanced OT problem>`.
        rank
            Rank of the :term:`low-rank OT` solver :cite:`scetbon:21b`.
            If :math:`-1`, full-rank solver :cite:`peyre:2016` is used.
        scale_cost
            How to re-scale the cost matrices. If a :class:`float`, the cost matrices
            will be re-scaled as :math:`\frac{\text{cost}}{\text{scale_cost}}`.
        batch_size
            Number of rows/columns of the cost matrix to materialize during the solver iterations.
            Larger value will require more memory.
        stage
            Stage by which to filter the :attr:`problems` to be solved.
        initializer
            How to initialize the solution. If :obj:`None`, ``'default'`` will be used for a full-rank solver and
            ``'rank2'`` for a low-rank solver.
        initializer_kwargs
            Keyword arguments for the ``initializer``.
        jit
            Whether to :func:`~jax.jit` the underlying :mod:`ott` solver.
        min_iterations
            Minimum number of :term:`(fused) GW <Gromov-Wasserstein>` iterations.
        max_iterations
            Maximum number of :term:`(fused) GW <Gromov-Wasserstein>` iterations.
        threshold
            Convergence threshold of the :term:`GW <Gromov-Wasserstein>` solver.
        linear_solver_kwargs
            Keyword arguments for the inner :term:`linear problem` solver.
        device
            Transfer the solution to a different device, see :meth:`~moscot.base.output.BaseDiscreteSolverOutput.to`.
            If :obj:`None`, keep the output on the original device.
        kwargs
            Keyword arguments for :meth:`~moscot.problems.space.AlignmentProblem.solve`.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`solutions` - the :term:`OT` solutions for each subproblem.
        - :attr:`stage` - set to ``'solved'``.
        """
        # TODO(michalk8): use locals (and in other places)
        return super().solve(  # type: ignore[return-value]
            alpha=alpha,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            rank=rank,
            scale_cost=scale_cost,
            batch_size=batch_size,
            stage=stage,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            jit=jit,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            linear_solver_kwargs=linear_solver_kwargs,
            device=device,
            **kwargs,
        )

    def interpolate_cell_state(
        self,
        source: K,
        intermediate: K,
        target: K,
        interpolation_parameter: Optional[float] = None,
        n_interpolated_cells: Optional[int] = None,
        account_for_unbalancedness: bool = False,
        batch_size: int = 256,
        posterior_marginals: bool = True,
        seed: Optional[int] = None,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> float:
        """Sample `n_interpolated` cells using the :term:`OT` coupling matrix. Interpolate gene expression and 
        cell coordinate based on the `interpolation_parameter`. Compute `Wasserstein distance <https://en.wikipedia.org/wiki/Wasserstein_metric>`_ between
        :term:`OT`-interpolated and intermediate cells (true data).

        .. seealso::
            - TODO(MUCDK): create an example showing the usage.

        This is a validation method which interpolates cells between the ``source`` and ``target`` distributions
        leveraging the :term:`OT` coupling to approximate cells at the ``intermediate`` time point.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        intermediate
            Key identifying the intermediate distribution.
        target
            Key identifying the target distribution.
        interpolation_parameter
            Interpolation parameter in :math:`(0, 1)` defining the weight of the ``source`` and ``target``
            distributions. If :obj:`None`, it is linearly interpolated.
        n_interpolated_cells
            Number of cells used for interpolation. If :obj:`None`, use the number of cells in the ``intermediate``
            distribution.
        account_for_unbalancedness
            Whether to account for unbalancedness by assuming exponential cell growth and death.
        batch_size
            Number of rows/columns of the cost matrix to materialize during :meth:`push` or :meth:`pull`.
            Larger value will require more memory.
        posterior_marginals
            Whether to use :attr:`posterior_growth_rates` or :attr:`prior_growth_rates`.
            TODO(MUCDK): needs more explanation
        seed
            Random seed used when sampling the interpolated cells.
        backend
            Backend used for the distance computation.
        kwargs
            Keyword arguments for the distance function, depending on the ``backend``:

            - ``'ott'`` - :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`.

        Returns
        -------
        The distance between :term:`OT`-interpolated cells and cells at the ``intermediate`` time point.
        It is recommended to compare this to the distances computed by :meth:`compute_time_point_distances` and
        :meth:`compute_random_distance`.
        """ 
        source_data, _, intermediate_data, _, target_data = self._get_data(  # type: ignore[misc]
            source,
            intermediate,
            target,
            posterior_marginals=posterior_marginals,
            only_start=False,
        )
        interpolation_parameter = self._get_interp_param(
            source, intermediate, target, interpolation_parameter=interpolation_parameter
        )
        n_interpolated_cells = n_interpolated_cells if n_interpolated_cells is not None else len(intermediate_data)
        gex_interpolation, spt_interpolation = self._interpolate_gex_and_spatial_with_ot(
            number_cells=n_interpolated_cells,
            source_data=source_data,
            target_data=target_data,
            source=source,
            target=target,
            interpolation_parameter=interpolation_parameter,
            account_for_unbalancedness=account_for_unbalancedness,
            batch_size=batch_size,
            seed=seed,
        )
        w_dist = self._compute_wasserstein_distance(intermediate_data, gex_interpolation, backend=backend, **kwargs)
        return gex_interpolation, spt_interpolation, w_dist
    
    def _interpolate_gex_and_spatial_with_ot(
        self,
        number_cells: int,
        source_data: ArrayLike,
        target_data: ArrayLike,
        source: K,
        target: K,
        interpolation_parameter: float,
        account_for_unbalancedness: bool = True,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Adapted from TemporalMixin._interpolate_gex_with_ot"""
        rows_sampled, cols_sampled = self._sample_from_tmap(
            source=source,
            target=target,
            n_samples=number_cells,
            source_dim=len(source_data),
            target_dim=len(target_data),
            batch_size=batch_size,
            account_for_unbalancedness=account_for_unbalancedness,
            interpolation_parameter=interpolation_parameter,
            seed=seed,
        )
        interpolated_gex = (
            source_data[np.repeat(rows_sampled, [len(col) for col in cols_sampled]), :] * (1 - interpolation_parameter)
            + target_data[np.hstack(cols_sampled), :] * interpolation_parameter
        )
        # Get the source and target spatial
        source_spatial=self.problems[(source, target)].adata_src.obsm[self.spatial_key]
        target_spatial=self.problems[(source, target)].adata_tgt.obsm[self.spatial_key]
        interpolated_spatial = (
            source_spatial[np.repeat(rows_sampled, [len(col) for col in cols_sampled]), :] * (1 - interpolation_parameter)
            + target_spatial[np.hstack(cols_sampled), :] * interpolation_parameter
        )

        return interpolated_gex, interpolated_spatial 

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return (
            _constants.SEQUENTIAL,
            _constants.TRIL,
            _constants.TRIU,
            _constants.EXPLICIT,
        )  # type: ignore[return-value]

    @property
    def _base_problem_type(self) -> Type[B]:
        return BirthDeathProblem  # type: ignore[return-value]
