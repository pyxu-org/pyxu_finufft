Pyxu_FINUFFT Documentation
##########################

This package provides a Pyxu interface to the `Flatiron Institute Non-uniform Fast Fourier Transform (FINUFFT)
<https://github.com/flatironinstitute/finufft>`_ library.

The *Non-Uniform Fast Fourier Transform (NUFFT)* generalizes the FFT to off-grid data.  There are three types of NUFFTs
proposed in the literature:

* Type 1 (*non-uniform to uniform*),
* Type 2 (*uniform to non-uniform*),
* Type 3 (*non-uniform to non-uniform*).

See the notes below, including [FINUFFT]_, for definitions of the various transforms and algorithmic details.

FINUFFT supports transforms of dimension :math:`d=\{1,2,3\}`.

Notes
-----

* **Mathematical Definition.**

  Let :math:`d\in\{1,2,3\}` and consider the mesh

  .. math::

     \mathcal{I}_{N_1,\ldots,N_d}
     =
     \mathcal{I}_{N_1} \times \cdots \times \mathcal{I}_{N_d}
     \subset \mathbb{Z}^d,

  where the mesh indices :math:`\mathcal{I}_{N_i}\subset\mathbb{Z}` are given for each dimension :math:`i=1,\dots, d`
  by:

  .. math::

     \mathcal{I}_{N_i}
     =
     \begin{cases}
         [[-N_i/2, N_i/2-1]], & N_i\in 2\mathbb{N} \text{ (even)}, \\
         [[-(N_i-1)/2, (N_i-1)/2]], & N_i\in 2\mathbb{N}+1 \text{ (odd)}.
     \end{cases}


  Then the NUFFT operators approximate, up to a requested relative accuracy :math:`\varepsilon > 0`, the following
  exponential sums:

  .. math::

     \begin{align}
         (1)\; &v_{\mathbf{n}} = \sum_{m=1}^{M} w_{m} e^{j \cdot s \cdot 2\pi \langle \mathbf{n}, \mathbf{x}_{m} \rangle}, \qquad &\text{Type 1 (non-uniform to uniform)}\\
         (2)\; &w_{m} = \sum_{\mathbf{n}} v_{\mathbf{n}} e^{j \cdot s \cdot 2\pi \langle \mathbf{n},
         \mathbf{x}_{m} \rangle}, \qquad &\text{Type 2 (uniform to non-uniform)}\\
         (3)\; &z_{n} = \sum_{m=1}^{M} w_{m} e^{j \cdot s \cdot 2\pi \langle \mathbf{v}_{n}, \mathbf{x}_{m} \rangle},
         \qquad &\text{Type 3 (non-uniform to non-uniform)},\\
     \end{align}

  where :math:`s \in \{+1, -1\}` defines the sign of the transform and :math:`v_{\mathbf{n}}, w_{m}, z_{n}\in
  \mathbb{C}`.  For the type-1 and type-2 NUFFTs, the non-uniform samples :math:`\mathbf{x}_{m}` are assumed to lie in
  :math:`[-\pi,\pi)^{d}`.  For the type-3 NUFFT, the non-uniform samples :math:`\mathbf{x}_{m}` and
  :math:`\mathbf{z}_{n}` are arbitrary points in :math:`\mathbb{R}^{d}`.

* **Adjoint NUFFTs.**

  The type-1 and type-2 NUFFTs with opposite signs form an *adjoint pair*.  The adjoint of the type-3 NUFFT is obtained
  by flipping the transform's sign and switching the roles of :math:`\mathbf{z}_n` and :math:`\mathbf{x}_{m}` in (3).

* **Complexity.**

  Naive evaluation of the exponential sums (1), (2) and (3) above costs :math:`O(NM)`, where :math:`N=N_{1}\ldots N_{d}`
  for the type-1 and type-2 NUFFTs.  NUFFT algorithms approximate these sums to a user-specified relative tolerance
  :math:`\varepsilon` in log-linear complexity in :math:`N` and :math:`M`.  The complexity of the various NUFFTs are
  given by (see [FINUFFT]_):

  .. math::

     &\mathcal{O}\left(N \log(N) + M|\log(\varepsilon)|^d\right)\qquad &\text{(Type 1 and 2)}\\
     &\mathcal{O}\left(\Pi_{i=1}^dX_iZ_i\sum_{i=1}^d\log(X_iZ_i) + (M + N)|\log(\varepsilon)|^d\right)\qquad &\text{(Type 3)}

  where :math:`X_i = \max_{j=1,\ldots,M}|(\mathbf{x}_{j})_i|` and :math:`Z_i = \max_{k=1,\ldots,N}|(\mathbf{z}_k)_i|`
  for :math:`i=1,\ldots,d`.  The terms above correspond to the complexities of the FFT and spreading/interpolation steps
  respectively.

* **Memory footprint.**

  The complexity and memory footprint of the type-3 NUFFT can be arbitrarily large for poorly-centered data, or for data
  with a large spread.  An easy fix consists in centering the data before/after the NUFFT via pre/post-phasing
  operations, as described in equation (3.24) of [FINUFFT]_.  This optimization is automatically carried out by FINUFFT
  if the compute/memory gains are non-negligible.

* **Backend.**

  The NUFFT transforms are computed via Python wrappers to `FINUFFT <https://github.com/flatironinstitute/finufft>`_ and
  `cuFINUFFT <https://github.com/flatironinstitute/cufinufft>`_.  (See also [FINUFFT]_ and [cuFINUFFT]_.) These
  librairies perform the expensive spreading/interpolation between nonuniform points and the fine grid via the
  "exponential of semicircle" kernel.

* **Optional Parameters.**

  [cu]FINUFFT exposes many optional parameters to adjust the performance of the algorithms, change the output format, or
  provide debug/timing information.  While the default options are sensible for most setups, advanced users may
  overwrite them via the ``kwargs`` parameter of :py:class:`~pyxu_finufft.operator.NUFFT1`,
  :py:class:`~pyxu_finufft.operator.NUFFT2`, and :py:class:`~pyxu_finufft.operator.NUFFT3`.  See the `guru interface
  <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_ from FINUFFT and its `companion page
  <https://finufft.readthedocs.io/en/latest/opts.html#options-parameters>`_ for details.

   .. admonition:: Hint

      The NUFFT is performed in **batches of (n_trans,)**, where `n_trans` denotes the number of simultaneous transforms
      requested.  (See the ``n_trans`` parameter of `finufft.Plan
      <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.)


Installation
------------

.. code-block:: bash

   pip install pyxu_finufft

.. todo::

   Explain how to use it via pyxu imports and not via pyxu_finufft.


.. toctree::
   :maxdepth: 1
   :hidden:

   api/index
   examples/index
   references
