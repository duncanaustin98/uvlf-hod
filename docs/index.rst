halogal
=======

``halogal`` models UV luminosity functions and galaxy clustering at high
redshift with a UV–halo-mass relation (UVHMR) and a halo occupation
distribution (HOD), and fits that model to data with ``halogal.inference``.

.. code-block:: bash

   pip install halogal                 # forward model
   pip install "halogal[inference]"    # + emcee, dynesty, corner, h5py

The compiled dependency ``camb`` (via ``halomod``) installs most reliably from
conda-forge; see ``environment.yml`` in the repository.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/basic_usage
   tutorials/mstar_hod_tutorial
   tutorials/ffb_hod_tutorial
   tutorials/inference_tutorial
   tutorials/validate_shuntov25

.. toctree::
   :maxdepth: 2
   :caption: User guide

   inference
   crosscorrelations

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   changelog
