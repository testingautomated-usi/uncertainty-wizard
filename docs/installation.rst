Installation
####################

The installation is as simple as 

.. code-block:: console

    pip install uncertainty-wizard

Then, in any file where you want to use uncertainty wizard, add the following import statement:

.. code-block:: python

    import uncertainty_wizard as uwiz

Dependencies
************
We acknowledge that a slim dependency tree is critical to many practical projects.
Thus, the only dependency of uncertainty wizard is ``tensorflow>=2.3.0``.

Note however that if you are still using python 3.6, you have to install
the backport `dataclasses`.

.. note::
    Uncertainty Wizard is tested with tensorflow 2.3.0 and tensorflow is evolving quickly.
    Please do not hesitate to report an issue if you find broken functionality in more recent tensorflow versions.


