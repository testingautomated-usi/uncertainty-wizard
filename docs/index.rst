Uncertainty Wizard
==================

Uncertainty wizard is a plugin on top of ``tensorflow.keras``,
allowing to easily and efficiently create uncertainty-aware deep neural networks:

- **Plain Keras Syntax:** Use the layers and APIs you know and love.
- **Conversion from keras:** Convert existing keras models into uncertainty aware models.
- **Smart Randomness:** Use the same model for point predictions and sampling based inference.
- **Fast ensembles:** Train and evaluate deep ensembles lazily loaded and using parallel processing.
- **Super easy setup:** Pip installable. Only tensorflow as dependency.



.. toctree::
    :caption: Documentation
    :maxdepth: 1

    Installation <installation>
    User Guide: Models <user_guide_models>
    User Guide: Quantifiers <user_guide_quantifiers>
    Examples <examples>
    Complete API <complete_api>
    Paper <paper>
    Sources on Github <https://github.com/testingautomated-usi/uncertainty_wizard>


.. _TensorflowGuide: https://www.tensorflow.org/guide

Note that our documentation assumes basic knowledge of the tensorflow.keras API.
If you do not know tensorflow.keras yet, check out the TensorflowGuide_.


