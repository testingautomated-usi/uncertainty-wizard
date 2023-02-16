![UNCERTAINTY WIZARD](https://github.com/testingautomated-usi/uncertainty-wizard/raw/main/docs/uwiz_logo.PNG)

<p align="center">
    <a href="https://black.readthedocs.io/en/stable/" alt="Code Style: Black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
    <a href="https://choosealicense.com/licenses/mit/" alt="License: MIT">
        <img src="https://img.shields.io/badge/license-MIT-green.svg" /></a>
    <a href="https://github.com/HunterMcGushion/docstr_coverage" alt="Docstr-Coverage: 100%">
        <img src="https://img.shields.io/badge/docstr--coverage-100%25-brightgreen.svg" /></a>
    <img src="https://github.com/testingautomated-usi/uncertainty-wizard/workflows/Unit%20Tests/badge.svg" />
    <a href='https://uncertainty-wizard.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/uncertainty-wizard/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh/testingautomated-usi/uncertainty-wizard">
        <img src="https://codecov.io/gh/testingautomated-usi/uncertainty-wizard/branch/main/graph/badge.svg?token=TWV2TCRE3E"/>
    </a>
    <a href="https://pypi.org/project/uncertainty-wizard/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/uncertainty-wizard">
    </a>
</p>

<p align="center" style="  height: 40px;  line-height: 40px">
    <img src="https://github.githubassets.com/images/icons/emoji/unicode/1f389.png" height=40px/>
    <img src="https://github.githubassets.com/images/icons/emoji/unicode/1f947.png" height=40px/>
    Best Paper Award at ICST 2021 - Testing Tool Track
    <img src="https://github.githubassets.com/images/icons/emoji/unicode/1f947.png" height=40px/>
    <img src="https://github.githubassets.com/images/icons/emoji/unicode/1f389.png" height=40px/>
</p>

Uncertainty wizard is a plugin on top of `tensorflow.keras`,
 allowing to easily and efficiently create uncertainty-aware deep neural networks:

* Plain Keras Syntax: Use the layers and APIs you know and love.
* Conversion from keras: Convert existing keras models into uncertainty aware models.
* Smart Randomness: Use the same model for point predictions and sampling based inference.
* Fast ensembles: Train and evaluate deep ensembles lazily loaded and using parallel processing - optionally on multiple GPUs.
* Super easy setup: Pip installable. Only tensorflow as dependency.

#### Installation

It's as easy as `pip install uncertainty-wizard`

#### Requirements
`uncertainty-wizard` is tested on python 3.8 and recent tensorflow versions. 
Other versions (python 3.6+ and tensorflow 2.3+) should mostly work as well, but may require some mild tweaks.


#### Documentation
Our documentation is deployed to
[uncertainty-wizard.readthedocs.io](https://uncertainty-wizard.readthedocs.io/).
In addition, as uncertainty wizard has a 100% docstring coverage on public method and classes,
your IDE will be able to provide you with a good amount of docs out of the box.

You may also want to check out the technical tool paper [(preprint)](https://arxiv.org/abs/2101.00982),
describing uncertainty wizard functionality and api as of version `v0.1.0`.

#### Examples
A set of small and easy examples, perfect to get started can be found in the 
[models user guide](https://uncertainty-wizard.readthedocs.io/en/latest/user_guide_models.html)
and the [quantifiers user guide](https://uncertainty-wizard.readthedocs.io/en/latest/user_guide_quantifiers.html).
Larger and examples are also provided - and you can run them in colab right away.
You can find them here: [Jupyter examples](https://uncertainty-wizard.readthedocs.io/en/latest/examples.html).

#### Authors and Papers
<!--- Dont forget to update sphinx documentation when changing this paragraph -->
Uncertainty wizard was developed by Michael Weiss and Paolo Tonella at USI (Lugano, Switzerland).
If you use it for your research, please cite these papers:

    @inproceedings{Weiss2021FailSafe,  
      title={Fail-safe execution of deep learning based systems through uncertainty monitoring},
      author={Weiss, Michael and Tonella, Paolo},
      booktitle={2021 14th IEEE Conference on Software Testing, Verification and Validation (ICST)},
      pages={24--35},
      year={2021},
      organization={IEEE} 
    }  

    @inproceedings{Weiss2021UncertaintyWizard,  
      title={Uncertainty-wizard: Fast and user-friendly neural network uncertainty quantification},
      author={Weiss, Michael and Tonella, Paolo},
      booktitle={2021 14th IEEE Conference on Software Testing, Verification and Validation (ICST)},
      pages={436--441},
      year={2021},
      organization={IEEE}
    }  

The first paper [(preprint)](https://arxiv.org/abs/2102.00902) provides 
an empricial study comparing the approaches implemented in uncertainty wizard,
and a list of lessons learned useful for reasearchers working with uncertainty wizard.
The second paper [(preprint)](https://arxiv.org/abs/2101.00982) is a technical tool paper,
 providing a more detailed discussion of uncertainty wizards api and implementation.

References to the original work introducing the techniques implemented 
in uncertainty wizard are provided in the papers listed above.

#### Contributing
Issues and PRs are welcome! Before investing a lot of time for a PR, please open an issue first, describing your contribution.
This way, we can make sure that the contribution fits well into this repository.
We also mark issues which are great to start contributing as as [good first issues](https://github.com/testingautomated-usi/uncertainty-wizard/contribute).
If you want to implement an existing issue, don't forget to comment on it s.t. everyone knows that you are working on it.
