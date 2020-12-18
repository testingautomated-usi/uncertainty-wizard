![UNCERTAINTY WIZARD](https://github.com/testingautomated-usi/uncertainty-wizard/raw/main/docs/uwiz_logo.PNG)

**WARNING** This is a pre-release, published while setting up the CI. 
The first official release will be deployed in a couple of days.

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
</p>

Uncertainty wizard is a plugin on top of `tensorflow.keras`,
 allowing to easily and efficiently create uncertainty-aware deep neural networks:

* Plain Keras Syntax: Use the layers and APIs you know and love.
* Conversion from keras: Convert existing keras models into uncertainty aware models.
* Smart Randomness: Use the same model for point predictions and sampling based inference.
* Fast ensembles: Train and evaluate deep ensembles lazily loaded and using parallel processing.
* Super easy setup: Pip installable. Only tensorflow as dependency.

#### Installation

It's as easy as `pip install uncertainty-wizard`

#### Requirements
- tensorflow >= 2.2.0
- python 3.6* / 3.7 / 3.8

Note that **tensorflow 2.4** has just been released. 
We will test and create compatibility with uncertainty wizard in the next couple of weeks.
Until then, please stick to tensorflow 2.3.x.

*python 3.6 requires to `pip install dataclasses`

#### Documentation
Our documentation is deployed to:
[uncertainty-wizard.readthedocs.io/](https://uncertainty-wizard.readthedocs.io/)

Note that we have a 100% docstring coverage on public method and classes.
Hence, your IDE will be able to provide you with a good amount of docs out of the box.

#### Examples
A set of small and easy examples, perfect to get started can be found in the 
[user guide for our models](https://uncertainty-wizard.readthedocs.io/en/latest/user_guide_models.html)
and the [user guide for our quantifiers](https://uncertainty-wizard.readthedocs.io/en/latest/user_guide_quantifiers.html)

Larger and examples are also provided - and you can run them in colab right away.
You can find them here: [List of jupyter examples](https://uncertainty-wizard.readthedocs.io/en/latest/examples.html)

#### Authors and Paper
``uncertainty-wizard`` was developed by Michael Weiss and Paolo Tonella at USI (Lugano, Switzerland).
If was first presented in the following paper 
[(preprint)](https://uncertainty-wizard.readthedocs.io/en/latest/literature.html):  

<details>  
  <summary>Fail-Safe Execution of Deep Learning based Systems through Uncertainty Monitoring (expand for BibTex)</summary>  

    @inproceedings{Weiss2021,  
      title={Fail-Safe Execution of Deep Learning based Systems through Uncertainty Monitoring},  
      author={Weiss, Michael and Tonella, Paolo},  
      booktitle={2021 IEEE 14th International Conference on Software Testing,   
        Validation and Verification (ICST)},  
      year={2021},  
      organization={IEEE},  
      note={forthcoming}  
    }  

</details>

We are also currently writing a technical tool paper, describing design choices and challenges.

#### Contributing
Issues and PRs are welcome! 
Before investing a lot of time for a PR, please open an issue first, describing your contribution.
This way, we can make sure that the contribution fits well into this repository.
