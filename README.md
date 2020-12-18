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
   
</p>

This library provides simple and transparent uncertainty and confidence quantification for 
fast-forward tensorflow.keras models.

Import `uncertainty_wizard as uwiz`, and you will be able to
* Calculate uncertainties based on MC-Dropout using the Sequential Keras API.
* Do the same on pre-trained models.
* Handle Deep Ensembles to get higher accuracy and better uncertainty scores.
* Multi-Process your Deep Ensembles on the GPU by switching just one integer.

Want to know how to do all that? Get started with the getting started guide (to be linked here soon).

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
A link to our documentation and user guide will be added here soon.

#### Examples
Our docs contain a list of jupyter based examples, which you can run in colab right away.
You can find them here: (Link will be added soon)

#### Authors and Paper
Uncertainty Wizard was developed at the Università della Svizzera Italiana (USI) in Lugano
by Michael Weiss under the supervision of Prof. Paolo Tonella.
If you like uncertainty wizard and use it for research, you can cite us:
    
    @inproceedings{Weiss2021,
      title={Fail-Safe Execution of Deep Learning based Systems through Uncertainty Monitoring},
      author={Weiss, Michael and Tonella, Paolo},
      booktitle={2021 IEEE 14th International Conference on Software Testing, 
        Validation and Verification (ICST)},
      year={2021},
      organization={IEEE},
      note={forthcoming}
    }

A preprint and a tool paper which provides a deeper technical discussion of ``uncertainty_wizard`` 
will be added in January at latest.

#### Contributing
Issues and PRs are welcome! 
Before investing a lot of time for a PR, please open an issue first, describing your contribution.
This way, we can make sure that the contribution fits well into this repository.
