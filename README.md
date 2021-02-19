# govuk-content-similarity

## Overview

Experimental work to identify semantically-similar content across all GOV.UK web pages.

## Context

Consider the case of a user searching on GOV.UK for information on paying the Dartford Crossing charge. They would
expect to only have one page on this on GOV.UK. However, if they find multiple pages on paying the Dartford Crossing
charge then this can be confusing to the user as they would not know immediately which one is appropriate.

- [Pay the Darford Crossing chart (Dart Charge)](https://www.gov.uk/pay-dartford-crossing-charge)
- [How to pay the Dart Charge](https://www.gov.uk/government/publications/dart-charge-leaflet)

In principle, GOV.UK may not want multiple pages on the Dartford Crossing charge for this reason and would prefer to
remove one of these pages. However, due to resource constraint, then it is difficult to identify these similar pages.

By leveraging the below two methods, we can identify semantically-similar pages at scale so that this problem can be
proactively managed.

1. Universal Sentence Encoder + Approximate Nearest Neighbour (Oh Yeah)
1. doc2vec + Self-Organising Maps (SOMs)

The key principles are to represent each GOV.UK page as a document embedding then use distance metrics to quantify
how semantically-similar two pages are.

## Requirements

To run the code in this GitHub repository, please make sure your system meets the following requirements:

- Unix-like operating system (macOS, Linux, …);
- [`direnv`](https://direnv.net/) installed, including shell hooks;
- [`.envrc`](.envrc) allowed/trusted by `direnv` to use the environment variables - see
[below](#allowingtrusting-envrc);
- Python 3.8 or above; and
- Python packages installed from the `requirements.txt` file.

Note there may be some Python IDE-specific requirements around loading environment variables, which are not considered
here.

### Allowing/trusting `.envrc`

To allow/trust the [`.envrc`](.envrc) run the `allow` command using `direnv` at the top level of this repository.

```shell script
direnv allow
```

> **Note:** If you're using PyCharm, then you will need to apply a few more steps before running `direnv allow` in your shell:
> - In your shell, run `pip install python-dotenv`.
> - On PyCharm, click `PyCharm` -> `Preferences` -> `Plugins` and download the `EnvFile` plugin.
> - On PyCharm, edit your configuration to `Enable EnvFile` by ticking the checkbox.
> - On PyCharm, click `PyCharm` -> `Preferences` -> `Build, Execution, Deployment` -> `Console` -> `Python Console` and in the `Starting script` section, add the following Python code:
>    + `from dotenv import load_dotenv`
>    + `load_dotenv()`

### Installing Python packages

To install required Python packages via `pip`, first [set up a Python virtual
environment](#creating-a-python-virtual-environment); this ensures you do not install the packages globally.

Then run the following `make` command at the top level of this repository:

```shell script
make requirements
```

Once you have installed the packages, remember to [set up pre-commit hooks](#installing-pre-commit-hooks).

### Creating a Python virtual environment

Creating a Python virtual environment depends on whether you are using [base Python](#base-python-interpreter) or
[Anaconda](#anaconda-interpreter) as your interpreter.

#### Base Python interpreter

If you are using base Python, there are multiple ways to create virtual environments in Python using `pip`, including
(but not limited to):

- [`venv`](https://docs.python.org/3/tutorial/venv.html);
- [`virtualenv`](https://virtualenv.pypa.io/en/stable/);
- [`pipenv`](https://github.com/pypa/pipenv); and
- [`pyenv`](https://github.com/pyenv/pyenv) with its `virtualenv` [plugin](https://github.com/pyenv/pyenv-virtualenv).

Follow the documentation of your chosen method to create a Python virtual environment.

#### Anaconda interpreter

If you are using [Anaconda or `conda`](https://www.anaconda.com/), following their
[documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to set up a
conda environment.

## Folder structure

An overview of the folder structure, and the top-level files can be found [here](docs/structure/README.md).

## Installing pre-commit hooks

This repo uses the Python package [`pre-commit`](https://pre-commit.com) to manage pre-commit hooks. Pre-commit hooks
are actions which are run automatically, typically on each commit, to perform some common set of tasks. For example, a
pre-commit hook might be used to run any code linting automatically, providing any warnings before code is committed,
ensuring that all of our code adheres to a certain quality standard.

For this repo, we are using `pre-commit` for a number of purposes:
- Checking for any secrets being committed accidentally;
- Checking for any large files (over 5MB) being committed; and
- Cleaning Jupyter notebooks, which means removing all outputs and execution counts.

We have configured `pre-commit` to run automatically on _every commit_. By running on each commit, we ensure
that `pre-commit` will be able to detect all contraventions and keep our repo in a healthy state.

In order for `pre-commit` to run, action is needed to configure it on your system.

- [Install](#installing-python-packages) the `pre-commit` package into your Python environment; and
- Run `pre-commit install` to set-up `pre-commit` to run when code is _committed_.

### Setting up a baseline for the `detect-secrets` hook (if one doesn't already exist)

The `detect-secrets` hook requires that you generate a baseline file if one is not already present within the root
directory. This is done via running the following at the root of the repo:

```shell script
detect-secrets scan > .secrets.baseline
```

Next, audit the baseline that has been generated by running:

```shell script
detect-secrets audit .secrets.baseline
```

When you run this command, you'll enter an interactive console and be presented with a list of high-entropy string /
anything which _could_ be a secret, and asked to verify whether or not this is the case. By doing this, the hook will
be in a position to know if you're later committing any _new_ secrets to the repo and it will be able to alert you
accordingly.

### If `pre-commit` detects secrets during commit:

If `pre-commit` detects any secrets when you try to create a commit, it will detail what it found and where to go to
check the secret.

If the detected secret is a false-positive, you should update the secrets baseline through the following steps:

- Run `detect-secrets scan --update .secrets.baseline` to index the false-positive(s);
- Next, audit all indexed secrets via `detect-secrets audit .secrets.baseline` (the same as during initial set-up, if a
secrets baseline doesn't exist); and
- Finally, ensure that you commit the updated secrets baseline in the same commit as the false-positive(s) it has been
updated for.

If the detected secret is actually a secret (or other sensitive information), remove the secret and re-commit. There is
no need to update the secrets baseline in this case.

If your commit contains a mixture of false-positives and actual secrets, remove the actual secrets first before
updating and auditing the secrets baseline.

### Note on Jupyter notebook cleaning

It may be necessary or useful to keep certain output cells of a Jupyter notebook, for example charts or graphs
visualising some set of data. To do this, add the following comment at the top of the input block:

```shell script
# [keep_output]
```

This will tell `pre-commit` not to strip the resulting output of this cell, allowing it to be committed.
