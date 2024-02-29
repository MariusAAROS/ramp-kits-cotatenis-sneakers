# COTATENIS-SNEAKERS | Multiclass classification on a sneakers dataset

**WARNING** : This project has been tested on Python 3.10 and 3.11, if you encounter any error please update your Python version.

## Getting started

First, clone the repository:

```git clone git@github.com:MariusAAROS/ramp-kits-cotatenis-sneakers.git```

### Install the required packages

To run a submission and the notebook, you will need to install `torch` as well as the other dependencies listed in `requirements.txt`, inside a dedicated environment.

Regarding the installation of `torch`, we recommand installing a version optimized for GPU. If you are using Windows, you may run the following command in your terminal :

```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

If you are using another OS, or if you do not have a GPU, you can find other `torch` versions on the [PyTorch website](https://pytorch.org/get-started/locally/).

Once you have installed `torch`, you may install the other required dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

### Challenge description

Get started on this challenge with the
[dedicated notebook](cotatenis_sneakers_starting_kit.ipynb) (note that you need to have your kaggle credentials set up to download the data from kaggle, see [here](https://www.kaggle.com/docs/api?utm_me=).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance, if you have some code to submit stored in a `my_submission` folder, this folder should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

To obtain more detailed information regarding this command line, you may run :

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
