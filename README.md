## Getting started

First, clone the repository:

```git clone git@github.com:MariusAAROS/ramp-kits-cotatenis-sneakers.git```

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

### Challenge description

Get started on this RAMP with the
[dedicated notebook](cotatenis_sneakers_starting_kit.ipynb) (note that you need to have your kaggle credentials set up to download the data from kaggle, see [here](https://www.kaggle.com/docs/api?utm_me=).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
