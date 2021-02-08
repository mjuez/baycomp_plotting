# Baycomp Plotting

The **baycomp_plotting** is a python package for building good-looking plots of 
bayesian posteriors obtained with [baycomp](https://github.com/janezd/baycomp).

This package could be useful for scientific purposes, specially in the area of 
Machine Learning.

## Author

- Mario Juez-Gil <<mariojg@ubu.es>>\
Department of Computer Science\
Universidad de Burgos\
[ADMIRABLE Research Group](https://admirable-ubu.es)

## Installation

This package can be installed using PIP.

```shell
pip install baycomp_plotting
```

## Basic Usage

The package can be imported as follows:

```python
import baycomp_plotting as bplt
```

Two plotting functions (`tern`, and `dens`), and one class with four matplotlib alternative colors (`Color`) are provided.

### Colors

Four alternative colors to default matplotlib colors are provided:

<img src="https://github.com/mjuez/assets/blob/main/baycomp_plotting/colors.jpg" width="70%">

_Example:_

```python
import baycomp_plotting as bplt

print(bplt.Color.BLUE)
```

_Output:_

```
'#008ece'
```

### Density plots

For plotting the comparison of two classifiers on a single dataset, `dens` function could be used. It's parameters are the following:

- `p`: baycomp posterior.
- `label`: label of the density function.
- `ls`: line style (use a matplotlib line style) [default: `-`]
- `color`: density function color [default: `Color.BLUE`]

_Example:_

```python
import baycomp_plotting as bplt
import baycomp as bc

posterior = bc.CorrelatedTTest(left_classifier_acc, right_classifier_acc, rope=0.01)
fig = bplt.dens(posterior, label='C1', ls='-', color=bplt.Color.BLUE)
```

_Output:_

<img src="https://github.com/mjuez/assets/blob/main/baycomp_plotting/dens_1.png" width="50%">

The output figure will have a new function named `add_posterior` so you can add more posteriors to the figure. The parameters are the same as for `dens`.

_Example:_

```python
import baycomp_plotting as bplt
import baycomp as bc

posterior = bc.CorrelatedTTest(left_classifier_1_acc, right_classifier_acc, rope=0.01)
posterior_1 = bc.CorrelatedTTest(left_classifier_2_acc, right_classifier_acc, rope=0.01)
fig = bplt.dens(posterior, label='C1', ls='-', color=bplt.Color.BLUE)
fig.add_posterior(posterior_1, label='C2', ls=(0,(5,1)), color=bplt.Color.GRAY)
fig.legend() # you can show the legend
```

_Output:_

<img src="https://github.com/mjuez/assets/blob/main/baycomp_plotting/dens_2.png" width="50%">

### Ternary plots

For plotting the comparison of two classifiers on multiple datasets using a ternary plot, `tern` function could be used. It's parameters are the following:

- `p`: baycomp posterior.
- `names`: an array containing Left and Right region labels. [default: `["L", "R"]`]

_Example:_

```python
import baycomp_plotting as bplt
import baycomp as bc

posterior = bc.HierarchicalTest(left_classifier_acc, right_classifier_acc, rope=0.01)
fig = bplt.tern(posterior)
```

_Output:_

<img src="https://github.com/mjuez/assets/blob/main/baycomp_plotting/tern_1.png" width="50%">

## Comparison against baycomp default plots

_Density:_

<img src="https://github.com/mjuez/assets/blob/main/baycomp_plotting/correlation.jpg" width="90%">

<img src="https://github.com/mjuez/assets/blob/main/baycomp_plotting/correlation_1.jpg" width="90%">

_Ternary:_

<img src="https://github.com/mjuez/assets/blob/main/baycomp_plotting/tern.jpg" width="90%">

## Contribute

Feel free to submit any pull requests 😊

## Acknowlegments

This work was supported by the pre-doctoral grant (EDU/1100/2017) of the 
Consejería de Educación of the Junta de Castilla y León, Spain, and the 
European Social Fund.

## License

This work is licensed under [GNU GPL v3](LICENSE).

## Citation policy

Please, cite this work as:

```
@software{baycomp_plotting,
  author       = {Mario Juez-Gil},
  title        = {{mjuez/baycomp_plotting}},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.4244542},
  url          = {https://doi.org/10.5281/zenodo.4244542}
}
```

