# Global subduction slow slip events and their association with earthquakes

This repository includes the reference material for *Global subduction slow slip events and their association with earthquakes* by Kelian Dascher-Cousineau and Roland Bürgmann. This study aggreagtes slow slip event catalogs and document their interplay with earthquakes.


The code includes:

1. Slow slip event catalogs in `Datasets/Slow_slip_datasets`
2. Jupyter notebooks reproducing the analysis `notebooks`
3. Source code for catalog manipulation and processing packaged in `src`

## Reproducing the results
To reproduce the results from the paper, please run the both `manuscript.ipynb` and `sensitivity.ipynb`

## Installation
The code has been tested on MacOS.


1. Make sure you have the latest version of [`conda`](https://docs.conda.io/en/latest/miniconda.html) installed.
2. Install the dependencies and create a new conda environment.
    ```bash
    conda env create -f environment.yml
    conda activate sse
    ```
3. Install the `src` package.
    ```bash
    pip install -e .
    ```


## Examples

### Making a map

```
from src.data import EarthquakeCatalog
from src.catalogs import MexicoSlowSlipCatalog

# Get slow slip events in Mexico
slowslip = MexicoSlowSlipCatalog()

# download and earthquake catalog
earthquakes = EarthquakeCatalog(
    filename='mexico_earthquakes', 
    use_other_catalog=True, 
    other_catalog=slowslip, 
    other_catalog_buffer=2,
)

# plot a basic map with the slow slip events and the earthquakes
ax = slowslip.plot_map()
earthquakes.plot_map(ax=ax)
```

### Quick stack

```
from stack import Stack
from matplotlib import pyplot as plt

stack = Stack()

times, temporal_density = stack.kde_timeseries() 

fig, ax = plt.subplots()
ax.plot(times, temporal_density) # note that the ends taper to zero 

```


## Available datatsets

### Slow slip datasets: 

Please refer to the [README](Datasets/Slow_slip_datasets/README) in `Datasets/Slow_slip_datasets` for a comprehensive description of the catalogs. Please cite original sources as listed in the [README](Datasets/Slow_slip_datasets/README)!

We currenly include the folowing slow slip event datasets:
### Regional Catalogs

| Catalog Name                                   |
|------------------------------------------------|
| `src.catalogs.slowslip.JapanSlowslipCatalog`   |
| `src.catalogs.slowslip.MexicoSlowslipCatalog`  |
| `src.catalogs.slowslip.CostaRicaSlowslipCatalog` |

### Individual Catalogs

| Catalog Name                                         | Link                                                              |
|------------------------------------------------------|-------------------------------------------------------------------|
| [`src.catalogs.slowslip.JapanSlowslipDatabase`](http://www-solid.eps.s.u-tokyo.ac.jp/~sloweq/) | [Link](http://www-solid.eps.s.u-tokyo.ac.jp/~sloweq/)             |
| `src.catalogs.slowslip.RoussetSlowSlipCatalog`       |                                                                   |
| `src.catalogs.slowslip.ElYousfiSlowSlipCatalog`      |                                                                   |
| `src.catalogs.slowslip.LouSlowSlipCatalog`           |                                                                   |
| `src.catalogs.slowslip.XieSlowSlipCatalog`           |                                                                   |
| `src.catalogs.slowslip.PerrySlowSlipCatalog`         |                                                                   |
| `src.catalogs.slowslip.WilliamsSlowSlipCatalog`      |                                                                   |
| `src.catalogs.slowslip.IkariSlowSlipCatalog`         |                                                                   |
| `src.catalogs.slowslip.MichelSlowSlipCatalog`        |                                                                   |
| `src.catalogs.slowslip.ChenSlowSlipCatalog`          |                                                                   |
| `src.catalogs.slowslip.OkadaAlaskaSlowSlipCatalog`   |                                                                   |
| `src.catalogs.slowslip.JaraSlowSlipCatalog`          |                                                                   |


The following is used for Slab2 data `src/data/slab2.py`

In addition, we provide access miscelaneous datasets not utilized in the study:
- `src/catalogs/swarm_catalogs.py`: swarm data from Nishikawa and Ide 2017
- `src/catalogs/earthquake_catalogs.py`: currently only supports the EST earthquake catalog





