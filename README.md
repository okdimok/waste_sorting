To run the scripts you need to install [Anaconda](https://www.anaconda.com/download/#linux)

If you are familiar with git, than in your terminal
```bash
git clone https://github.com/okdimok/waste_sorting.git
```
Otherwise just download [a zipped version of this repository](https://github.com/okdimok/waste_sorting/archive/master.zip) and extract it to a "waste_sorting" folder.

Than you need to create an environment with the packages, needed for the class.u

```bash
cd waste_sorting
conda env create -n waste_sorting -f conda_environment.yml
. activate waste_sorting
jupyter notebook
```

Make sure to chose waste_sorting kernel.
