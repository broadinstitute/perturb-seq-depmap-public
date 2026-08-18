# perturb-seq-depmap
DepMap's Perturb-seq pilot in 16 cell lines

# Setup

# Python

We use python 3.11 and poetry for package management. You may install poetry according to the instructions [here](https://python-poetry.org/docs/). Navigate to the root directory and run the commands below. You can use [pyenv](https://github.com/pyenv/pyenv) to install Python versions as needed if your environment is not using python 3.11.
```
# install required python version 
pyenv install 3.11
# specify the python version to use in the current directory
peynv local 3.11
# install packages using poetry
poetry install
```

## R

We use R version 4.4.2 and Renv for package management in R. Install R version 4.4.2 from CRAN [here](https://cran.r-project.org/). If you use the CRAN installer, the latest version installed should be set as the current version. If you need to manually change the version of R used, you can create a symbolic link to this version with the commands (e.g. for Apple silicon): 
```
unlink /Library/Frameworks/R.framework/Versions/Current
ln -s /Library/Frameworks/R.framework/Versions/4.4-arm64 /Library/Frameworks/R.framework/Versions/Current
```

Open the project file perturb-seq-pilot.Rproj in an RStudio session. This should automatically configure `renv` for package management. Run `renv:restore()` to install the requirements from the corresponding `renv.lock`. Note that the R environment requires the installation of some C packages (`gcc`, `openssl@3`, `freetype`, `harfbuzz`, `fribidi`). These may need to be installed manually (e.g. using [Homebrew](https://brew.sh/)). 

# Running SCEPTRE

In the RStudio terminal run `Rscript src/sceptre_benchmark.R` to perform differential expression between knockouts and control guides and for all cell lines. Afterwards, run `Rscript src/arm_truncation_analysis.R` to perform differential expression analysis between control cells with arm loss or gain and control cells with intact arms. To process the deep rescreen data, run `Rscript src/sceptre_h2h_full.R` and `Rscript src/sceptre_h2h_downsamples.R`

# Running python analysis

Generate all tables using the `generate.py` script by running `poetry run python src/generate.py`. Panels for each figure may be produced by running the figure-specific scripts.
```
poetry run python src/figure1.py
poetry run python src/figure2.py
poetry run python src/figure3.py
poetry run python src/figure4.py
poetry run python src/figure5.py
poetry run python src/figure6.py
poetry run python src/supplemental1.py
poetry run python src/supplemental2.py
poetry run python src/supplemental3.py
poetry run python src/supplemental4.py
poetry run python src/supplemental5.py
poetry run python src/supplemental6.py
poetry run python src/supplemental7.py
```