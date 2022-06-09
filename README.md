# Identification of high order closure terms from fully kinetic simulations using machine learning

__Authors__: [Brecht Laperre](https://orcid.org/0000-0001-7218-3561), [Jorge Amaya](https://orcid.org/0000-0003-1320-8428), [Sara Jamal](https://orcid.org/0000-0002-3929-6668), [Giovanni Lapenta](https://orcid.org/0000-0002-3123-4024)

__DOI__: [10.1063/5.0066397](https://doi.org/10.1063/5.0066397)

Published in Physics of Plasmas (Vol.29, Issue 3), 2022.

## How to cite

> B. Laperre, J. Amaya, S. Jamal, and G. Lapenta, "Identification of high order closure terms from fully kinetic simulations using machine learning", Physics of Plasmas 29, 032706 (2022). https://doi.org/10.1063/5.0066397 

-----

# How to use

1. Download the simulation data from [OSF](https://osf.io/gts8e/).
2. Place the downloaded files in their respective folders in `data/raw`
3. Run the Makefile to prepare the experiments with the configurations available in `config/data`:  
```
make prepare_experiments
```  
 4. To train either a linear regressor, histogram gradient boosting regressor or multilayer perceptron with the configuration files provided in `config/model` and `config/experiment`, run one of the following commands.
 ```
 make linreg_experiment
 make hgbr_experiment
 make mlp_experiment
 ```
 5. To evaluate the models created by the previous command, run the following make command
 ```
 make evaluate
 ```
This will generate tables with results in the `results/` folder.


