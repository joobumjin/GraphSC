# Graph SC

This on-going project is exploring how GNNs can learn from stem cell microscopy imaging.

This project was developed as an advancement upon [this paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6994191), which explored the application of traditional ML models, MLPs, and CNNs on stem cell microscopy imaging, while also introducing QBAM, a new method for gathering said imaging. The relevant data for this publication can be found [here](https://isg.nist.gov/deepzoomweb/data/RPEimplants).

You can read about this project's results as of May 2025 in a report form through my [Undergraduate Honor's Thesis](https://drive.google.com/file/d/1-m_HFV6-Dp_7ZMAOASNqEYBrU09KBUT2/view?usp=sharing). Additionally, check out model tuning performance in [this W&B project for a model multitasking on 2 different objectives](https://wandb.ai/bumjin_joo-brown-university/qbam-Both-Flex-Multi/workspace?nw=nwuserbumjin_joo)

Most of the code is run on Brown's Oscar CCV compute cluster using Slurm, as defined by the `.sh` files in the repo.

This particular repository was built from an older repository, whose work I inherited and significantly expanded upon.

All files meant to run the experiments and tasks of the project (*e.g.* optimizing model hyperparameters on TER values) are found in the root directory (*e.g.* `optuna_search.py`)

All code relevant to the models used throughout this project can be found in the `models/` directory

Code relevant to preprocessing data files into dataloaders can be found in the `preprocessing/` directory

Code used to run, train, test, and interpret models can be found in the `utils/` directory

Finally, the Slurm job submit scripts can be found within the `run_sh/` directory

<!-- # SLab-GNN
This was done using python version 3.9.19

## Helpful Links

### Cellpose
* Train and run Cellpose: https://colab.research.google.com/drive/1-CXFO6vhielLmazHwLlDYlFjgx0mmFF3#scrollTo=Da-Rtx09DEZB
* Cellpose Inference only: https://colab.research.google.com/drive/1z711ShE75MchIgRAZJxOm48Mi5AQSDK-#scrollTo=kmUFyN6NoDNH

### GNN
* Train and test GNN: https://colab.research.google.com/drive/1gDCU4T7D1FIASSv5FrEVjnlbApu-tY_C?usp=sharing -->
