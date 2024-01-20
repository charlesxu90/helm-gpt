# HELM-GPT: de novo macrocyclic peptide design using generative pre-trained transformer

! [HELM-GPT-image](./helm-gpt.png)
## Installation and running
### Clone and Create environment
Clone git repository and then create the environment as follows.

```commandline
mamba env create -f environment.yml
mamba activate helm-gpt-env
```


```commandline
mamba install -c conda-forge rdkit
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Install dependencies to run agent

