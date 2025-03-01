This system prototype is developed atop FedScale ([fedscale.ai](https://fedscale.ai/)), which provides high-level APIs to implement FL algorithms, deploy and evaluate them at scale across diverse hardware and software backends. 
FedScale also includes the largest FL benchmark that contains FL tasks ranging from image classification and object detection to language modeling and speech recognition. 
Moreover, it provides datasets to faithfully emulate FL training environments where FL will realistically be deployed.


## Getting Started
The following installation instructions are from the original [FedScale repository](https://github.com/symbioticlab/fedscale).

### Quick Installation (Linux)

You can simply run `install.sh`.

```
source install.sh # Add `--cuda` if you want CUDA 
pip install -e .
```

Update `install.sh` if you prefer different versions of conda/CUDA.

### Installation from Source (Linux/MacOS)

If you have [Anaconda](https://www.anaconda.com/products/distribution#download-section) installed and cloned FedScale, here are the instructions.
```
cd FedScale

# Please replace ~/.bashrc with ~/.bash_profile for MacOS
FEDSCALE_HOME=$(pwd)
echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc 
echo alias fedscale=\'bash $FEDSCALE_HOME/fedscale.sh\' >> ~/.bashrc 
conda init bash
. ~/.bashrc

conda env create -f environment.yml
conda activate fedscale
pip install -e .
```

Finally, install NVIDIA [CUDA 10.2](https://developer.nvidia.com/cuda-downloads) or above if you want to use FedScale with GPU support.

## FedScale Runtime
FedScale Runtime is an scalable and extensible deployment as well as evaluation platform to simplify and standardize FL experimental setup and model evaluation. 
It evolved from our prior system, [Oort](https://github.com/SymbioticLab/Oort), which has been shown to scale well and can emulate FL training of thousands of clients in each round.

Please go to `./fedscale/cloud` directory and follow the [README](./fedscale/cloud/README.md) to set up FL training scripts and the [README](./fedscale/edge/android/README.md) for practical on-device deployment.


## Repo Structure

```
Repo Root
|---- fedscale          # FedScale source code
  |---- cloud           # Core of FedScale service
  |---- utils           # Auxiliaries (e.g, model zoo and FL optimizer)
  |---- edge            # Backends for practical deployments (e.g., mobile)
  |---- dataloaders     # Data loaders of benchmarking dataset

|---- docker            # FedScale docker and container deployment (e.g., Kubernetes)
|---- benchmark         # FedScale datasets and configs
  |---- dataset         # Benchmarking datasets
  |---- configs         # Example configurations

|---- scripts           # Scripts for installing dependencies
|---- examples          # Examples of implementing new FL designs
|---- docs              # FedScale tutorials and APIs
```

## Starting an Experiment
First, make sure that you are in the repo root directory. Then, run the following command to start a job:
```
fedscale driver start benchmark/configs/fmow/fmow.yml
```