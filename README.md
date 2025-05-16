This system prototype is developed atop FedScale ([fedscale.ai](https://fedscale.ai/)), which provides high-level APIs to implement FL algorithms, deploy and evaluate them at scale across diverse hardware and software backends. 
FedScale also includes the largest FL benchmark that contains FL tasks ranging from image classification and object detection to language modeling and speech recognition. 
Moreover, it provides datasets to faithfully emulate FL training environments where FL will realistically be deployed.


## Getting Started
The following installation instructions are from the original [FedScale repository](https://github.com/symbioticlab/fedscale).

### Installation from Source (Linux/MacOS)

If you have [Anaconda](https://www.anaconda.com/products/distribution#download-section) installed and cloned FedScale, here are the instructions.
```
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

Please install NVIDIA [CUDA 10.2](https://developer.nvidia.com/cuda-downloads) or above.


## Repo Structure

```
Repo Root
|---- fedscale          # FedScale source code
  |---- cloud           # Core of FedScale service
  |---- utils           # Auxiliaries (e.g, model zoo and FL optimizer)
  |---- edge            # Backends for practical deployments (e.g., mobile)
  |---- dataloaders     # Data loaders of benchmarking dataset

|---- benchmark         # FedScale datasets and configs
  |---- dataset         # Benchmarking datasets
  |---- configs         # Example configurations

|---- scripts           # Scripts for installing dependencies
|---- examples          # Examples of implementing new FL designs
|---- docs              # FedScale tutorials and APIs
|---- workspace         # Trained models (for embeddings- and gradients-based FIELDING) and client records
```
## Preparing datasets

### Access trained models for getting embeddings or gradients as client representations

Download the `checkpoint` folder from our [asset Google Drive](https://drive.google.com/drive/folders/1KzyvdQzZh2ruRNv9X3vwnbG0v8x_OGWT?usp=sharing) and place it under `workspace`.

### Download datasets
#### Functional Map of the World (fMoW) Dataset
The original dataset repo is at [Functional Map of the World (fMoW) Dataset](https://github.com/fMoW/dataset). 
```
cd [path_to_your_preferred_dataset_directory (by default it would be "/data")]
aws s3 cp s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2 ./
bzip2 -dk manifest.json.bz2
aws s3 cp s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/ . --recursive
```
See the original repo [AWS section](https://github.com/fMoW/dataset?tab=readme-ov-file#aws) for more details. Please make sure that you downloaded `fMoW-rgb`.

Download `country_or_zone_samples_in_2015-2018.json` from our [asset Google Drive](https://drive.google.com/drive/folders/1KzyvdQzZh2ruRNv9X3vwnbG0v8x_OGWT?usp=sharing) into `workspace/scripts/data_prepare`. Then use the following commands to preprocess
```
cd workspace/scripts/data_prepare
python3 prepare_fmow_datasets.py
```

#### Cityscapes
Download the Cityscapes dataset using the instructions on the [website](https://www.cityscapes-dataset.com/) and extract the `leftImg8bit` subdirectories from `leftImg8bit_trainvaltest.zip` into `benchmark/dataset/data/cityscape` (If you prefer extracting the raw data to a different directory, please update `prepare_cityscape_datasets.py` and `benckmark/configs/cityscape/cityscape.yml`accordingly).

After downloading the raw data, use the following commands to preprocess
```
cd workspace/scripts/data_prepare
python3 prepare_cityscape_datasets.py
```
#### Waymo Open
Download the [Ekya](https://github.com/edge-video-services/ekya) preprocessed Waymo dataset from [here](https://drive.google.com/drive/u/1/folders/1dJjnrHfV86eYB4nuMFrNU_kPUzzSknXb) and extract into `$FEDSCALE_HOME/benchmark/dataset/data/waymo_classified` (If you prefer extracting the data to a different directory, please update `benckmark/configs/waymo_classified/waymo_classified.yml` accordingly).

#### Open Images
Download the data directly from [FedScale Datasets](https://fedscale.ai/docs/dataset) with the following command
```
cd benchmark/dataset
bash download.sh download open_images
```

### A note on setting the `data_dir` field in config files
By default, this should just be `$FEDSCALE_HOME/benchmark/dataset/data/[fmow|cityscape|openImg|waymo_classified]`. 

For Cityscapes, Open Images,and Waymo Open, if you prefer storing the datasets under a different `data_dir`, please also move the `client_data_mapping` folder under your choice of `data_dir`.

For FMoW, there is no need to touch the `data_dir` field. Please update the `img_root` argument on line 307 and 309 of `fedscale/cloud/fllibs.py` to be the `output_dir` you specified in `prepare_fmow_datasets.py`.


## Starting an Experiment
First, make sure that you are in the repo root directory. Then, run the following command to start a job:
```
fedscale driver start benchmark/configs/fmow/fmow.yml
```

## Reference
### Ekya: Continuous Learning of Video Analytics Models on Edge Compute Servers
```
@inproceedings {276952,
    author={Romil Bhardwaj and Zhengxu Xia and Ganesh Ananthanarayanan and Junchen Jiang and Yuanchao Shu and Nikolaos Karianakis and Kevin Hsieh and Paramvir Bahl and Ion Stoica}
    title = {{Ekya: Continuous Learning of Video Analytics Models on Edge Compute Servers}},
    booktitle = {USENIX Symposium on Networked Systems Design and Implementation (NSDI 22)},
    year = {2022},
    address = {Renton, WA},
    url = {https://www.usenix.org/conference/nsdi22/presentation/bhardwaj},
    publisher = {USENIX Association},
    month = apr,
}
```
### FedScale: Benchmarking Model and System Performance of Federated Learning at Scale
```
@inproceedings{fedscale-icml22,
    title={FedScale: Benchmarking Model and System Performance of Federated Learning at Scale},
    author={Fan Lai and Yinwei Dai and Sanjay S. Singapuram and Jiachen Liu and Xiangfeng Zhu and Harsha V. Madhyastha and Mosharaf Chowdhury},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2022}
}
```