# Faithful inference chains Extraction for Fact Verification over Multi-view Heterogeneous Graph with Causal Intervention
KG-based fact verification verifies the truthfulness of claims by retrieving evidence graphs from the knowledge graph. The *faithful inference chains*, which are precise relation paths between the mentioned entities and evidence entities, retrieve precise evidence graphs addressing poor performance and weak logic for fact verification. Due to the diversity of relation paths, existing methods rarely extract faithful inference chains. To alleviate these issues, we propose Multi-view Heterogeneous Graph with Causal Intervention (MHGCI): (i) We construct a Multi-view Heterogeneous Graph enhancing relation path extraction from the view of different mentioned entities. (ii) We propose a self-optimizing causal intervention model to generate assistant entities mitigating the out-of-distribution problem caused by counterfactual relations. (iii) We propose a grounding method to extract evidence graphs from the KG by faithful inference chains. Experiments on the public KG-based fact verification dataset FactKG demonstrate that our model provides precise evidence graphs and achieves state-of-the-art performance.

## Getting Started
### Dataset
You can download the FactKG dataset [here](https://drive.google.com/drive/folders/1q0_MqBeGAp5_cBJCBf_1alYaYm14OeTk?usp=share_link). Create a new `data` folder and place the dataset in this folder

### Environment

```shell
pip install -r requirements.txt
```

### Data preprocessing

```shell
cd retrieved
python data_process.py
python dev_data.py
```

## Model Training

```shell
python train.py --experiment_name <experiment_name>
```

## Model Prediction

```shell
python train.py --load_path <weight_path> --wandb_use False
```

## Overall Performance

```shell
cd /with_evidence/classifier
python baseline.py --data_path <dataset_path> --kg_path <kg_path.pickle> --mode train
```

