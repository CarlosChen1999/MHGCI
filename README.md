# Extracting Faithful Rationales in Knowledge Graph for Fact Verification
Fact verification (FV) via reasoning on knowledge graph (KG) retrieve the evidence from the KG to verify truthfulness of the claims. Extracting \textit{faithful rationales}, which are precise relation paths of entities involved in the claims, prevents noise paths from affecting the reasoning model and enhances the credibility of the FV system. However, existing methods rarely extract faithful rationales in multi-relation extraction, especially under the interference of counterfactual relations. To alleviate these issue, we propose multi-layer relation extraction with causal inference (MRCI). Our method consists of three components: (i) We construct a multi-layer relation extraction graph to solves multi-relation extractions through multiple inferences between anchors. (ii) Based on causal analysis, we conducted self-optimizing causal intervention model for assistant anchors to mitigate the counterfactual problem. (iii) We designed a grounding method to extract evidence from the KG database by faithful rationales. Experimental results on a large fact verification via reasoning on knowledge graph dataset FactKG have demonstrated that MRCI can extract faithful rationales and achieve improved results with the accuracy of 84.41\%.

## Getting Started
### Dataset
You can download the FactKG dataset [here](https://drive.google.com/drive/folders/1q0_MqBeGAp5_cBJCBf_1alYaYm14OeTk?usp=share_link). Place the dataset file in the data folder.

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
cd /home/user/cdq/FactKG/with_evidence/classifier
python baseline.py --data_path <dataset_path> --kg_path <kg_path.pickle> --mode train
```

