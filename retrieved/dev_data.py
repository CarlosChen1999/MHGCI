import argparse
import pickle
import logging
import json
import os
from itertools import permutations, chain
from tqdm import tqdm

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory_path',default='../data/factkg', type=str)
    parser.add_argument('--kg_path',default='../data/dbpedia_2015_undirected_light.pickle', type=str)
    parser.add_argument('--output_directory_path',default='./data', type=str)

    args = parser.parse_args()
    return args

def get_data():
    data_directory_path = args.data_directory_path
    output_directory_path = args.output_directory_path
    #test_path = args.test_path

    ### load the dataset 
    with open(f'{data_directory_path}/factkg_train.pickle', 'rb') as file:
        train_data = pickle.load(file)

    with open(f'{data_directory_path}/factkg_dev.pickle', 'rb') as file:
        dev_data = pickle.load(file)

    with open(f'{data_directory_path}/factkg_test.pickle', 'rb') as file:
        test_data = pickle.load(file)

    # print("load all")
    return train_data,dev_data,test_data

if __name__=='__main__':
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    formator = logging.Formatter(fmt="%(asctime)s | [ %(levelname)s ] | [%(message)s]",datefmt="%X")
    sh = logging.StreamHandler()
    fh = logging.FileHandler("./test_log_01.log", encoding="utf-8")
    logger.addHandler(sh)
    sh.setFormatter(formator)

    args=define_argparser()

    train_data,dev_data,test_data=get_data()

    filename=["dev"]
    k=0
    for dataset in [dev_data]:
        datas=[]
        id=0
        claims={}
        entities={}
        relations={}
        for claim,value in tqdm(dataset.items()):
            a = []
            for item in value['Entity_set']:
                if item not in a:
                    a.append(item)
            value['Entity_set']=a
            for entity in value['Entity_set']:
                claims[f"{id}"]=claim
                entities[f"{id}"]=entity
                relations[f"{id}"]=[]
                for rels in value['Evidence'][entity]:
                    for rel in rels:
                        if rel not in relations[f"{id}"]:
                            relations[f"{id}"].append(rel)
                id+=1
        with open("dev_class.json", "w") as fh:
            json.dump({'claims': claims, 'entity': entities, 'relation': relations}, fh)

    with open('./result_eval.json', mode='rb') as f:
        pre = json.load(f)

    # for id,value in entities.items():
    #     if pre['entity'][id]!=entities[id]:
    #         print(id)

            

