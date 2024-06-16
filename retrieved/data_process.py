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

def evidence2triples(evidence):
    triples = []
    for entity1,relations1 in evidence.items():
        for entity_relation1 in relations1:
            search_relation=[]
            for relation1 in entity_relation1:
                if '~' != relation1[0]:
                    search_relation.append('~'+relation1)
                else:
                    search_relation.append(relation1.replace('~', '', 1))
            search_relation.sort()
            flag=False
            for entity2,relations2 in evidence.items():
                for entity_relation2 in relations2:
                    entity_relation2.sort()
                    if search_relation == entity_relation2:
                        flag=True
                        for relation2 in entity_relation2:
                            if '~' != relation2[0]:
                                if [entity2,relation2,entity1] not in triples:
                                    triples.append([entity2,relation2,entity1])
                            else:
                                if [entity1,relation2.replace('~', '', 1),entity2] not in triples:
                                    triples.append([entity1,relation2.replace('~', '', 1),entity2])
                        break
                    # if search_relation == entity_relation2:
                    #     flag=True
                    #     if [entity2,entity_relation2,entity1] not in triples and [entity1,entity_relation1.sort(),entity2] not in triples :
                    #         triples.append([entity2,entity_relation2,entity1])
                    #     break
            if flag==False:
                for relation1 in entity_relation1:
                    if '~' != relation1[0]:
                        triples.append([entity1,relation1,'-'])
                    else:
                        triples.append(['-',relation1.replace('~', '', 1),entity1])
                # triples.append([entity1,entity_relation1,'-'])
    return triples

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

    filename=["train","dev","test"]
    k=0
    for dataset in [train_data,dev_data,test_data]:
        datas=[]
        id=0
        for claim,value in tqdm(dataset.items()):
            id+=1
            data={'id':id,'claim':claim}
            if 'Evidence' in value:
                value['Evidence']=evidence2triples(value['Evidence'])
            for i in value:
                if i == 'Entity_set':
                    a = []
                    for item in value['Entity_set']:
                        if item not in a:
                            a.append(item)
                    value['Entity_set']=a
                data[i]=value[i]
            # print(data)

            datas.append(data)
            
        with open(f'{args.output_directory_path}/{filename[k]}.json',mode='w',encoding='utf8') as f:
            json.dump(datas,f)
            logger.info(f"write in {args.output_directory_path}/{filename[k]}.json")
            k+=1



            

