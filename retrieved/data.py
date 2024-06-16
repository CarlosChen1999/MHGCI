from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

import json
import torch
from tqdm import tqdm

class FactKG_Retrieve(Dataset):
    def __init__(self, dataset_path, type, tokenizer):
        with open(f'{dataset_path}/{type}.json', mode='r', encoding='utf8') as f:
            dataset = json.load(f)
        with open('../data/relations_for_final.pickle', mode='rb') as f:
            relations = pickle.load(f)
        self.type = type
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.relations = relations
        self.feature=self.dataset_feature(dataset)
        # self.feature=self.dataset_feature_atlop(dataset)

    def __getitem__(self, idx):
        return self.feature[idx]

    def __len__(self):
        return len(self.feature)

    def dataset_feature(self, datatset):
        features = []
        pos,neg=0,0
        for data in tqdm(datatset,desc='load '+self.type+'.data'):
            input = self.tokenizer.encode_plus(
                data['claim'], add_special_tokens=True, padding = True, truncation=True)
            entity = data['Entity_set']
            if len(entity) == 1:
                entity.append('-')
            hts = []
            entity_postion=[]
            map_e_pos=[]
            for e in entity:
                e_ids=self.tokenizer.encode(e.replace('_',' '), add_special_tokens=False)
                e_pos=self.get_entity_pos(input['input_ids'],e,e_ids)
                entity_postion.append(e_pos)
                map_e_pos.append(e)
            
            input_unk=self.reduce_entity(input['input_ids'],entity_postion)
            labels=[]
            for h in entity:
                for t in entity:
                    h_id=map_e_pos.index(h)
                    t_id=map_e_pos.index(t)
                    if h != t and (h_id, t_id) not in hts :
                        
                        hts.append((h_id, t_id))
                        if 'Evidence' in data:
                            entity_pair_relations = [0] * len(self.relations)
                            flag_neg=True # 区分正负样例（正：有关系的，负：没关系）
                            for triple in data['Evidence']:
                                if h == triple[0] and t == triple[2]:
                                    relation_idx=self.relations.index(triple[1])
                                    entity_pair_relations[relation_idx] = 1
                                    flag_neg=False
                                elif t == triple[0] and h == triple[2]:
                                    relation_idx=self.relations.index('~'+triple[1])
                                    entity_pair_relations[relation_idx] = 1
                                    flag_neg=False
                            if flag_neg:
                                neg+=1
                                entity_pair_relations=[1]+entity_pair_relations
                            else:
                                pos+=1
                                entity_pair_relations=[0]+entity_pair_relations
                            labels.append(entity_pair_relations)
            entity_list=[]
            for ent in entity:
                entity_list.append(self.tokenizer.encode(ent))
            feature= {'input_ids': input['input_ids'],
                      'input_unk_ids': input_unk,
                                       'hts': hts,
                                       'pos':entity_postion,
                                       'entity':entity_list,
                                       'labels': labels,
                                       }
            features.append(feature)

        print("# of positive examples {}.".format(pos))
        print("# of negative examples {}.".format(neg))
        return features
    
    def longest_common_subsequence(self,input_ids, h_ids):
        # 创建一个二维数组来保存子问题的解
        dp = [[0] * (len(h_ids) + 1) for _ in range(len(input_ids) + 1)]

        # 填充数组
        for i in range(1, len(input_ids) + 1):
            for j in range(1, len(h_ids) + 1):
                if input_ids[i - 1] == h_ids[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # 回溯查找最长公共子序列的位置
        index_i, index_j = len(input_ids), len(h_ids)
        while index_i > 0 and index_j > 0:
            if input_ids[index_i - 1] == h_ids[index_j - 1]:
                index_i -= 1
                index_j -= 1
            elif dp[index_i - 1][index_j] > dp[index_i][index_j - 1]:
                index_i -= 1
            else:
                index_j -= 1

        start_index = index_i
        end_index = start_index + dp[len(input_ids)][len(h_ids)] - 1

        return start_index, end_index
    
    def get_entity_pos(self,input_ids,h,h_ids):
        h_pos=[0] * len(input_ids)
        if h.replace('_',' ')=='-':
            h_pos[0] = 1
        else:
            h_pos_start = [i for i in range(len(input_ids)) if input_ids[i:i+len(h_ids)] == h_ids]
            if h_pos_start==[]:
                e_start,e_end=self.longest_common_subsequence(input_ids, h_ids)
                if e_end >= e_start:
                    for k in range(e_start, e_end+1):
                        h_pos[k] = 1
                else:
                    h_pos[0] = 1
            else:
                for i in h_pos_start:
                    for k in range(i, i+len(h_ids)):
                        h_pos[k] = 1
        return h_pos
    
    def reduce_entity(self, input_ids,entity_pos):
        unk_inputs=[]
        for e_pos in entity_pos:
            unk_input_ids=input_ids
            interested_indices = [index for index, value in enumerate(e_pos) if value == 1]
            if len(interested_indices) > 0:
                unk_input_ids = [self.tokenizer.unk_token_id if i in interested_indices else value for i, value in enumerate(unk_input_ids)]
                unk_inputs.append(unk_input_ids)
            else:
                print("error:entity not found!")
        return unk_inputs
    
if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    FactKG_Retrieve('./data','train',tokenizer)

