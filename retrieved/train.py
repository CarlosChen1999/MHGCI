import torch
import argparse
import pickle
import logging
import json
import os
import numpy as np
import wandb
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import BertTokenizer, BertModel,AutoConfig,AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from functools import reduce
from data import FactKG_Retrieve
from model import REModel
from utils import set_seed,collate_fn

def train_model(args,model,train_dataloader, dev_dataloader, test_dataloader,dev_dataset):
    # 调整不同层的学习率
    new_layer = ["extractor", "bilinear", "fc", "linear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    
    # set_seed(args)
    # model.zero_grad()

    num_epoch=args.num_train_epochs
    global_step = 0
    best_recall = 0.0
    
    total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    logger.info("Total steps: {}".format(total_steps))
    logger.info("Warmup steps: {}".format(warmup_steps))
    logger.info('start training!')
    for epoch in range(int(args.num_train_epochs)):
        model.train()
        optimizer.zero_grad()
        for data in tqdm(train_dataloader, desc=f'Training epoch{epoch}', leave=False):
            inputs, lab_tensor = data
            inputs = {'input_ids': inputs[0].to(args.device),
                    'attention_mask': inputs[1].to(args.device),
                    'pos': inputs[2],
                    'hts': inputs[3],
                    'entity':inputs[4].to(args.device),
                    'entity_mask':inputs[5].to(args.device),
                    'entity_num':inputs[6],
                    'input_unk_ids':inputs[7].to(args.device),
                    'input_unk_mask':inputs[8].to(args.device),
                    'labels':lab_tensor,
                #   'relations':relation_ids.to(args.device),
                #   'relations_mask':relation_ids_mask.to(args.device)
                }
            # prob = model(inputs)
            # loss = F.nll_loss(prob, lab_tensor)
            loss,_ = model(**inputs)
            # running_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if args.wandb_use=='True':
                    wandb.log({'Epoch': epoch, 'Step': global_step, 'Loss': loss*args.gradient_accumulation_steps})
                else:
                    logger.info({'Epoch': epoch, 'Step': global_step, 'Loss': loss*args.gradient_accumulation_steps})
            if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0 or global_step==total_steps:
                logger.info('Start eval!')
                with torch.no_grad():
                    precision,recall,f1 = eval_model(model, dev_dataloader,dev_dataset)
                    logger.info('Dev total precision: {0}, recall: {1}, f1: {2}, '.format(precision,recall,f1))
                    if args.wandb_use=='True':
                        wandb.log({'precision': precision, 'recall': recall, 'f1': f1})
                    if recall > best_recall:
                        best_recall = recall
                        torch.save({'epoch': epoch,
                                    'model': model.state_dict(),
                                    # 'best_recall': best_recall}, args.model_path + f"/{datetime.now().strftime('%H:%M:%S')}_{best_recall}.pt")
                                    'best_recall': best_recall}, args.model_path + f"/{args.experiment_name}.pt")
                                    # 'best_recall': best_recall}, args.model_path + f"/best_noloss.pt")
                                    # 'best_recall': best_recall}, args.model_path + f"/ablation_study_double.pt")
                        logger.info("Saved best epoch {0}, best accuracy {1}".format(epoch, best_recall))

def eval_model(model, validset_reader,dataset):
    model.eval()

    preds = []
    lab_tensors=[]
    for data in tqdm(validset_reader, desc='Evaling', leave=False):
        inputs, lab_tensor = data
        inputs = {'input_ids': inputs[0].to(args.device),
                  'attention_mask': inputs[1].to(args.device),
                  'pos': inputs[2],
                  'hts': inputs[3],
                  'entity':inputs[4].to(args.device),
                  'entity_mask':inputs[5].to(args.device),
                  'entity_num':inputs[6],
                  'input_unk_ids':inputs[7].to(args.device),
                  'input_unk_mask':inputs[8].to(args.device),
                #   'relations':relation_ids.to(args.device),
                #   'relations_mask':relation_ids_mask.to(args.device)
                  }
        pred, *_ = model(**inputs)

        pred = pred.cpu().numpy()
        lab_tensor = np.concatenate(lab_tensor, axis=0).astype(np.float32)
        # pred[np.isnan(pred)] = 0
        pred=pred[:,1:]
        lab_tensor=lab_tensor[:,1:]
        
        preds.append(pred)
        lab_tensors.append(lab_tensor)

    lab_tensors=np.concatenate(lab_tensors, axis=0).astype(np.float32)
    preds=np.concatenate(preds, axis=0).astype(np.float32)

    # lab_tensors_flatten=lab_tensors.flatten()
    # preds_flatten=preds.flatten()

    # precision = precision_score(lab_tensors_flatten, preds_flatten)
    # recall = recall_score(lab_tensors_flatten, preds_flatten)
    # f1 = f1_score(lab_tensors_flatten, preds_flatten)

    with open('../data/relations_for_final.pickle', mode='rb') as f:
        relations = pickle.load(f)
    claims={}
    entity={}
    output={}
    k=0
    result_i=0
    for i in range(len(dataset)):
        entity_relation={}
        for ent in dataset.dataset[i]['Entity_set']:
            if ent=='-':
                continue
            entity_relation[ent]=[]
        for hts in dataset.feature[i]['hts']:
            entity1=dataset.dataset[i]['Entity_set'][hts[0]]
            entity2=dataset.dataset[i]['Entity_set'][hts[1]]
            for relation in [relations[q] for q in np.nonzero(preds[k] == 1)[0]] :
                
                if entity1 !='-' and relation not in entity_relation[entity1] :
                    entity_relation[entity1].append(relation)
                
            k+=1

        for key,value in entity_relation.items():
            claims[f"{result_i}"]=dataset.dataset[i]['claim']
            entity[f"{result_i}"]=key
            output[f"{result_i}"]=value
            result_i+=1
        
        i+=1

    with open('./dev_class.json', mode='rb') as f:
        dev_gts = json.load(f)
    gts = [value for key,value in dev_gts["relation"].items()]
    prs = [value for key,value in output.items()]

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for gt, pr in zip(gts, prs):
    # Calculate precision, recall
        for label in pr:
            if label in gt and label != 'Unknown':
                true_positives += 1
            elif label not in gt and label != 'Unknown':
                false_positives += 1
        for label in gt:
            if label != 'Unknown' and label not in pr:
                false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

    return precision,recall,f1

def eval_report(model, validset_reader,dataset,type=None):
    with torch.no_grad():
        model.eval()

        preds = []
        lab_tensors=[]
        for data in tqdm(validset_reader, desc='Predicting', leave=False):
            inputs, lab_tensor = data
            inputs = {'input_ids': inputs[0].to(args.device),
                    'attention_mask': inputs[1].to(args.device),
                    'pos': inputs[2],
                    'hts': inputs[3],
                    'entity':inputs[4].to(args.device),
                    'entity_mask':inputs[5].to(args.device),
                    'entity_num':inputs[6],
                    'input_unk_ids':inputs[7].to(args.device),
                    'input_unk_mask':inputs[8].to(args.device),
                    }
            pred, *_ = model(**inputs)

            pred = pred.cpu().numpy()
            lab_tensor = np.concatenate(lab_tensor, axis=0).astype(np.float32) if type=='dev' else lab_tensor
            pred=pred[:,1:]
            lab_tensor=lab_tensor[:,1:] if type=='dev' else lab_tensor
            
            preds.append(pred)
            lab_tensors.append(lab_tensor) if type=='dev' else lab_tensor

        lab_tensors=np.concatenate(lab_tensors, axis=0).astype(np.float32) if type=='dev' else lab_tensor
        preds=np.concatenate(preds, axis=0).astype(np.float32)

        # 自己分析问题
        with open('../data/relations_for_final.pickle', mode='rb') as f:
            relations = pickle.load(f)
        res = []
        k=0
        for i in range(len(dataset)):
            for hts in dataset.feature[i]['hts']:
                res.append(
                        {
                            'id': dataset.dataset[i]['id'],
                            'claim': dataset.dataset[i]['claim'],
                            'hts': hts,
                            'preds': [relations[q] for q in np.nonzero(preds[k] == 1)[0]],
                            'lab_tensors': [relations[q] for q in np.nonzero(lab_tensors[k] == 1)[0]] if type=='dev' else lab_tensor,
                        }
                    )
                k+=1
            i+=1

        # 对比分析问题
        claims={}
        entity={}
        output={}

        path={}
        k=0
        result_i=0
        for i in tqdm(range(len(dataset)),desc="relation finding"):
            entity_relation={}
            entity_relation_path={}
            for ent in dataset.dataset[i]['Entity_set']:
                if ent=='-':
                    continue
                entity_relation[ent]=[]
                entity_relation_path[ent]=[]
            for hts in dataset.feature[i]['hts']:
                entity1=dataset.dataset[i]['Entity_set'][hts[0]]
                entity2=dataset.dataset[i]['Entity_set'][hts[1]]
                for relation in [relations[q] for q in np.nonzero(preds[k] == 1)[0]] :
                    if entity1 !='-' and relation not in entity_relation[entity1] :
                        entity_relation[entity1].append(relation)
                    # if entity2 !='-' and inverse_relationship(relation) not in entity_relation[entity2]:
                    #     entity_relation[entity2].append(inverse_relationship(relation))
                # 记录path
                if entity1 !='-' and [relations[q] for q in np.nonzero(preds[k] == 1)[0]] not in entity_relation_path[entity1]:
                    entity_relation_path[entity1].append([relations[q] for q in np.nonzero(preds[k] == 1)[0]])
                
                k+=1

            for key,value in entity_relation.items():
                claims[f"{result_i}"]=dataset.dataset[i]['claim']
                entity[f"{result_i}"]=key
                output[f"{result_i}"]=value
                path[f"{result_i}"]=entity_relation_path[key]
                result_i+=1
            
            i+=1

        if type=='dev':
            with open('./dev_class.json', mode='rb') as f:
                dev_gts = json.load(f)
            gts = [value for key,value in dev_gts["relation"].items()]
            prs = [value for key,value in output.items()]

            true_positives = 0
            false_positives = 0
            false_negatives = 0
            for gt, pr in zip(gts, prs):
            # Calculate precision, recall
                for label in pr:
                    if label in gt and label != 'Unknown':
                        true_positives += 1
                    elif label not in gt and label != 'Unknown':
                        false_positives += 1
                for label in gt:
                    if label != 'Unknown' and label not in pr:
                        false_negatives += 1

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

        # 对比分析
        with open("result_eval.json", "w") as fh:
            json.dump({'claims': claims, 'entity': entity, 'output': output, 'path': path}, fh)

        # 自己分析
        with open("result.json", "w") as fh:
            json.dump(res, fh)
        if  type=='dev':
            return precision,recall,f1
        else:
            return 0,0,0

# 调整为逆关系
def inverse_relationship(relation):
    if '~' !=relation[0]:
        relation=relation
    else:
        relation=relation.replace('~', '', 1)
    return relation

if __name__=='__main__':
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    formator = logging.Formatter(fmt="%(asctime)s | [ %(levelname)s ] | [%(message)s]",datefmt="%X")
    sh = logging.StreamHandler()
    fh = logging.FileHandler("./train_log.log", encoding="utf-8")
    logger.addHandler(sh)
    sh.setFormatter(formator)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--wandb_use',default='True', type=str)
    parser.add_argument('--experiment_name',default='test', type=str)
    parser.add_argument('--data_path',default='./data', type=str)
    parser.add_argument('--model_path',default='./model', type=str)
    parser.add_argument('--load_path',default='', type=str,help="set train or predict model path")
    parser.add_argument('--model_name_or_path',default='bert-base-uncased', type=str)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--batch_size", default=64, type=int,help="Batch size for training.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--dropout', type=float, default=0, help='Dropout.')
    parser.add_argument('--patience', type=int, default=20, help='Patience')
    parser.add_argument("--seed", type=int, default=88,
                        help="random seed for initialization")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")

    parser.add_argument("--eval_step", default=1200, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    args = parser.parse_args()
    
    # with open(f'{args.data_path}/train.json',mode='r',encoding='utf8') as f:
    #     train_datas=json.load(f)
    # with open(f'{args.data_path}/dev.json',mode='r',encoding='utf8') as f:
    #     dev_datas=json.load(f)
    # with open(f'{args.data_path}/test.json',mode='r',encoding='utf8') as f:
    #     test_datas=json.load(f)

    if args.wandb_use=='True':
        config = dict (
            learning_rate = args.learning_rate,
            epoch=args.num_train_epochs,
            dropout=args.dropout,
            batch_size=args.batch_size,
            )
        wandb.init(
            project="FactKG",
            name=args.experiment_name,
            notes="factkg-two",
            config=config,
            )


    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    with open('../data/relations_for_final.pickle', mode='rb') as f:
        relations = pickle.load(f)
    args.n_labels = len(relations)+1
    
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.n_labels,
    )
    # config.cls_token_id = tokenizer.cls_token_id
    config.unk_token_id = tokenizer.unk_token_id
    config.transformer_type = 'bert'

    logger.info('Load data!')
    if os.path.exists(f'{args.data_path}/train_feature.pkl'):
        with open(f'{args.data_path}/train_feature.pkl', mode='rb') as f:
            train_datas = pickle.load(f)
    else:
        train_datas=FactKG_Retrieve(args.data_path,'train',tokenizer)
        with open(f'{args.data_path}/train_feature.pkl', mode='wb') as f:
            pickle.dump(train_datas,f)

    if os.path.exists(f'{args.data_path}/dev_feature.pkl'):
        with open(f'{args.data_path}/dev_feature.pkl', mode='rb') as f:
            dev_datas = pickle.load(f)
    else:
        dev_datas=FactKG_Retrieve(args.data_path,'dev',tokenizer)
        with open(f'{args.data_path}/dev_feature.pkl', mode='wb') as f:
            pickle.dump(dev_datas,f)

    if os.path.exists(f'{args.data_path}/test_feature.pkl'):
        with open(f'{args.data_path}/test_feature.pkl', mode='rb') as f:
            test_datas = pickle.load(f)
    else:
        test_datas=FactKG_Retrieve(args.data_path,'test',tokenizer)
        with open(f'{args.data_path}/test_feature.pkl', mode='wb') as f:
            pickle.dump(test_datas,f)


    train_dataloader = torch.utils.data.DataLoader(train_datas,batch_size=args.batch_size,shuffle=True,num_workers=8,collate_fn=collate_fn)
    dev_dataloader = torch.utils.data.DataLoader(dev_datas,batch_size=args.batch_size,shuffle=False,num_workers=8,collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_datas,batch_size=args.batch_size,shuffle=False,num_workers=8,collate_fn=collate_fn)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    set_seed(args)
    model=REModel(config,model, num_labels=args.n_labels,dropout_prob=args.dropout)
    model.to(args.device)

    if args.load_path == "":  # Training
        train_model(args, model, train_dataloader, dev_dataloader, test_dataloader,dev_datas)
    else:  # Testing
        model.load_state_dict(torch.load(args.load_path)['model'])
        precision,recall,f1 = eval_report(model, test_dataloader,test_datas,'test')
        logger.info('Dev total precision: {0}, recall: {1}, f1: {2}, '.format(precision,recall,f1))


