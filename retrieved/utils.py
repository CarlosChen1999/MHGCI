import torch
import random
import numpy as np

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    # TODO[unk]
    input_unk_ids = [torch.concat((torch.tensor(f["input_unk_ids"], dtype=torch.long),torch.zeros(len(f["input_unk_ids"]), max_len - len(f["input_unk_ids"][0]), dtype=torch.long)),dim=1) for f in batch]
    input_unk_ids = torch.cat(input_unk_ids,dim=0)
    input_unk_mask = [torch.concat((torch.ones_like(torch.tensor(f["input_unk_ids"], dtype=torch.float)),torch.zeros(len(f["input_unk_ids"]), max_len - len(f["input_unk_ids"][0]), dtype=torch.float)),dim=1) for f in batch]
    input_unk_mask=torch.cat(input_unk_mask, dim=0)

    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    hts = [f["hts"] for f in batch]
    # pos = [f["pos"] for f in batch]
    # for f in batch:
    #     torch.concat((torch.tensor(f["pos"]),torch.zeros(len(f["pos"]), max_len - len(f["pos"][0]))),dim=1)
    # 实体处理
    max_entity=max([len(x) for f in batch for x in f["entity"]])
    entity_list,entity_mask,entity_num=[],[],[]
    for ents in [f['entity'] for f in batch]:
        entity_num.append(len(ents))
        for x in ents:
            entity_list.append(x+[0]*(max_entity - len(x)))
            entity_mask.append([1.0] * len(x) + [0.0] * (max_entity - len(x)))
    entity_list = torch.tensor(entity_list, dtype=torch.long)
    entity_mask = torch.tensor(entity_mask, dtype=torch.float)

    pos = [torch.concat((torch.tensor(f["pos"]),torch.zeros(len(f["pos"]), max_len - len(f["pos"][0]))),dim=1) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask,pos, hts,entity_list,entity_mask,entity_num,input_unk_ids,input_unk_mask), labels
    return output