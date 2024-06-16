import torch
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits >= th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

class REModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1 , dropout_prob=0, num_heads=8):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        # 注意力机制
        self.num_heads=num_heads
        self.linear_query = nn.Linear(emb_size, num_heads * emb_size)
        self.linear_key = nn.Linear(emb_size, num_heads * emb_size)
        self.linear_value = nn.Linear(emb_size, num_heads * emb_size)
        self.linear_final = nn.Linear(num_heads * emb_size, emb_size)

        # vae编码器
        self.linear_vae = nn.Linear(emb_size, num_heads * emb_size)
        self.fc1 = nn.Linear(config.hidden_size, 400)
        self.fc21 = nn.Linear(400, 20)  # mean
        self.fc22 = nn.Linear(400, 20)  # logvar
        # vae解码器
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, config.hidden_size)

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        # self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.dropout=nn.Dropout(dropout_prob)

    def encode(self, input_ids, attention_mask):
        output=self.model(input_ids=input_ids,attention_mask=attention_mask,output_attentions=True)
        sequence_output = output[0]
        attention = output[-1][-1]
        return sequence_output, attention


    def vae_loss_function(self,recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x)
        return BCE
    
    # DAEG
    def get_hrt_unk(self, sequence_output, attention,unk_sequence_output,unk_attention, entity_pos, hts):
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        tss_vae,mu_vae,logvar_vae=[],[],[]

        for i in range(len(entity_pos)): # i is batch
            entity_embs, entity_atts = [], []
            entity_unk_embs, entity_unk_atts = [], []
            k=0
            for e_pos in entity_pos[i]:
                interested_indices = torch.nonzero(e_pos == 1).squeeze(1)
                if interested_indices.numel() > 0:
                    e_emb = sequence_output[i][interested_indices].mean(dim=0)# embedding均值
                    e_att = attention[i,:,interested_indices].mean(dim=1)# attention均值
                    entity_embs.append(e_emb)
                    entity_atts.append(e_att)

                    e_emb_unk = unk_sequence_output[i][k][interested_indices].mean(dim=0)# embedding均值
                    e_att_unk = unk_attention[i][k,:,interested_indices].mean(dim=1)# attention均值
                    entity_unk_embs.append(e_emb_unk)
                    entity_unk_atts.append(e_att_unk)
                else:
                    print("error:entity not found!")
                k+=1

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            entity_unk_embs = torch.stack(entity_unk_embs, dim=0)  # [n_e, d]
            entity_unk_atts = torch.stack(entity_unk_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.tensor(hts[i], dtype=torch.long).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])# ent,emb
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            ts_unk = torch.index_select(entity_unk_embs, 0, ht_i[:, 1])

            # 注意力机制
            batch_size = hs.size(0)
            query = F.relu(self.linear_query(ts-ts_unk)).view(batch_size, -1, self.num_heads, self.emb_size).transpose(1, 2)
            key = F.relu(self.linear_key(hs)).view(batch_size, -1, self.num_heads, self.emb_size).transpose(1, 2)
            value = F.relu(self.linear_value(ts-ts_unk)).view(batch_size, -1, self.num_heads, self.emb_size).transpose(1, 2)

            scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.emb_size, dtype=torch.float32))

            attention_weights = F.softmax(scores, dim=-1)
            weighted_sum = torch.matmul(attention_weights, value)+torch.matmul(1-attention_weights,self.random_noise(value))/ torch.sqrt(torch.tensor(self.emb_size, dtype=torch.float32))
            weighted_sum = weighted_sum.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.emb_size)#合并多头

            output = F.relu(self.linear_final(weighted_sum))+ts_unk
            ts_change=torch.squeeze(output,1)

            hss.append(hs)
            tss.append(ts)
            # rss.append(rs)
            tss_vae.append(ts_change)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        # rss = torch.cat(rss, dim=0)

        tss_vae = torch.cat(tss_vae, dim=0)
        return hss, rss, tss ,tss_vae,mu_vae,logvar_vae
    
    

    def forward(self,
                input_ids=None,
                attention_mask=None,
                pos=None,
                hts=None,
                entity=None,
                entity_mask=None,
                entity_num=None,
                input_unk_ids=None,
                input_unk_mask=None,
                labels=None
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        unk_sequence_output, unk_attention = self.encode(input_unk_ids, input_unk_mask)
        unk_sequence_output = torch.split(unk_sequence_output, entity_num, dim=0)

        hs, rs, ts ,tss_vae,mu_vae,logvar_vae= self.get_hrt_unk(sequence_output, attention,unk_sequence_output,unk_attention, pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, tss_vae], dim=1)))

        bl=hs
        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output