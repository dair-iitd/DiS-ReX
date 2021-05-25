import torch
from torch import nn, optim
from torch.nn import functional as F
class LanguageFilter(nn.Module):
    def __init__(self,num_lang,embed_dim,ffn_dim,mem_embed_dim,num_layers,num_heads):
        super().__init__()
        self.convert2emb = nn.Parameter(torch.empty(num_lang,mem_embed_dim))
        nn.init.xavier_normal_(self.convert2emb)
        self.embed_dim = embed_dim
        self.num_lang = num_lang
        self.mem_embed_dim = mem_embed_dim
        self.write_mha = nn.MultiheadAttention(mem_embed_dim, num_heads,kdim=embed_dim,vdim=embed_dim,dropout=0.5)
        self.fc = nn.Sequential(nn.Linear(embed_dim,ffn_dim),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(ffn_dim,embed_dim),nn.Dropout(p=0.5))
        self.read_mha = nn.MultiheadAttention(embed_dim, num_heads,kdim=mem_embed_dim,vdim=mem_embed_dim,dropout=0.5)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.num_layers = num_layers
        self.gru = nn.GRUCell(mem_embed_dim,mem_embed_dim)
    
    def forward(self,bag,language_mask):
        bsz,_ = language_mask.shape
        # bag--> bsz, bag_size, self.embed_dim
        # language_mask--> bsz,self.num_lang (a float)
        # memory  = self.convert2emb(language_mask)
        memory = language_mask.unsqueeze(2)*self.convert2emb.unsqueeze(0)
        # memory-->bsz,mem_embed_dim
        #memory =  memory.reshape(bsz,self.num_lang,self.mem_embed_dim)
        memory = memory.permute(1,0,2) # num_lang, bsz, mem_embed_dim
        bag = bag.permute(1,0,2) # bag_size, bsz, embed_dim
        for layers in range(self.num_layers):
            residual = bag
            bag = self.layer_norm(bag)
            # print(bag.shape,memory.shape)
            write_value,_ = self.write_mha(query=memory,key=bag,value=bag) #num_lang, bsz, mem_embed_dim
            memory = self.gru(write_value.reshape(self.num_lang*bsz,self.mem_embed_dim) , memory.reshape(self.num_lang*bsz,self.mem_embed_dim))
            memory = memory.reshape(self.num_lang,bsz,self.mem_embed_dim)
            read_value,_ = self.read_mha(query=bag,key=memory,value=memory,key_padding_mask=language_mask.bool()) # bag_size, bsz, embed_dim
            read_value = F.dropout(read_value, p=0.5, training=self.training)
            bag = residual+read_value
            residual = bag 
            bag = self.fc(bag)+residual
        return bag.permute(1,0,2)


''' example use
bag = torch.randn(32,2,128)
language_mask =  torch.empty(32,4).random_(2)

lf = LanguageFilter(4,embed_dim=128,ffn_dim=256,mem_embed_dim=64,num_layers=4,num_heads=4).to(device)
lf(bag.to(device),language_mask.to(device)).shape --> 32, 2, 128

'''





class IntraBagAttention(nn.Module):
    """
    Instance attention for bag-level relation extraction.
    """

    def __init__(self, 
                sentence_encoder, 
                num_class, 
                rel2id, 
                num_languages = 4,
                mem_embed_dim = 64,
                lang_filter_layers = 4,
                lang_filter_heads = 4):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.embed_dim = self.sentence_encoder.hidden_size
        #self.lang_filter = LanguageFilter(num_languages,self.embed_dim,2*self.embed_dim,mem_embed_dim,lang_filter_layers, lang_filter_heads)
        self.num_class = num_class
        self.num_lang = num_languages
        self.fc = nn.Linear(self.embed_dim, num_class)
        self.language_relation_embedding = nn.Parameter(torch.empty(self.num_class,num_languages,self.embed_dim))
        nn.init.xavier_normal_(self.language_relation_embedding)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ...,
                  'h': {'pos': [start, end], ...},
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        self.eval()
        tokens = []
        pos1s = []
        pos2s = []
        masks = []
        for item in bag:
            token, pos1, pos2, mask = self.sentence_encoder.tokenize(item)
            tokens.append(token)
            pos1s.append(pos1)
            pos2s.append(pos2)
            masks.append(mask)
        tokens = torch.cat(tokens, 0).unsqueeze(0)  # (n, L)
        pos1s = torch.cat(pos1s, 0).unsqueeze(0)
        pos2s = torch.cat(pos2s, 0).unsqueeze(0)
        masks = torch.cat(masks, 0).unsqueeze(0)
        scope = torch.tensor([[0, len(bag)]]).long()  # (1, 2)
        bag_logits = self.forward(None, scope, tokens, pos1s, pos2s, masks, train=False).squeeze(0)  # (N) after softmax
        score, pred = bag_logits.max(0)
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)

    def forward(self, label, scope, lang_mask, token, pos1, pos2, mask=None, train=True, bag_size=0):
        """
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            lang_mask : (B,)
            token: (nsum, L), index of tokens
            pos1: (nsum, L), relative position to head entity
            pos2: (nsum, L), relative position to tail entity
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)
        """
        if bag_size > 0:
            token = token.view(-1, token.size(-1))
            pos1 = pos1.view(-1, pos1.size(-1))
            pos2 = pos2.view(-1, pos2.size(-1))
            if mask is not None:
                mask = mask.view(-1, mask.size(-1))
        else:
            begin, end = scope[0][0], scope[-1][1]
            token = token[:, begin:end, :].view(-1, token.size(-1))
            pos1 = pos1[:, begin:end, :].view(-1, pos1.size(-1))
            pos2 = pos2[:, begin:end, :].view(-1, pos2.size(-1))
            if mask is not None:
                mask = mask[:, begin:end, :].view(-1, mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))
        if mask is not None:
            rep = self.sentence_encoder(token, pos1, pos2, mask)  # (nsum, H)
        else:
            rep = self.sentence_encoder(token, pos1, pos2)  # (nsum, H)

        # Attention
        if train:
            if bag_size == 0:
                bag_rep = []
                query = torch.zeros((rep.size(0))).long()
                if torch.cuda.is_available():
                    query = query.cuda()
                for i in range(len(scope)):
                    query[scope[i][0]:scope[i][1]] = label[i]
                att_mat = self.fc.weight.data[query]  # (nsum, H)
                att_score = (rep * att_mat).sum(-1)  # (nsum)

                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]]  # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]])  # (n)
                    bag_rep.append(
                        (softmax_att_score.unsqueeze(-1) * bag_mat).sum(0))  # (n, 1) * (n, H) -> (n, H) -> (H)
                bag_rep = torch.stack(bag_rep, 0)  # (B, H)
            else:
                batch_size = label.size(0)
                query = label.unsqueeze(1)  # (B, 1)
                att_mat = self.language_relation_embedding[query]  # (B, 1, numlang, H)
                att_mat = att_mat.unsqueeze(3) # (B, 1, numlang,1, H)
                # print("Size of attention mat is {}".format(att_mat.size()))
                rep = rep.view(batch_size, self.num_lang, 1, bag_size, -1) # (B, numlang, 1, bag,H)
                #rep = self.lang_filter(rep,lang_mask)
                att_score = (rep * att_mat).sum(-1)  # (B, numlang, numlang, bag)
                # print("Shape of attention score is {}".format(att_score.size()))
                softmax_att_score = self.softmax(att_score)  # (B, numlang, numlang, bag)
                bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(
                    3)  # (B, numlang,numlang,bag,1) * (B, numlang,1, bag, H) -> (B,numlang,numlang, bag, H) -> (B,num_lang, num_lang, H)
            bag_rep = self.drop(bag_rep)
            lang_rel_embedding = self.language_relation_embedding.permute(1,2,0) #(num_lang, H , N)
            bag_logits = self.fc(bag_rep) #(B,num_lang,num_lang, N)

            
            bag_rep = bag_rep.unsqueeze(3) #(B,num_lang, num_lang, 1, H)
            score2 = torch.matmul(bag_rep,lang_rel_embedding).squeeze(3) # #(B,num_lang, num_lang, 1, H) * (num_lang,H,N) -> (B,num_lang,num_lang,1, N) -> (B,num_lang,num_lang, N)
            # print("Shape of bag logits is {}".format(bag_logits.size()))
            # print("Shape of score2 is {}".format(score2.size()))
            bag_logits += score2

            lang_mask = lang_mask.reshape(batch_size,self.num_lang,1,1) # (B, numlang,1,1)
            bag_logits = (bag_logits*lang_mask).sum((1,2)) # (B, N)
            
        else:
            if bag_size == 0:
                bag_logits = []
                att_score = torch.matmul(rep, self.fc.weight.data.transpose(0, 1))  # (nsum, H) * (H, N) -> (nsum, N)
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]]  # (n, H)
                    softmax_att_score = self.softmax(
                        att_score[scope[i][0]:scope[i][1]].transpose(0, 1))  # (N, (softmax)n)
                    rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat)  # (N, n) * (n, H) -> (N, H)
                    logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel))  # ((each rel)N, (logit)N)
                    logit_for_each_rel = logit_for_each_rel.diag()  # (N)
                    bag_logits.append(logit_for_each_rel)
                bag_logits = torch.stack(bag_logits, 0)  # after **softmax**
            else:
                batch_size = rep.size(0) // (bag_size*self.num_lang)
                rep = rep.view(-1, self.embed_dim)
                att_score = torch.matmul(rep, self.language_relation_embedding.permute(2,1,0).reshape(self.embed_dim,-1))  # (B*num_lang*bag, H) * (H, numlang*N) -> (B*nsum*bag,num_lang*N)
                att_score = att_score.view(batch_size, self.num_lang,  bag_size,-1)  # (B, numlang, bag,numlang*N)
                rep = rep.view(batch_size, -1,bag_size, self.embed_dim)  # (B, numlang, bag, H)
                #rep = self.lang_filter(rep,lang_mask)
                softmax_att_score = self.softmax(att_score.transpose(3,2))  # (B, numlang, numlang*N, (softmax)bag)
                rep_for_each_rel = torch.matmul(softmax_att_score, rep)  # (B, numlang, numlang*N, bag) * (B,numlang,bag, H) -> (B,numlang,numlang*N, H)

                rep_for_each_rel = rep_for_each_rel.view(batch_size,self.num_lang,self.num_lang,-1,self.embed_dim) # (B, numlang, numlang, N, H)
                lang_rel_embedding = self.language_relation_embedding.permute(1,2,0)
                score1 = self.fc(rep_for_each_rel)  # (B,numlang,numlang, N, N)
                score2 = torch.matmul(rep_for_each_rel,lang_rel_embedding) # (B, numlang, numlang, N, H) * (numlang, H, N) -> (B, numlang, numlang, N, N)
                bag_logits = (score1 + score2) # (B, numlang,numlang, N,N)
                lang_mask = lang_mask.reshape(batch_size,self.num_lang,1,1,1) # (B, numlang,1,1,1)
                bag_logits = (bag_logits*lang_mask).sum((1,2)) # (B, N, N)
                bag_logits = self.softmax(bag_logits).diagonal(dim1=1, dim2=2)  # (B, N)

        
        return bag_logits
