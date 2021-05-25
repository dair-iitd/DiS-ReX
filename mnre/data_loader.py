import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics

lang2id = {}
class BagREDataset(data.Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """

    def __init__(self, path, rel2id, tokenizer,num_lang = 4, entpair_as_bag=False, bag_size=0, mode= "Train"):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            entpair_as_bag: if True, bags are constructed based on same
                entity pairs instead of same relation facts (ignoring
                relation labels)
        """
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.num_lang = num_lang
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size
        self.id2rel = {}
        for k,v in self.rel2id.items():
            self.id2rel[v] = k
        
        
        # Load the file
        f = open(path)
        self.data = []
        for line in f:
            line = line.rstrip()

            if len(line) > 0:
                self.data.append(eval(line))

        f.close()

        # Construct bag-level dataset (a bag contains instances sharing the same relation fact)

        self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
        self.bag_scope = []
        self.name2id = {}
        self.bag_name = []
        self.facts = {}
        for idx, item in enumerate(self.data):
            fact = (item['h']['id'], item['t']['id'], item['relation'])

            if item["language"] not in lang2id:
                if mode == "Train":
                    lang2id[item["language"]] = len(lang2id)
                else:
                    print("Error... This language does not exist in training set. Exiting....")
                    exit(0)
            


            if item['relation'] != 'NA':
                self.facts[fact] = 1
            if entpair_as_bag:
                name = (item['h']['id'], item['t']['id'])
            else:
                name = fact
            if name not in self.name2id:
                self.name2id[name] = len(self.name2id)
                self.bag_scope.append({})
                self.bag_name.append(name)
            if lang2id[item["language"]] not in self.bag_scope[self.name2id[name]]:
                self.bag_scope[self.name2id[name]][lang2id[item["language"]]] = []
            self.bag_scope[self.name2id[name]][lang2id[item["language"]]].append(idx)
            self.weight[self.rel2id[item['relation']]] += 1.0
        self.weight = 1.0 / (self.weight ** 0.05)
        self.weight = torch.from_numpy(self.weight)

    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag_dict = self.bag_scope[index]

        bag = []
        lang_mask = []
        rel = -1
        for i in range(self.num_lang):
            lang_mask.append(0)
        for i in range(self.num_lang):
            if i in bag_dict:
                #print("TEST")
                bag_lang = bag_dict[i]
                if self.bag_size > 0:
                    if self.bag_size <= len(bag_lang):
                        resize_bag = random.sample(bag_lang, self.bag_size)
                    else:
                        resize_bag = bag_lang + list(np.random.choice(bag_lang, self.bag_size - len(bag_lang)))
                    bag_lang = resize_bag
                lang_mask[i] = 1
                rel = self.rel2id[self.data[bag_lang[0]]['relation']]
            else:
                bag_lang = []
                for j in range(self.bag_size):
                    bag_lang.append(0)
            bag.append(bag_lang)

        seqs = None
        if rel < 0 :
            print("ERROR .... relation is negative. Exiting...\n")
            exit(1)
        #print("Relation : ", rel)
        for bag_lang in bag:
            for sent_id in bag_lang:
                #print("Sentence id = ", sent_id)
                try:
                    item = self.data[sent_id]
                except:
                    print("ERROR...\n")
                    print(sent_id)
                seq = list(self.tokenizer(item))
                lang = item["language"]
                if seqs is None:
                    seqs = []
                    for i in range(len(seq)):
                        seqs.append([])
                for i in range(len(seq)):
                    seqs[i].append(seq[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)  # (n, L), n is the size of bag
            seqs[i] = torch.reshape(seqs[i], (self.num_lang,self.bag_size,-1))
        return [rel, self.bag_name[index], self.num_lang*self.bag_size,lang_mask] + seqs

    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count,lang_mask = data[:4]
        seqs = data[4:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)  # (sumn, L)
            seqs[i] = seqs[i].expand(
                (torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1,) + seqs[i].size())
        scope = []  # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert (start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long()  # (B)
        lang_mask = torch.tensor(lang_mask).float()
        return [label, bag_name, scope,lang_mask] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count,lang_mask = data[:4]
        seqs = data[4:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0)  # (batch, bag, L)
        scope = []  # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        #scope = torch.tensor(scope).long()
        label = torch.tensor(label).long()  # (B)
        lang_mask = torch.tensor(lang_mask).float() # (B)
        return [label, bag_name, scope, lang_mask] + seqs

    def eval(self, pred_result):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)
        P_10R = False  # To check if recall has reached 0.1
        P_30R = False  # To check if recall has reached 0.3
        p10_val = 0.0
        p30_val = 0.0
        for i, item in enumerate(sorted_pred_result):
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
            prec_temp = float(correct) / float(i + 1)
            prec.append(prec_temp)
            rec_temp = float(correct) / float(total)
            rec.append(rec_temp)
            if not P_10R:
                if rec_temp >= 0.1:
                    p10_val = prec_temp
                    P_10R = True
            if not P_30R:
                if rec_temp >= 0.3:
                    p30_val = prec_temp
                    P_30R = True
        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec)
        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        return {'micro_p': np_prec, 'micro_r': np_rec, 'micro_p_mean': mean_prec, 'micro_f1': f1, 'auc': auc,
                'p@10': p10_val, 'p@30': p30_val}


def BagRELoader(path, rel2id, tokenizer, batch_size,
                shuffle, num_lang = 4, entpair_as_bag=False, bag_size=0, num_workers=8, mode = "Train",
                collate_fn=BagREDataset.collate_fn):
    if bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path, rel2id, tokenizer, num_lang = num_lang, entpair_as_bag=entpair_as_bag, bag_size=bag_size, mode = mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
