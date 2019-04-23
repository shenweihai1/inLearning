import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from torch.utils.data import Dataset
class quarel(Dataset):
    def __init__(self,which='train'):
        self.sample=[]
        if which=='train':
            f1 = open('./data/corpus_train_true.txt')
            g1 = f1.readlines()
            f1.close()
            f2 = open('./data/corpus_train_false.txt')
            g2 = f2.readlines()
            f2.close()
            for i in range(len(g1) // 3):
                self.sample.append(((g1[i*3].strip(),g1[i*3+1].strip()),1.0))
            for i in range(len(g2)//3):
                self.sample.append(((g2[i * 3].strip(), g2[i * 3 + 1].strip()), -1.0))
        elif which=='test':
            f1 = open('./data/corpus_test_true.txt')
            g1 = f1.readlines()
            f1.close()
            f2 = open('./data/corpus_test_false.txt')
            g2 = f2.readlines()
            f2.close()
            for i in range(len(g1) // 3):
                self.sample.append(((g1[i * 3].strip(), g1[i * 3 + 1].strip()), 1.0))
            for i in range(len(g2) // 3):
                self.sample.append(((g2[i * 3].strip(), g2[i * 3 + 1].strip()), -1.0))
        else:
            f1 = open('./data/corpus_dev_true.txt')
            g1 = f1.readlines()
            f1.close()
            f2 = open('./data/corpus_dev_false.txt')
            g2 = f2.readlines()
            f2.close()
            for i in range(len(g1) // 3):
                self.sample.append(((g1[i * 3].strip(), g1[i * 3 + 1].strip()), 1.0))
            for i in range(len(g2) // 3):
                self.sample.append(((g2[i * 3].strip(), g2[i * 3 + 1].strip()), -1.0))
    def __getitem__(self,index):
        return self.sample[index]
    def __len__(self):
        return len(self.sample)

par='dev'

trainloader = torch.utils.data.DataLoader(quarel(which='train'), batch_size=1, shuffle=True,num_workers=10)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load('./model_data/pretrained_hacked.model'))
model.eval()
for param in model.parameters():
    param.requires_grad=False
model=model.cuda()
results=[]
for x,y in iter(trainloader):
    tokenized_text_a=tokenizer.tokenize(x[0][0])
    tokenized_text_b = tokenizer.tokenize(x[1][0])
    indexed_tokens_a = tokenizer.convert_tokens_to_ids(tokenized_text_a)
    indexed_tokens_b = tokenizer.convert_tokens_to_ids(tokenized_text_b)
    indexed_tokens=indexed_tokens_a+indexed_tokens_b
    segments_ids=[0 for _ in range(len(indexed_tokens_a))]+[1 for _ in range(len(indexed_tokens_b))]
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    segments_tensors = torch.tensor([segments_ids]).cuda()
    encoded_layers, _ = model(tokens_tensor, segments_tensors)
    results.append((encoded_layers[-1].cpu(),y))
torch.save(results,'./data/'+par+'.data')

