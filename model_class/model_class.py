from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import sklearn as skl
import numpy as np
import torch

from auxilary_functions import *

class MyDataset:
    def __init__(self,datavals,datatags, shuffle = False):            
        self.datavals = datavals
        self.datatags = datatags
        
        if(shuffle):
            self.datavals, self.datatags = skl.utils.shuffle(self.datavals, self.datatags)
            
    def __len__(self):
        return len(self.datavals)
    
    def __getitem__(self,idx):
        return self.datavals[idx], self.datatags[idx]
    
LabtoNum = {'offline_crime': 0, 'online_crime': 1,
 'drugs': 2, 'gambling': 3, 'pornography': 4,
 'prostitution': 5, 'slavery': 6, 'suicide': 7,
 'terrorism': 8, 'weapons': 9, 'body_shaming': 10,
 'health_shaming': 11, 'politics': 12, 'racism': 13,
 'religion': 14, 'sexual_minorities': 15, 'sexism': 16, 'social_injustice': 17}

NumToLab = {0: 'offline_crime', 1: 'online_crime',
 2: 'drugs', 3: 'gambling', 4: 'pornography',
 5: 'prostitution', 6: 'slavery', 7: 'suicide',
 8: 'terrorism', 9: 'weapons', 10: 'body_shaming',
 11: 'health_shaming', 12: 'politics', 13: 'racism',
 14: 'religion', 15: 'sexual_minorities', 16: 'sexism', 17: 'social_injustice'}

class hdm_model():

    def __init__(self, base_model_path = "DeepPavlov/rubert-base-cased-conversational"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # загрузка модели
        # при загрузке устанавливаются трансляторы и размеры по умолчанию
        # при загрузке датасета голова переписывается в любом случае
        model = BertForSequenceClassification.from_pretrained(base_model_path, num_labels = 18, 
                                                              problem_type="multi_label_classification")
        tokenizer=BertTokenizer.from_pretrained(base_model_path, do_lower_case=False)

        self.LabtoNum = LabtoNum
        self.NumToLab = NumToLab
        
        for param in model.bert.parameters():
            param.requires_grad = False 
        self.classifier = pipeline("text-classification", model=model,tokenizer=tokenizer, device = self.device)
        self.classifier.model.bert.eval()
        self.sm = torch.nn.Softmax(dim = -1)

    def load_data(self, data_path = "sencitive_topics.xls", batch_size = 4, ratio = 0.8, ):
        datavals, datalabels, self.LabtoNum, self.NumToLab = load_dset_from_file(data_path)

        data = MyDataset(datavals, datalabels, shuffle = True)
        train, valid = torch.utils.data.random_split(data,[ratio, 1 - ratio])
        # сохранение лоадеров
        self.trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size)
        self.validloader = torch.utils.data.DataLoader(valid, batch_size = batch_size)
        # введение в пайплайн инфо о классах и переписыание головы под размер датасета
        self.classifier.model.config.id2label = NumToLab
        self.classifier.model.config.label2id = LabtoNum
        self.LabtoNum = LabtoNum
        self.NumToLab = NumToLab
        
        self.classifier.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.classifier.model.bert.config.hidden_size, 
                            len(LabtoNum.values()))
        )

        self.wts = torch.tensor(class_weight.compute_class_weight(class_weight = 'balanced',
                                classes = list(LabtoNum.keys()), 
                                y = np.array([NumToLab[idx] for elm in train for idx, val in enumerate(elm[1]) if val != 0 ])))
        
    def fit(self, Criterion = torch.nn.CrossEntropyLoss, 
                  Optimizer = torch.optim.Adadelta,
                  n_epoch = 20):
        criterion = Criterion(weight=self.wts.to(self.device))
        optimizer = Optimizer(self.classifier.model.parameters(), lr=0.8, weight_decay = 0)
        
        self.classifier, self.trace = train_model(self.classifier, 
                                                  self.trainloader, 
                                                  self.validloader, criterion, optimizer, num_epochs=n_epoch, bert_dropout = True)

        return self.trace

    def predict(self, obj):
        inp = self.classifier.tokenizer(obj, return_tensors = "pt", padding=True)
        inp = inp.to(self.classifier.device)
        outputs = self.classifier.model(**inp)
        res = outputs.logits.cpu()[0]
        res = self.sm(res)
        result_dict = {elm : float(res[num]) for num, elm in enumerate(self.LabtoNum.keys())}

        return result_dict
    
    def load(self, model_path, only_head = False):  
        if only_head:
            self.classifier.model.classifier.load_state_dict(torch.load(model_path))
        else:
            self.classifier.model.load_state_dict(torch.load(model_path))

    def save(self, model_path, only_head = False):
        if only_head:
            torch.save(self.classifier.model.classifier.state_dict(), model_path)
        else:
            torch.save(self.classifier.model.state_dict(), model_path)

    def evaluate(self):
        predictions = predict_labels(self.classifier, self.validloader, scores = True)

        ConfusionMatrixDisplay.from_predictions(torch.flatten(torch.tensor([list(torch.argmax(lbl, dim = 1)) for _, lbl in self.validloader])),
                                        torch.flatten(torch.argmax(torch.tensor(predictions), dim = 2)), 
                                        display_labels = list(self.LabtoNum.keys()), include_values = False,
                                        cmap = 'Blues', normalize = 'true', xticks_rotation = 'vertical')

        print(classification_report(torch.flatten(torch.tensor([list(torch.argmax(lbl, dim = 1)) for _, lbl in self.validloader])),
                                    torch.flatten(torch.argmax(torch.tensor(predictions), dim = 2)), 
                                    target_names = list(self.LabtoNum.keys())))
        return predictions
    
if __name__ == '__main__':
    model = hdm_model()
    model.load("models/_bs_4_do_ON_wts_ON_100_epo_adadelta")
    pred = model.predict('Ненавижу нигеров')
    print([f"{elm[0]}     ({elm[1]:.2f})" for elm in pred.items() if elm[1] > 0.5])
    pred = model.predict('Привет! Как дела?')
    print([f"{elm[0]}     ({elm[1]:.2f})" for elm in pred.items() if elm[1] > 0.5])