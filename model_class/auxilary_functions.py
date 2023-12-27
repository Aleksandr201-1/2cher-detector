import xlrd

import torch
import numpy as np
from tqdm import tqdm
import copy
import time

def load_dset_from_file(filename):
    rb = xlrd.open_workbook(filename)
    #выбираем активный лист

    sheet = rb.sheet_by_index(0)

    datavals = []
    datalabels = []


    #формируем трансляторы
    labels = sheet.row_values(0)[2:]
    LabtoNum = {}
    NumToLab = {}

    cnt = 0
    for elm in labels:
        if elm == '':
            break
        LabtoNum[elm] = cnt
        NumToLab[cnt] = elm
        cnt += 1

    #выгружаем данные
    for index in range(1, sheet.nrows):
        data = sheet.row_values(index)[1]
        labels = sheet.row_values(index)[2:]
        
        if labels[19] == '':
            datavals.append(data)
            datalabels.append(torch.tensor([float(l) if l != '' else 0.0 for l in labels[0:19]]))
    return datavals, datalabels, LabtoNum, NumToLab

def test(model, testloader, criterion, verbal = True, rec_loss = False):
    #was_training = copy.deepcopy(model.model.training)
    #model.model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    if(verbal):
        iterator = tqdm(testloader)
    else:
        iterator = testloader
        
    for inputs, labels in iterator:
        inputs = inputs
        labels = labels
        
        with torch.no_grad():
            inp = model.tokenizer(inputs, return_tensors = "pt", padding=True)
            inp = inp.to(model.device)
            outputs = model.model(**inp)
            
        res = outputs.logits
        pred = torch.argmax(res, dim = 1)
        
        loss = criterion(res, labels.to(model.device))
        running_loss += loss.item()
        running_corrects += sum(pred.to("cpu") == torch.argmax(labels.to("cpu"), dim = 1))
        
    ovl_loss = running_loss / len(testloader)
    ovl_acc = float(running_corrects) / (len(testloader) * testloader.batch_size)

    if(verbal):
        print('test | loss function val: {} accuracy: {}'.format(ovl_loss, ovl_acc))
    
    #if(was_training):
    #    model.model.train()
        
    if(rec_loss):
        return ovl_acc, ovl_loss
    return ovl_acc

def train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=25,
                verbal = True, bert_dropout = True):
    
    model.model.classifier.train()
    
    if(bert_dropout):
        model.model.bert.train()
    else:
        model.model.bert.eval()
    #--
    
    tracing = []
    since = time.time()

    best_model_wts = copy.deepcopy(model.model.state_dict())
    best_acc = 0.0
    
    if(verbal):
        iterator_1 = range(num_epochs)
    else:
        iterator_1 = tqdm(range(num_epochs))
            
    #--
    model.model.bert.eval()
    model.model.classifier.eval()
    
    train_accurcy, train_loss = test(model,trainloader,criterion, verbal = verbal, rec_loss = True)
    validation_accurcy, validation_loss = test(model,valloader,criterion, verbal = verbal, rec_loss = True)
    
    if(bert_dropout):
        model.model.bert.train()
        model.model.classifier.train()
    else:
        model.model.bert.eval()
        model.model.classifier.train()
    #--
        
    tracing.append([-1,train_loss,train_accurcy,validation_loss, validation_accurcy])
    
    for epoch in iterator_1:
        torch.cuda.empty_cache()

        running_loss = 0.0
        running_corrects = 0
        
        if(verbal):
            iterator = tqdm(trainloader)
        else:
            iterator = trainloader
            
        for inputs, labels in iterator:
            inputs = inputs
            labels = labels
            
            #--------
            optimizer.zero_grad()
            #--------

            inp = model.tokenizer(inputs, return_tensors = "pt", padding=True)
            inp = inp.to(model.device)
            outputs = model.model(**inp)
            
            res = outputs.logits
            pred = torch.argmax(res, dim = 1)

            loss = criterion(res, labels.to(model.device))
            
            #обучение
            loss.backward()
            optimizer.step()
            #--------
            
            #трейсинг
            running_loss += loss.item()
            running_corrects += sum(pred.to("cpu") == torch.argmax(labels.to("cpu"), dim = 1))

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = float(running_corrects) / (len(trainloader) * trainloader.batch_size)

        if(verbal):
            print('epoch {} | loss function val: {} accuracy: {}'.format(epoch, epoch_loss, epoch_acc))

        #--
        model.model.bert.eval()
        model.model.classifier.eval()
        
        validation_accurcy, validation_loss = test(model,valloader,criterion, verbal = verbal, rec_loss = True)
        
        if(bert_dropout):
            model.model.bert.train()
            model.model.classifier.train()
        else:
            model.model.bert.eval()
            model.model.classifier.train()
        #--
        
        tracing.append([epoch,epoch_loss,epoch_acc, validation_loss, validation_accurcy])
        
        if validation_accurcy > best_acc:
            del best_model_wts
            torch.cuda.empty_cache()
            best_acc = validation_accurcy
            best_model_wts = copy.deepcopy(model.model.state_dict())

        if(verbal):
            print()

    time_elapsed = time.time() - since
    
    #if(verbal):
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.model.load_state_dict(best_model_wts)
    
    #--
    model.model.classifier.eval()
    model.model.bert.eval()
    
    return model, np.array(tracing)

def predict_labels(classifier,dataloader,verbal = True, scores = False):
    pred_labels = []

    if(verbal):
        iterator = tqdm(dataloader)
    else:
        iterator = dataloader
        
    for inputs, labels in iterator:
        inputs = inputs
        labels = labels

        #numlabel = torch.tensor(classifier.model.config.label2id[labels]).to(device)
        
        inp = classifier.tokenizer(inputs, return_tensors = "pt", padding=True)
        inp = inp.to(model.device)
        outputs = classifier.model(**inp)

        res = outputs.logits
        
        if(not scores):
            # не работает
            pred = int(torch.argmax(res, dim = 1))        
            pred_labels += (classifier.model.config.id2label[pred])
            
        else:
            pred = outputs.logits
            pred_labels.append(pred.detach().to('cpu').numpy())
            
        
    return pred_labels