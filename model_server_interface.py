from flask import Flask, request, jsonify
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, pipeline
import torch
import datetime

app = Flask(__name__)

# загрузка модели
#-----------------------------------------------------------------
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased-conversational', num_labels = len(LabtoNum.values()), problem_type="multi_label_classification")
tokenizer=BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-conversational', do_lower_case=False)
# model.classifier = torch.nn.Sequential(
#     torch.nn.Linear(768, len(LabtoNum.values())),
#     torch.nn.Softmax(dim = -1)
# )
sm = torch.nn.Softmax(dim = -1)
for param in model.bert.parameters():
    param.requires_grad = False
model.config.id2label = NumToLab
model.config.label2id = LabtoNum
classifier = pipeline("text-classification", model=model,tokenizer=tokenizer, device = device)
classifier.model.load_state_dict(torch.load('models/_bs_4_do_ON_wts_ON_25_epo_adam_1e-4_0.57'))
classifier.model.eval()
#-----------------------------------------------------------------

@app.route('/ping', methods=['POST', 'GET'])
def hello():
    print('Hello')
    print(request.method)
    #print(request.json)
    return jsonify({'Hello' : 'Hello'})

@app.route('/check', methods=['GET'])
def check():
    sentence = request.form.get('sentence')
    user_id = request.form.get('user_id')
    date = datetime.datetime.now()
    with torch.no_grad():
        inp = classifier.tokenizer(sentence, return_tensors = "pt", padding=True)
        inp = inp.to(device)
        outputs = classifier.model(**inp)
        res = outputs.logits.cpu()[0]
        res = sm(res)
        result_dict = {elm : float(res[num]) for num, elm in enumerate(LabtoNum.keys())}
        sorted_results = sorted(result_dict.items(), key = lambda x : x[1], reverse = True)
        str_date = str(date)
        logs = {'user_id' : user_id, 'date' : str_date[11:], 'sentence' : sentence, 'probs' : sorted_results}
        with open('logs/logs-' + str_date[0:10], 'a') as logfile:
            logfile.write(str(logs) + '\n')

        return jsonify({'probs' : sorted_results})

if __name__ == "__main__":
    app.run(debug=False, port = 8085)