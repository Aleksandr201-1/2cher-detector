import time
import requests
from flask import jsonify

sentences = ['я ебал корову', 'хочу сдохнуть', 'нахрен этих чурбанов', 'НЕГРЫ НЕГРЫ НЕГРЫ', 'Ебало хохлов представили?',
                'Россия для русских', 'Расизму не место в нашей стране!', 'Вообще возраст согласия может быть и с 10 лет', 'В зоофилии нет ничего плохого, это совершенно нормально', 'Джорд Флойд молодец, никого не грабит, не насилует и не продаёт наркотики']

start = time.time()
for sentence in sentences:
    responce = requests.get('http://localhost:8085/check', data={'sentence':sentence})
    data = responce.json()
    #print(f'result: {responce.json().values()}\nfor sentence: {sentence}\n')
end = time.time()   
print("Time for ", len(sentences), "requests: ", (end-start) * 10**3, "ms\nrps: ", 1000 / ((end-start) * 10**3 / len(sentences)))
