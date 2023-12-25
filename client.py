from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route('/main/', methods=['post', 'get'])
def check():
    list_of_hate = []
    sentence = ''
    responce = requests.get('http://localhost:8085/check', data={'sentence':sentence})
    if request.method == 'POST':
        sentence = request.form.get('sentence')
        responce = requests.get('http://localhost:8085/check', data={'sentence':sentence})
    
    for val in responce.json().values():
        for v in val:
            list_of_hate.append([v[0], round(v[1], 2)])

    return render_template('check.html', list_of_hate=list_of_hate)


if __name__ == "__main__":
    app.run(debug=False, port = 8086)