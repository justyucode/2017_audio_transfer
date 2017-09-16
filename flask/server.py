import flask
from flask import Flask, request
import requests

react_url = "TODO"
example_wav_file_path = "TODO"

@app.route('/', methods=['GET', 'POST'])
def index():
    string = "hello world!"

    files = {'file': open(example_wav_file_path, 'rb')}
    requests.post(react_url, files=files)
    return string

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
