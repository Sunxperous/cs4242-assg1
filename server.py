import os
import sys
import time, pdb

from flask import Flask, render_template, request


app = Flask(__name__, static_url_path='')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    f_name, f_extension = os.path.splitext(f.filename)

    return render_template('index.html')


if __name__ == "__main__":
    print('Server started!')
    app.run(host='0.0.0.0', debug=True) # Debug true to auto-reload on code change.
