from flask import Flask, abort, jsonify, request, render_template
import json
import os
import io
import pandas as pd
from learning_path import Learning_Path
app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='templates')
# refers to application_top
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static')
data = pd.read_csv(APP_STATIC+'/sample_dataset_learnavi.csv')
source = data.source
target = data.target
node_all = source.append(target)
node_name = data.source_name.append(data.target_name)
dicti = source.append(target)
dict_name = data.source_name.append(data.target_name)
node_dict = dict(zip(dict_name, dicti))
edge_dict = dict(zip(data.edge_name, data.edge))
model = Learning_Path()


@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def home():

    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
@app.route('/all', methods=['POST', 'GET'])
def contoh():

    return render_template('index_all.html')


@app.route('/', methods=['POST', 'GET'])
@app.route('/prediksi', methods=['POST', 'GET'])
def predict():
    src = request.args.get('src')
    rel = request.args.get('rel')
    print(src, rel)
    src = node_dict[src]
    rel = edge_dict[rel]
    search = data[(data.source == src) & (data.edge == rel)].reset_index(drop=True)
    dest_name = set(search.target_name)
    dest = set(search.target)
    result = []
    for i in dest:
        x = [src, rel, i]
        print(x)
        x = model.predict(x)
        print(x)
        result.append(x)
    output = sorted(set(zip(dest_name, result)), reverse=True)
    return jsonify(list(output))


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    app.run()
