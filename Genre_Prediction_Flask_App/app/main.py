from os.path import join, dirname
from . import predictor

from flask import Flask, render_template, Response, request
import jsonpickle
dirname = dirname(__file__)

# start flask
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 #10 MB

# render default webpage
@app.route('/')
def home():
    return render_template('index.html', title="Genre Prediction")

@app.route('/predict_with_db', methods=['POST'])
def predictWithDb():
    if request.method == 'POST':
        save_path = join(dirname, "temp.wav")
        request.files['music_file'].save(save_path)
        print("File written")

        result_json = None
        try:
            result_json = predictor.predictGenreWithModel(save_path)
        except Exception as e:
            print(e)
            return Response(response={'status': 'Predction Problem! Check music file, maybe.'}, status=503, mimetype="application/json")

        response_pickled = jsonpickle.encode(result_json)
        return Response(response=response_pickled, status=200, mimetype="application/json")

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File Too Large (Must be less than 10MB)', 413

@app.errorhandler(503)
def request_model_not_responding(error):
    return 'Model is Not responding', 503
