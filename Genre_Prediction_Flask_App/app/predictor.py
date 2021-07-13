from os.path import join, dirname
import pickle, librosa
import numpy as np

dirname = dirname(__file__)

min_max_scaler_path = join(dirname, '../model/min_max_scaler.pickle.dat')
model_path = join(dirname, '../model/model.pickle.dat')
loaded_model = pickle.load(open(model_path, "rb"))
min_max_scaler = pickle.load(open(min_max_scaler_path, "rb"))

#Testing with all other database images
def predictGenreWithModel(wav_path):
	# make predictions for test data
	a = getmetadata(wav_path)
	d1 = np.array(a)
	data1 = min_max_scaler.transform([d1])
	genre_result = loaded_model.predict(data1)[0]
	print("Prediction: ", genre_result)
	return {"genre_result": genre_result}

def getmetadata(filename):
    y, sr = librosa.load(filename)

    #fetching tempo
    onset_env = librosa.onset.onset_strength(y, sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    #fetching beats
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)

    #chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    #rmse
    rmse = librosa.feature.rms(y=y)

    #fetching spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y, sr=sr)[0]

    #spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    #fetching spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr)[0]

    #zero crossing rate
    zero_crossing = librosa.feature.zero_crossing_rate(y)

    #mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    #metadata dictionary
    metadata_dict = {'chroma_stft_mean':np.mean(chroma_stft),'rms_mean':np.mean(rmse),
                     'spectral_centroid_mean':np.mean(spec_centroid),'spectral_bandwidth_mean':np.mean(spec_bw),
                     'rolloff_mean':np.mean(spec_rolloff), 'zero_crossing_rate_mean':np.mean(zero_crossing), 'harmony_mean':np.mean(y_harmonic)}

    for i in range(1,21):
        metadata_dict.update({'mfcc'+str(i)+'_mean':np.mean(mfcc[i-1])})

    metadata_dict.update({'tempo':tempo})
    return list(metadata_dict.values())
