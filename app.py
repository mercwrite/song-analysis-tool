from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import os

app = Flask(__name__)
CORS(app) 

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file temporarily
    filepath = os.path.join('/tmp', file.filename)
    file.save(filepath)

    try:
        # Load audio file for analysis using librosa
        y, sr = librosa.load(filepath)

        # BPM (tempo)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Key and mode estimation
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = np.argmax(np.mean(chroma, axis=1))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_idx]

        # Mode detection (Major/Minor)
        mode = "minor" if chroma[3].mean() > chroma[0].mean() else "major"

        return jsonify({
            "tempo": tempo,
            "key": key,
            "mode": mode
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Remove the file after processing
        os.remove(filepath)


if __name__ == '__main__':
    app.run(debug=True)
