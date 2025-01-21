from flask import Flask, request, jsonify, render_template
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
import os
import wave
import io  # Import the io module
from flask_socketio import SocketIO
from datetime import datetime
import base64
from pydub import AudioSegment
import tempfile
import soundfile as sf
import numpy as np
from pydub.utils import which

# Configure FFmpeg path
ffmpeg_path = os.path.join(os.path.dirname(__file__), 'bin', 'ffmpeg.exe')
AudioSegment.converter = ffmpeg_path
print(f"Using FFmpeg from: {ffmpeg_path}")  # Debug print

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

PROJECT_ID = "ee-linkdatauz"

@app.route('/')
def index():
    return render_template('index.html')  # Serve the index.html file

def transcribe_sync_chirp2(audio_file_path: str) -> str:
    """Transcribes an audio file using the Chirp 2 model of Google Cloud Speech-to-Text V2 API.
    
    Args:
        audio_file_path (str): Path to the local audio file to be transcribed.
    
    Returns:
        str: The combined transcription results from the audio file.
    """
    # Instantiate a client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint="us-central1-speech.googleapis.com",
        )
    )

    # Read the audio file as bytes
    try:
        with open(audio_file_path, "rb") as f:
            audio_content = f.read()
    except Exception as e:
        print(f"Error reading the audio file: {e}")
        return ""

    # Use BytesIO to create a file-like object from the audio content
    with io.BytesIO(audio_content) as audio_file:
        with wave.open(audio_file, 'rb') as wave_file:
            total_frames = wave_file.getnframes()
            frame_rate = wave_file.getframerate()
            duration = total_frames / frame_rate

            # Ensure the audio format is correct
            if wave_file.getsampwidth() != 2:  # Check for 16-bit audio
                raise ValueError("Audio file must be 16-bit PCM format.")
            if wave_file.getcomptype() != 'NONE':  # Check for uncompressed audio
                raise ValueError("Audio file must be uncompressed PCM format.")

    # Split audio into chunks if longer than 60 seconds
    chunk_duration = 60  # seconds
    transcripts = []

    for start in range(0, int(duration), chunk_duration):
        end = min(start + chunk_duration, int(duration))
        audio_chunk = audio_content[start * frame_rate * 2:end * frame_rate * 2]  # Assuming 16-bit audio

        # Specify the audio configuration
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["uz-UZ"],
            model="chirp_2",
            features=cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True
            ),
        )

        request = cloud_speech.RecognizeRequest(
            recognizer=f"projects/{PROJECT_ID}/locations/us-central1/recognizers/_",
            config=config,
            content=audio_chunk,
        )

        # Transcribe the audio into text
        response = client.recognize(request=request)

        # Collect the transcription for the current chunk
        transcripts_chunk = [result.alternatives[0].transcript for result in response.results]
        transcripts.append(' '.join(transcripts_chunk))  # Merge chunk transcriptions

    return ' '.join(transcripts)  # Merge all transcriptions

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file provided.'})

    audio_file = request.files['audio']
    audio_content = audio_file.read()
    transcript = transcribe_sync_chirp2(audio_content)
    
    return jsonify({'success': True, 'text': transcript})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    try:
        # Decode base64 audio chunk
        audio_data = base64.b64decode(data['audio'].split(',')[1])
        
        # Save directly as WAV since we're already receiving WAV format
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            wav_file.write(audio_data)
            wav_path = wav_file.name

        # Verify the WAV file
        with wave.open(wav_path, 'rb') as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getframerate() != 16000:
                raise ValueError("Invalid WAV format")

        # Process with Chirp 2
        transcript = transcribe_sync_chirp2(wav_path)
        
        # Clean up temp file
        os.remove(wav_path)
        
        # Send transcription back to client
        if transcript:
            socketio.emit('transcription', {'text': transcript})
        else:
            raise ValueError("No transcription result")
            
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        socketio.emit('error', {'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3105))
    socketio.run(app, debug=True, host = '0.0.0.0', port = port, allow_unsafe_werkzeug=True)
