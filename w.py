from flask import Flask, jsonify
import socket
import threading
import logging
import numpy as np
import io
import soundfile as sf
import time
import argparse
from whisper_online_server import Connection, add_shared_args, OnlineASRProcessor, WhisperTimestampedASR, FasterWhisperASR

# Configuration
WHISPER_SERVER_IP = '0.0.0.0'
WHISPER_SERVER_PORT = 43007
WHISPER_MODEL = 'medium'
WHISPER_LANGUAGE = 'en'
FLASK_SERVER_PORT = 5000
SAMPLING_RATE = 16000
CHUNK = 960  # 960 samples per chunk to fit with rate

app = Flask(__name__)
transcribed_text = []

# Initialize the Whisper ASR model
parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default=WHISPER_SERVER_IP)
parser.add_argument("--port", type=int, default=WHISPER_SERVER_PORT)

add_shared_args(parser)
args = parser.parse_args()
min_chunk = args.min_chunk_size
language = args.lan
size = args.model

if args.backend == "faster-whisper":
    from faster_whisper import WhisperModel
    asr_cls = FasterWhisperASR
else:
    import whisper
    import whisper_timestamped
    asr_cls = WhisperTimestampedASR

asr = asr_cls(modelsize=size, lan=language, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
online = OnlineASRProcessor(asr, None, buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))

class ServerProcessor:
    def __init__(self, c, online_asr_proc, min_chunk):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.last_end = None

    def receive_audio_chunk(self):
        out = []
        while sum(len(x) for x in out) < self.min_chunk * SAMPLING_RATE:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE", samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
            audio, _ = librosa.load(sf, sr=SAMPLING_RATE)
            out.append(audio)
        if not out:
            return None
        return np.concatenate(out)

    def format_output_transcript(self, o):
        if o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            self.last_end = end
            return "%1.0f %1.0f %s" % (beg, end, o[2])
        else:
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = online.process_iter()
            transcribed_text.append(o)
            try:
                self.send_result(o)
            except BrokenPipeError:
                break

def audio_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((WHISPER_SERVER_IP, WHISPER_SERVER_PORT))
        s.listen(1)
        logging.info('INFO: Listening on ' + str((WHISPER_SERVER_IP, WHISPER_SERVER_PORT)))
        while True:
            conn, addr = s.accept()
            logging.info('INFO: Connected to client on {}'.format(addr))
            connection = Connection(conn)
            proc = ServerProcessor(connection, online, min_chunk)
            proc.process()
            conn.close()
            logging.info('INFO: Connection to client closed')

@app.route('/transcriptions', methods=['GET'])
def get_transcriptions():
    global transcribed_text
    return jsonify(transcribed_text)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    audio_thread = threading.Thread(target=audio_server)
    audio_thread.daemon = True
    audio_thread.start()
    app.run(host='0.0.0.0', port=FLASK_SERVER_PORT)
