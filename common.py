import onnxruntime as ort
import librosa

def init_session(model_path):
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model_path, sess_options=opts)
    return sess


def read_wav(path: str):
    return librosa.load(path, sr=16000)