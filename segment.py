import numpy as np
from common import init_session, read_wav

def get_segments(modal_path, samples: np.ndarray, sample_rate: int):
    session = init_session(modal_path)
    
    # Conv1d & MaxPool1d & SincNet https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/models/blocks/sincnet.py#L50-L71
    frame_size = 270
    frame_start = 721
    window_size = sample_rate * 10 # 10s

    # State and offset
    is_speeching = False
    offset = frame_start
    start_offset = 0

    # Pad end with silence for full last segment
    samples = np.pad(samples, (0, window_size), 'constant') 
    segments = []

    for start in range(0, len(samples), window_size):
        window = samples[start:start + window_size]
        ort_outs: np.array = session.run(None, {'input': window[None, None, :]})[0][0]
        for probs in ort_outs:
            predicted_id = np.argmax(probs)
            if predicted_id != 0:
                if not is_speeching:
                    start_offset = offset
                    is_speeching = True
            elif is_speeching:
                start = round(start_offset / sample_rate, 3)
                end = round(offset / sample_rate, 3)
                segments.append({'start': start, 'end': end})
                is_speeching = False
            offset += frame_size
    return segments
