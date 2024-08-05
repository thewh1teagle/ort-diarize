import numpy as np
import kaldi_native_fbank as knf
from common import init_session

def compute_fbank(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.snip_edges = True

    opts.mel_opts.num_bins = 80
    opts.mel_opts.debug_mel = False

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, samples.tolist())
    fbank.input_finished()

    features = []
    for i in range(fbank.num_frames_ready):
        f = fbank.get_frame(i)
        features.append(f)
    features = np.stack(features, axis=0)
    
    # Apply CMN (Cepstral Mean Normalization)
    features = features - np.mean(features, axis=0)
    
    return features

class EmbeddingExtractor:
    def __init__(self, model_path: str) -> None:
        self.session = init_session(model_path)
    
    def compute(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        feats = compute_fbank(samples, sample_rate)
        feats = np.expand_dims(feats, axis=0) # add batch dimension

        embeddings = self.session.run(output_names=['embs'],
                                input_feed={'feats': feats})
        return embeddings[0][0]
