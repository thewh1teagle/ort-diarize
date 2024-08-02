import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from common import init_session


def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def compute_fbank(samples,
                  sample_rate,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """Extract fbank, similar to the one in wespeaker.dataset.processor,
       While integrating the wave reading and CMN.
    """
    waveform = torch.tensor(samples, dtype=torch.float32)
    
    # Add channel dimension if waveform is 1D
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    waveform = waveform * (1 << 15)  # Scale to 16-bit range
    
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      sample_frequency=sample_rate,
                      window_type='hamming',
                      use_energy=False)
    
    # CMN, without CVN
    mat = mat - torch.mean(mat, dim=0)
    return mat


class EmbeddingExtractor:
    def __init__(self, model_path: str) -> None:
        self.session = init_session(model_path)
    
    def compute(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        feats = compute_fbank(samples, sample_rate)
        feats = feats.unsqueeze(0).numpy()  # add batch dimension

        embeddings = self.session.run(output_names=['embs'],
                                input_feed={'feats': feats})
        return embeddings[0][0]
