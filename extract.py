import numpy as np
import torch
import kaldi_native_fbank as knf
import torchaudio.compliance.kaldi as kaldi
from common import init_session

def compute_fbank_kaldi_native(samples: np.ndarray, sample_rate: int) -> np.ndarray:
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
    
    def compute(self, samples: np.ndarray, sample_rate: int, use_native_fbank = True) -> np.ndarray:
        if use_native_fbank:
            feats = compute_fbank_kaldi_native(samples, sample_rate)
        else:
            feats = compute_fbank(samples, sample_rate)
        feats = np.expand_dims(feats, axis=0) # add batch dimension

        embeddings = self.session.run(output_names=['embs'],
                                input_feed={'feats': feats})
        return embeddings[0][0]
