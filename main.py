# python3 -m venv venv 
# source venv/bin/activate
# pip install onnxruntime numpy librosa torch torchvision torchaudio
# wget https://github.com/pengzhendong/pyannote-onnx/blob/master/pyannote_onnx/segmentation-3.0.onnx
# wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx
# wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/sam_altman.wav
# python3 main.py

from segment import get_segments
from identify import SpeakerEmbeddingManager
from extract import EmbeddingExtractor
from common import read_wav

if __name__ == '__main__':
    samples, sample_rate = read_wav('sam_altman.wav')
    
    embedding_manager = SpeakerEmbeddingManager(3)
    extractor = EmbeddingExtractor('wespeaker_en_voxceleb_CAM++.onnx')
    segments = get_segments('segmentation-3.0.onnx', samples, sample_rate)
    for segment in segments:
        start_sample = int(segment['start'] * sample_rate)
        end_sample = int(segment['end'] * sample_rate)
        segment_samples = samples[start_sample:end_sample]
        embedding = extractor.compute(segment_samples, sample_rate)
        speaker = embedding_manager.get_speaker(embedding, threshold=0.4)
        segment['speaker'] = speaker
        print(segment)