"""
python3 -m venv venv 
source venv/bin/activate
pip install onnxruntime numpy librosa kaldi-native-fbank
wget https://github.com/pengzhendong/pyannote-onnx/raw/master/pyannote_onnx/segmentation-3.0.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/5_speakers.wav
python3 main.py
"""

from segment import get_segments
from identify import SpeakerEmbeddingManager
from extract import EmbeddingExtractor
from common import read_wav

if __name__ == '__main__':
    samples, sample_rate = read_wav('5_speakers.wav')
    
    num_speakers = 5
    
    extractor = EmbeddingExtractor('wespeaker_en_voxceleb_CAM++.onnx')
    segments = get_segments('segmentation-3.0.onnx', samples, sample_rate)
    
    embedding_manager = SpeakerEmbeddingManager(num_speakers)
    for segment in segments:
        start_sample = int(segment['start'] * sample_rate)
        end_sample = int(segment['end'] * sample_rate)
        segment_samples = samples[start_sample:end_sample]
        embedding = extractor.compute(segment_samples, sample_rate)

        speaker = embedding_manager.get_speaker(embedding, threshold=0.5)
        if not speaker and len(embedding_manager.get_all_speakers()):
            speaker = embedding_manager.get_speaker(embedding, threshold=0)
            
        segment['speaker'] = speaker
        print(segment)
        