import numpy as np

class SpeakerEmbeddingManager:
    def __init__(self, max_speakers = 10):
        self.max_speakers = max_speakers
        self.speakers = {}
        self.next_speaker_id = 1

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_speaker(self, embedding, threshold=0.5):
        best_speaker_id = None
        best_similarity = threshold
        
        for speaker_id, speaker_embedding in self.speakers.items():
            similarity = self.cosine_similarity(embedding, speaker_embedding)
            if similarity > best_similarity:
                best_speaker_id = speaker_id
                best_similarity = similarity

        if best_speaker_id is not None:
            return best_speaker_id
        
        if len(self.speakers) < self.max_speakers:
            return self._add_speaker(embedding)
        return None

    def _add_speaker(self, embedding):
        speaker_id = self.next_speaker_id
        self.speakers[speaker_id] = embedding
        self.next_speaker_id += 1
        return speaker_id

    def get_all_speakers(self):
        """Return a dictionary of all speaker embeddings."""
        return self.speakers