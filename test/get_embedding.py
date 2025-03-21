import ollama
import numpy as np

def test_get_embedding():
    texts = ['Hello','Hi','See you later','See you']
    response = ollama.embed(model = 'nomic-embed-text', input = texts)
    embeddings = np.array(response.embeddings)
    assert embeddings.shape == (4,768)
    print("Test get embedding: success")

if __name__ == '__main__':
    test_get_embedding()