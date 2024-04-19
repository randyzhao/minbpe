from my import basic
import os
import pytest

def load_taylor_swift():
    dirname = os.path.dirname(os.path.abspath(__file__))
    taylorswift_file = os.path.join(dirname, "taylorswift.txt")

    with open(taylorswift_file, "r", encoding="utf-8") as f:
        taylorswift_text = f.read()

    return taylorswift_text

def test_basic():
    tokenizer = basic.BasicTokenizer()
    text = load_taylor_swift()
    tokenizer.train(text, 400, False)
    encoded_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_ids)
    print(decoded_text[:100])
    assert text == decoded_text