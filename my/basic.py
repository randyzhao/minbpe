import collections

from typing import List

def get_pair_stat(ids):
    pair_counter = collections.Counter(zip(ids, ids[1:]))
    return sorted(pair_counter.items(), key=lambda item: item[1], reverse=True)

def encode_pair_with_id(ids, pair, new_id) -> List[int]:
    result = []
    i = 0
    while i <= len(ids) - 1:
        if i <= len(ids) - 2 and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result

class BasicTokenizer:
    def train(self, text, vocab_size, verbose=False):
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        origin_len = len(ids)

        vocab = {id: bytes([id]) for id in range(256)}
        merges = {}
        num_merges = vocab_size - 256

        for i in range(256, vocab_size):
            merge_id = i - 255
            pair_counts = get_pair_stat(ids)
            # max_pair: (int, int)
            max_pair, max_pair_count = pair_counts[0]
            if max_pair_count == 1:
                break

            # {i -> byte1.byte2.byte3...byte_n}
            vocab[i] = vocab[max_pair[0]] + vocab[max_pair[1]]
            merges[max_pair] = i
            ids = encode_pair_with_id(ids, max_pair, i)
            print(f'merge {merge_id} / {num_merges}: {max_pair} -> {i} had {max_pair_count} occurrences')
        
        compressed_len = len(ids)
        print(f'compression ratio: {compressed_len / origin_len}')
        self.vocab = vocab
        self.merges = merges

    def encode(self, text):
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)

        while True:
            pair_id = self._find_lowest_mergable_pair(ids)
            if not pair_id:
                break
            ids = encode_pair_with_id(ids, pair_id[0], pair_id[1])
        
        return ids

    def _find_lowest_mergable_pair(self, ids):
        mergable_pairs = [pair for pair in self.merges if pair in zip(ids, ids[1:])]
        if not mergable_pairs:
            return None
        min_pair = min(mergable_pairs, key=self.merges.get)
        return (min_pair, self.merges[min_pair])

    def decode(self, ids):
        text_bytes = b''.join(self.vocab[idx] for idx in ids)
        return text_bytes.decode('utf-8', errors='replace')



if __name__ == '__main__':
    # print(get_pair_stat([1, 2, 3, 1, 2, 3]))

    assert encode_pair_with_id([1, 2, 3, 2, 3], (2, 3), 4) == [1, 4, 4]
    assert encode_pair_with_id([1, 2, 3, 2], (2, 3), 4) == [1, 4, 2]
    
    TEST_STR = 'Aab, cab, dab, cbb'
    tokenizer = BasicTokenizer()
    tokenizer.train(TEST_STR, vocab_size=258)
    encoded = tokenizer.encode(TEST_STR)
    print(encoded)
    print(tokenizer.decode(encoded))
    assert tokenizer.decode(tokenizer.encode(TEST_STR)) == TEST_STR 

    print('ALL TESTS PASSED!')
