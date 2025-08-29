import collections
import regex as re

PRE_TOKENIZER_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE_Trainer:
    def __init__(self, vocab_size, corpus_path, special_tokens=None):
        self.vocab_size = vocab_size
        self.corpus_path = corpus_path
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_in_bytes = [t.encode("utf-8") for t in self.special_tokens]
        self._init_vocab()
        self.merges = []

    def _init_vocab(self):
        self.vocab = self.special_tokens_in_bytes + [bytes([i]) for i in range(256)]

    def _split_corpus(self):
        with open(self.corpus_path, 'r') as f:
            corpus = f.read()
        # 用special_tokens中的元素分割corpus
        if self.special_tokens:
            # Escape special tokens for regex
            special_tokens_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            # Split corpus, keeping the special tokens as separate segments
            parts = [p for p in re.split(f"({special_tokens_pattern})", corpus) if (p and p not in self.special_tokens)]
        else:
            parts = [corpus]

        segment_freqs = collections.defaultdict(int)
        for part in parts:
            for match in re.finditer(PRE_TOKENIZER_PAT, part):
                segment = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
                segment_freqs[segment] += 1
        return segment_freqs

    def _get_pair_freqs(self, segment_freqs):
        pair_freqs = collections.defaultdict(int)
        for segment, freq in segment_freqs.items():
            for i in range(len(segment) - 1):
                pair = (segment[i], segment[i+1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _find_max_pair_and_update(self, segment_freqs, pair_freqs):
        max_freq = 0
        max_pair = None
        for pair, freq in pair_freqs.items():
            if freq > max_freq:
                max_freq = freq
                max_pair = pair
            elif freq == max_freq:
                if pair[0] > max_pair[0]:
                    max_pair = pair
                elif pair[0] == max_pair[0] and pair[1] > max_pair[1]:
                    max_pair = pair
        del pair_freqs[max_pair]
        token1, token2 = max_pair
        new_token = token1 + token2
        new_segments = collections.defaultdict(int)
        segments_to_delete = set()
        for segment, freq in segment_freqs.items():
            old_segment = segment
            segment = list(segment)
            updated = False
            i = 0
            while i < len(segment) - 1:
                if segment[i] == token1 and segment[i+1] == token2:
                    if i > 0:
                        old_pair = (segment[i-1], segment[i])
                        pair_freqs[old_pair] -= freq
                        if pair_freqs[old_pair] <=0:
                            del pair_freqs[old_pair]

                        new_pair = (segment[i-1], new_token)
                        pair_freqs[new_pair] += freq

                    if i+2 < len(segment):
                        old_pair = (segment[i+1], segment[i+2])
                        pair_freqs[old_pair] -= freq
                        if pair_freqs[old_pair] <=0:
                            del pair_freqs[old_pair]

                        new_pair = (new_token, segment[i+2])
                        pair_freqs[new_pair] += freq

                    segment[i] = new_token
                    del segment[i+1]
                    updated = True
                i += 1
            if updated:
                new_segments[tuple(segment)] += freq
                segments_to_delete.add(old_segment)

        for segment in segments_to_delete:
            del segment_freqs[segment]
        for segment, freq in new_segments.items():
            segment_freqs[segment] += freq

        return max_pair

    def train(self):
        segment_freqs = self._split_corpus()
        pair_freqs = self._get_pair_freqs(segment_freqs)
        while len(self.vocab) < self.vocab_size:
            max_pair = self._find_max_pair_and_update(segment_freqs, pair_freqs)
            self.merges.append(max_pair)
            self.vocab.append(max_pair[0] + max_pair[1])
        return self.merges
