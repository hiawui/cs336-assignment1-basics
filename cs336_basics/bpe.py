import collections
import heapq
import regex as re
from concurrent.futures import ThreadPoolExecutor
from line_profiler import profile

PRE_TOKENIZER_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class TokenPair:
    def __init__(self, first: bytes, second: bytes):
        self.first = first
        self.second = second

    def __hash__(self):
        return hash((self.first, self.second))

    def __eq__(self, other: 'TokenPair'):
        return self.first == other.first and self.second == other.second

    def __ne__(self, other: 'TokenPair'):
        return not self.__eq__(other)

    def __lt__(self, other: 'TokenPair'):
        return self.first < other.first or (self.first == other.first and self.second < other.second)

    def __le__(self, other: 'TokenPair'):
        return self.first < other.first or (self.first == other.first and self.second <= other.second)

    def __gt__(self, other: 'TokenPair'):
        return self.first > other.first or (self.first == other.first and self.second > other.second)

    def __ge__(self, other: 'TokenPair'):
        return self.first > other.first or (self.first == other.first and self.second >= other.second)

class TokenPairFreq:
    def __init__(self, pair: TokenPair, freq: int):
        self.pair = pair
        self.freq = freq

    def __hash__(self):
        return hash((self.pair.first, self.pair.second, self.freq))

    def __eq__(self, other: 'TokenPairFreq'):
        return self.pair == other.pair and self.freq == other.freq

    def __ne__(self, other: 'TokenPairFreq'):
        return not self.__eq__(other)

    def __lt__(self, other: 'TokenPairFreq'):
        return self.freq < other.freq or (self.freq == other.freq and self.pair < other.pair)

    def __le__(self, other: 'TokenPairFreq'):
        return self.freq < other.freq or (self.freq == other.freq and self.pair <= other.pair)

    def __gt__(self, other: 'TokenPairFreq'):
        return self.freq > other.freq or (self.freq == other.freq and self.pair > other.pair)

    def __ge__(self, other: 'TokenPairFreq'):
        return self.freq > other.freq or (self.freq == other.freq and self.pair >= other.pair)

class BPE_Trainer:
    def __init__(self, vocab_size, corpus_path, special_tokens=None, num_threads=8):
        self.vocab_size = vocab_size
        self.corpus_path = corpus_path
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_in_bytes = [t.encode("utf-8") for t in self.special_tokens]
        self.num_threads = num_threads
        self._init_vocab()
        self.merges = []

    def _init_vocab(self):
        self.vocab = self.special_tokens_in_bytes + [bytes([i]) for i in range(256)]

    def _split_corpus_part(self, part: str):
        segment_freqs = collections.Counter()
        for match in re.finditer(PRE_TOKENIZER_PAT, part):
            segment = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
            segment_freqs[segment] += 1
        return segment_freqs

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

        segment_freqs = collections.Counter()
        part_futures = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as thread_pool:
            for part in parts:
                part_futures.append(thread_pool.submit(self._split_corpus_part, part))
            for part_future in part_futures:
                segment_freqs.update(part_future.result())
        return segment_freqs

    def _get_pairs_from_segment(self, segment: tuple[bytes]):
        return [TokenPair(t0, t1) for t0, t1 in zip(segment, segment[1:])]

    def _get_pair_info(self, segment_freqs: collections.Counter):
        pair_freqs = collections.Counter()
        pair_to_segs = collections.defaultdict(set)
        seg_to_pair_freqs = collections.defaultdict(collections.Counter)
        for segment, freq in segment_freqs.items():
            for pair in self._get_pairs_from_segment(segment):
                pair_freqs[pair] += freq
                pair_to_segs[pair].add(segment)
                seg_to_pair_freqs[segment][pair] += freq
        return pair_freqs, pair_to_segs, seg_to_pair_freqs

    def _merge_segment(
            self, 
            segment: tuple[bytes], 
            freq: int, 
            new_pair: TokenPair, 
            pair_freqs: collections.Counter, 
            pair_to_segs: collections.defaultdict[TokenPair, set[tuple[bytes]]],
            seg_to_pair_freqs: collections.defaultdict(collections.Counter)
        ):
        seg_pair_freqs = seg_to_pair_freqs.pop(segment, collections.Counter())
        for pair, sfreq in seg_pair_freqs.items():
            pair_freqs[pair] -= sfreq
            segs = pair_to_segs[pair]
            if segment in segs:
                segs.remove(segment)
                if not segs:
                    del pair_to_segs[pair]
                    del pair_freqs[pair]

        new_segment = []
        i = 0
        last_index = len(segment) - 1
        while i < last_index:
            if segment[i] == new_pair.first and segment[i+1] == new_pair.second:
                new_segment.append(new_pair.first + new_pair.second)
                i += 2  # 跳过已合并的 token
            else:
                new_segment.append(segment[i])
                i += 1
        if i == last_index:
            new_segment.append(segment[last_index])
        new_segment = tuple(new_segment)
        for pair in self._get_pairs_from_segment(new_segment):
            pair_freqs[pair] += freq
            pair_to_segs[pair].add(new_segment)
            seg_to_pair_freqs[new_segment][pair] += freq
        return new_segment

    @profile
    def train(self):
        segment_freqs = self._split_corpus()
        pair_freqs, pair_to_segs, seg_to_pair_freqs = self._get_pair_info(segment_freqs)
        while len(self.vocab) < self.vocab_size:
            new_pair = heapq.nlargest(1, pair_freqs.items(), key=lambda x: TokenPairFreq(x[0], x[1]))[0][0]
            self.vocab.append(new_pair.first + new_pair.second)
            self.merges.append(new_pair)
            del pair_freqs[new_pair]
            segs_to_update = pair_to_segs[new_pair]
            for segment in list(segs_to_update):
                freq = segment_freqs.pop(segment, 0)
                new_segment = self._merge_segment(segment, freq, new_pair, pair_freqs, pair_to_segs, seg_to_pair_freqs)
                segment_freqs[new_segment] += freq

        return self.merges
