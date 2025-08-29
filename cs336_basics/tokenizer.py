from typing import Iterable
import regex as re

from .bpe import PRE_TOKENIZER_PAT

class BPE_Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.bytes_to_id = {}
        self.id_to_bytes = {}
        for i, b in vocab.items():
            self.bytes_to_id[b] = i
            self.id_to_bytes[i] = b

        self.merges_rank = {merge: i for i, merge in enumerate(merges)}
        if special_tokens:
            # Sort special tokens by length descending, so longer tokens are matched first
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = []
        self.special_tokens_pattern = "|".join(re.escape(token) for token in self.special_tokens) if self.special_tokens else None
        self.pre_tokenizer = re.compile(PRE_TOKENIZER_PAT)


    def _pre_tokenize(self, text: str) -> Iterable[str]:
        parts = re.splititer(f"({self.special_tokens_pattern})", text) if self.special_tokens else [text]
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                yield part
            else:
                for match in self.pre_tokenizer.finditer(part):
                    yield match.group()
    
    def _merge_tokens(self, text: str) -> list[int]:
        text_bytes = [bytes([b]) for b in text.encode("utf-8")]
        merged = True
        while merged:
            merged = False
            best_rank = float("inf")
            best_positions = []
            for i in range(len(text_bytes) - 1):
                if (text_bytes[i], text_bytes[i+1]) in self.merges_rank:
                    rank = self.merges_rank[(text_bytes[i], text_bytes[i+1])]
                    if rank < best_rank:
                        best_rank = rank
                        best_positions = [i]
                    elif rank == best_rank and (i-1) > best_positions[-1]:
                        best_positions.append(i)
            if best_positions:
                merged = True
                for i in best_positions:
                    text_bytes[i] = text_bytes[i] + text_bytes[i+1]
                    text_bytes[i+1] = None
                text_bytes = [b for b in text_bytes if b is not None] # remove None
        return [self.bytes_to_id[b] for b in text_bytes]


    def encode(self, text: str) -> list[int]:
        ids = []
        for part in self._pre_tokenize(text):
            if part in self.special_tokens:
                ids.append(self.bytes_to_id[part.encode("utf-8")])
            else:
                ids.extend(self._merge_tokens(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text_bytes = b''
        for id in ids:
            if id in self.id_to_bytes:
                text_bytes += self.id_to_bytes[id]
            else:
                print(f"id {id} not found in vocab")
        return text_bytes.decode("utf-8", errors="replace")
