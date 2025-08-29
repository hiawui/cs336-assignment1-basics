import pickle

from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

def test_load_test_train_bpe_special_tokens():
    """
    读取 test_train_bpe_special_tokens.pkl 文件的数据并返回
    """
    file_path = FIXTURES_PATH.parent / "_snapshots" / "test_train_bpe_special_tokens.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    assert "vocab_values" in data

    single_byte_count = 0
    # 打印出data["vocab_values"]中所有单字节字符对应的数值（按数字打印）
    for val in data["vocab_values"]:
        if isinstance(val, bytes) and len(val) == 1:
            print(int(val[0]))
            single_byte_count += 1

    print(f"single_byte_count: {single_byte_count}")

def test_read_tinystories_sample_with_mmap():
    """
    使用mmap读取fixtures中的tinystories_sample.txt内容
    """
    import mmap

    file_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 128, access=mmap.ACCESS_READ) as mm:
            mm.seek(1)
            # 简单断言内容非空
            print(f'len(content): {len(mm)}')
            print(f'content: {mm[:]}')

def test_ref_bpe():
    bytes_to_unicode = gpt2_bytes_to_unicode()
    unicode_to_bytes = {v: k for k, v in bytes_to_unicode.items()}

    def str_to_bytes(s: str) -> bytes:
        return bytes([unicode_to_bytes[c] for c in s])
    
    def bytes_to_str(b: bytes) -> str:
        return "".join([bytes_to_unicode[b] for b in b])

    merges = []
    with open(FIXTURES_PATH / "train-bpe-reference-merges.txt", "r") as f:
        for line in f:
            pair = line.strip().split(" ")
            merges.append((str_to_bytes(pair[0]), str_to_bytes(pair[1])))

    print(f"merges: {merges[:10]}")
            
    with open(FIXTURES_PATH / "corpus.en", "rb") as f:
        corpus_in_bytes = f.read()
    corpus_bytes_segments = [bytes([b]) for b in corpus_in_bytes]

    print(f"corpus_bytes_segments: {corpus_bytes_segments[:10]}")
    for i in range(16):
        pair = merges[i]
        freq = 0
        j = 0
        while j < len(corpus_bytes_segments) - 1:
            if corpus_bytes_segments[j] == pair[0] and corpus_bytes_segments[j+1] == pair[1]:
                corpus_bytes_segments[j] = pair[0] + pair[1]
                del corpus_bytes_segments[j+1]
                freq += 1
            j += 1

        print(f"pair: {bytes_to_str(pair[0])} {bytes_to_str(pair[1])}, freq: {freq}")


