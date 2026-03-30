# 字节对编码 (Byte-Pair Encoding, BPE) 是最主流的子词分词算法之一
import re,collections

# 准备语料库
vocab = {
    'h u g </w>': 1,
    'p u g </w>': 1, 
    'p u n </w>': 1, 
    'b u n </w>': 1
}
# 设置合并次数
num_merge = 6

def get_stats(vocab):
    """统计词元对频率"""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in) -> dict:
    """合并词元对"""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for word in v_in:
        # p 这个规则 = 只匹配“被空格包围的 u g”，左右必须是空格
        w_out = p.sub(''.join(pair), word)
        # 1. v_in[word]：去旧词表里拿 h u g </w> 对应的出现次数
        # 2. v_out[w_out] = ：把新单词放进新词表，并把次数赋值给它
        v_out[w_out] = v_in[word]

    return v_out

for i in range (num_merge):
    pairs = get_stats(vocab)
    if not pairs:
        break

    print(f"当前词组对: {pairs}")
    # max(字典, key=用什么来比大小)
    best = max(pairs, key = pairs.get)
    print(f"频率最高的是: {best}")
    vocab = merge_vocab(best, vocab)
    print(f"第{i+1}次合并: {best} -> {''.join(best)}")
    print(f"新词表（部分）: {list(vocab.keys())}")
    print("-" * 20)

