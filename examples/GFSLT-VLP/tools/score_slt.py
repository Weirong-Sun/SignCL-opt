"""
计算手语翻译论文常用指标：BLEU-1/2/3/4 和 ROUGE-L。

用法（在 examples/GFSLT-VLP 目录下）：
    python tools/score_slt.py out/dsl_LLMD_bs8/tmp_pres.txt out/dsl_LLMD_bs8/tmp_refs.txt

输入：
    hyp_file  模型译文，每行一句（--eval 单卡运行时生成的 tmp_pres.txt）
    ref_file  标准译文，每行一句（tmp_refs.txt），行数必须和 hyp_file 一样
输出：
    打印 BLEU-1..4（sacreBLEU 语料级，和训练日志里的 BELU-4 算法相同）
    以及 ROUGE-L（和 nlg-eval / pycocoevalcap 相同：按句计算 LCS 的 F 值，beta=1.2，再取平均）
"""
import sys

from sacrebleu.metrics import BLEU


def lcs_length(a, b):
    """两个词列表的最长公共子序列长度。"""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hyp, ref, beta=1.2):
    """单句 ROUGE-L F 值。"""
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    lcs = lcs_length(h, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(h), lcs / len(r)
    return (1 + beta ** 2) * prec * rec / (rec + beta ** 2 * prec)


def main(hyp_file, ref_file):
    hyps = [line.strip() for line in open(hyp_file, encoding="utf-8")]
    refs = [line.strip() for line in open(ref_file, encoding="utf-8")]
    if len(hyps) != len(refs):
        raise ValueError(f"行数不一致：{hyp_file} 有 {len(hyps)} 行，{ref_file} 有 {len(refs)} 行")

    for n in range(1, 5):
        score = BLEU(max_ngram_order=n).corpus_score(hyps, [refs]).score
        print(f"BLEU-{n}: {score:.2f}")
    rl = 100 * sum(rouge_l(h, r) for h, r in zip(hyps, refs)) / len(hyps)
    print(f"ROUGE-L: {rl:.2f}")
    print(f"sentences: {len(hyps)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
