"""内置训练语料：一段关于投机解码与大模型推理加速的技术短文。

用于构建确定性 n-gram 语言模型（草稿模型与目标模型共享同一语料），
让示例无需外部权重即可产出连贯的英文技术文本、验证完整推理链路。
"""
from __future__ import annotations

CORPUS = (
    "Speculative decoding accelerates large language model inference by using a "
    "small draft model to propose several tokens ahead of time. The large target "
    "model then verifies the whole draft in a single forward pass and keeps only "
    "the tokens it agrees with. Because the target reads many tokens at once, its "
    "memory bandwidth is used far more efficiently than in plain autoregressive "
    "generation. The draft model is cheap and fast, while the target model is slow "
    "but accurate. When the draft guesses well, the system accepts many tokens in "
    "one step and the throughput improves by a large factor. When the draft is wrong, "
    "the target rejects the bad tokens and corrects them with its own distribution. "
    "This accept and correct loop preserves the exact output distribution of the "
    "target model, so there is no loss in quality. The key idea is to turn serial "
    "token generation into a parallel verification problem. A faster draft yields "
    "higher speed, and a better aligned draft yields a higher acceptance rate. "
    "Adaptive draft length adjusts how many tokens the draft proposes based on the "
    "recent acceptance rate. The token tree extends the idea by verifying several "
    "candidate sequences at once. KV cache reuse lets the target skip recomputing "
    "the shared prefix for every position. Sparse attention keeps only the most "
    "relevant keys and reduces the cost of long context. Together these techniques "
    "make local and on device inference much more practical. A good draft model is "
    "small enough to run on a laptop yet close enough to the target to guess well. "
    "The draft reuses the target cache and index so it adds almost no extra memory. "
    "This design is exactly what the DFly draft model brings to long context and "
    "sparse attention models. It is a plug in acceleration component that does not "
    "change the weights or the output of the target model. Developers can therefore "
    "drop it into an existing serving stack without retraining anything. The only "
    "cost is running the tiny draft model, which is negligible next to the target. "
    "Measuring acceptance rate and speedup is essential to tune the draft length. "
    "A short draft wastes the target forward pass while a long draft wastes draft "
    "compute on rejected tokens. The sweet spot balances the two costs. In practice "
    "an acceptance rate around eighty percent already gives a strong speedup. "
    "Speculative decoding is one of the most reliable ways to make large models "
    "feel fast without touching their quality. It is widely used in production "
    "serving systems and in edge and mobile deployments. The same principle applies "
    "to any autoregressive model whether it generates text, code, audio or actions."
)
