"""
action_tokenizer.py

Todo:
1. 在__call__的时候就把action映射成1-255
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase

class ActionTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1) -> None :
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins) #将连续值映射成bins个区间
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2 #取出每个小区间的中心

        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - 1 - self.n_bins)
        # 其实是把action token当作词汇表中的一部分，把词汇表中最不常出现的东西替代掉。
        # 因为ids的范围是[0, self.tokenizer.vocab_size-1]

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """
        converts a continuous action into a action language

        :param action: Continuous action to convert.
        """
        action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
        discretized_action = np.digitize(action, self.bins)
        #print(f"discretized_action: {discretized_action}")

        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.action_token_begin_idx + discretized_action))
        else:
            return self.tokenizer.batch_decode((self.action_token_begin_idx + discretized_action).tolist())
    
    def covert_token_ids_to_actions(self, action_token_ids: List[int]) -> np.ndarray:
        """
        converts a list of token ids into a continuous action

        :param token_ids: List of token ids to convert.
        """
        discretized_action = np.array(action_token_ids) - self.action_token_begin_idx
        #print(f"discretized_action: {discretized_action}")
        discretized_action = np.clip(discretized_action - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_action]

    @property
    def vocab_size(self) -> int:
        return self.n_bins

"""
# 测试修正后的代码
class DummyTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
    
    def decode(self, token_ids):
        return f"tokens_{token_ids}"
    
    def batch_decode(self, batch_token_ids):
        return [self.decode(ids) for ids in batch_token_ids]

# 测试
tokenizer = DummyTokenizer(vocab_size=50000)
action_tokenizer = ActionTokenizer(tokenizer, bins=256, min_action=-1, max_action=1)

# 测试动作编码
test_actions = np.array([0.5, -1, 1, 0.0])
encoded = action_tokenizer(test_actions)
print(f"Encoded: {encoded}")

decoded = [49935, 49744, 49999, 49871]
decoded = action_tokenizer.covert_token_ids_to_actions(decoded)
print(f"Decoded: {decoded}")
"""