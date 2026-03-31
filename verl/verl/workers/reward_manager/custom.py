# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.utils.check_cover import get_cover_scores
from verl.workers.reward_manager.utils.check_cf import get_cf_scores
from verl.workers.reward_manager.utils.aux_rewards import calculate_json_format_reward, calculate_length_reward, get_token_num

import json
import re
import numpy as np


@register("custom")
class CustomRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        score_data = []
        valid_response_lengths = []

        format_rewards = []
        cots = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if not response_str.strip().startswith("<think>"):
                response_str = "<think>" + response_str
            
            format_reward = 0
            format_rewards.append(format_reward)

            cot = response_str.split("</think>")[0].split("<think>")[-1].strip()
            cots.append(cot)

            # print("response str:")
            # print(response_str)

            response = response_str.split("</think>")[-1].strip()
            item = {
                "response": response,
                "title": data_item.non_tensor_batch.get("title"),
                "content": data_item.non_tensor_batch.get("content"),
                "summary_points": json.loads(data_item.non_tensor_batch.get("summary_points")),
                "r1_response": data_item.non_tensor_batch.get("r1_response")
            }
            score_data.append(item)

            valid_response_lengths.append(valid_response_length)

        # 1. InfoCover Reward
        print("calculating InfoCover scores...")
        cover_rewards = get_cover_scores(score_data)
        print("InfoCover scores calculated.")

        # 2. 简洁性和流畅性
        print("calculating Conciseness and Fluency scores...")
        cf_rewards = get_cf_scores(score_data)
        print("Conciseness and Fluency scores calculated.")

        # 3. Json Format Reward response
        print("calculating format scores...")
        json_format_rewards = [calculate_json_format_reward(item["response"]) for item in score_data]
        print("format scores calculated.")

        # 4. 长度惩罚
        print("calculating length scores...")
        length_rewards = []
        for format_reward, json_format_reward, cot, item in zip(format_rewards, json_format_rewards, cots, score_data):
            length_reward = calculate_length_reward(
                format_reward,
                json_format_reward,
                item["response"],
                cot,
                item["r1_response"],
                self.tokenizer
            )
            length_rewards.append(length_reward)
        print("length scores calculated.")

        answer_lengths = [get_token_num(item["response"], self.tokenizer) for item in score_data]
        cot_lengths = [get_token_num(cot, self.tokenizer) for cot in cots]
        # 
        metric_recalls, metric_precisons = [], []
        metric_cover_rewards = []
        for i in range(len(data)):
            (ex_recall, ex_precison), (short_recall, short_precision), (long_recall, long_precision) = cover_rewards[i]
            ex_f1 = 0.5 * ex_recall + 0.5 * ex_precison
            short_f1 = 0.5 * short_recall + 0.5 * short_precision
            long_f1 = 0.5 * long_recall + 0.5 * long_precision
            cover_reward = 0.2 * ex_f1 + 0.3 * short_f1 + 0.5 * long_f1
            metric_cover_rewards.append(cover_reward)
            # 
            metric_recalls.append(0.2 * ex_recall + 0.3 * short_recall + 0.5 * long_recall)
            metric_precisons.append(0.2 * ex_precison + 0.3 * short_precision + 0.5 * long_precision)
            
            cf_reward = cf_rewards[i]
            format_reward = format_rewards[i]
            json_format_reward = json_format_rewards[i]

            total_format_reward = json_format_reward

            if total_format_reward != 0:
                cover_reward = 0
                cf_reward = 0

            length_reward = length_rewards[i]
            total_reward = 0.8 * cover_reward + 0.2 * cf_reward + total_format_reward + length_reward
            reward_tensor[i, valid_response_lengths[i] - 1] = total_reward

        return reward_tensor, metric_cover_rewards, metric_recalls, metric_precisons, cf_rewards, format_rewards, json_format_rewards, length_rewards, cot_lengths, answer_lengths
