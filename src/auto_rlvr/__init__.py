import torch
import sglang as sgl
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn

MODEL = "qwen/qwen2.5-0.5b-instruct"


def get_log_probs(input_ids: torch.Tensor,
                  rollout_ids: torch.Tensor, policy: nn.Module):
    full_id = torch.cat((input_ids, rollout_ids), dim=-1)
    logits = policy(full_id, dtype=torch.bfloat16).logits
    return torch.log_softmax(logits[:, input_ids.shape[-1] - 1: -1], dim=-1, dtype=torch.float)


def policy_gradient_reinforce_loss(rollout_ids: torch.Tensor, rollout_log_probs: torch.Tensor,
                                   reward: float) -> torch.Tensor:
    tlog_probs = rollout_log_probs.gather(dim=-1, index=rollout_ids.unsqueeze(-1)).squeeze(-1)
    loss = -(reward * tlog_probs).mean()
    return loss


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to('cuda')

    rollout_model = sgl.Engine(
        model_path=MODEL,
        skip_tokenizer_init=True,
        #    enable_memory_saver=True,
        mem_fraction_static=0.1,
    )
    # todo верл
    prompts = ["привет как дела"]
    prompt_ids = []
    for p in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
        )["input_ids"]
        prompt_ids.append(list(ids))

    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 128,
        "stop_token_ids": [tokenizer.eos_token_id],
    }
    rollout = rollout_model.generate(input_ids=prompt_ids, sampling_params=sampling_params)

    rollout_ids = []
    for r in rollout:
        rollout_ids.append(r['output_ids'])

    input_ids = torch.tensor(prompt_ids).to('cuda')
    rollout_ids = torch.tensor(rollout_ids).to('cuda')
    rollout_log_probs = get_log_probs(input_ids, rollout_ids, model)
    loss = policy_gradient_reinforce_loss(rollout_ids, rollout_log_probs, 1)
