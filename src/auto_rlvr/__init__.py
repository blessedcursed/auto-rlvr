import torch
import sglang as sgl
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch import nn
from torch.nn.utils.rnn import pad_sequence

MODEL = "qwen/qwen2.5-0.5b-instruct"


def get_log_probs(input_ids: torch.Tensor,
                  rollout_ids: torch.Tensor,
                  attention_mask: torch.Tensor,
                  policy: nn.Module) -> torch.Tensor:
    full_id = torch.cat((input_ids, rollout_ids), dim=-1)
    logits = policy(full_id, dtype=torch.bfloat16, attention_mask=attention_mask).logits
    return torch.log_softmax(logits[:, input_ids.shape[-1] - 1: -1], dim=-1, dtype=torch.float)


def policy_gradient_reinforce_loss(rollout_ids: torch.Tensor,
                                   rollout_mask: torch.Tensor,
                                   rollout_log_probs: torch.Tensor,
                                   reward: torch.Tensor) -> torch.Tensor:
    tlog_probs = rollout_log_probs.gather(dim=-1, index=rollout_ids.unsqueeze(-1)).squeeze(-1) * rollout_mask
    loss = -(reward.unsqueeze(-1) * tlog_probs).sum() / rollout_mask.sum()
    return loss


def verify_reward(completions: list[str]) -> torch.Tensor:
    with torch.no_grad():
        super_banana_love_mode = False
        if super_banana_love_mode:
            reward_tensor = torch.tensor([c.lower().count('banana') for c in completions],
                                         dtype=torch.float).to('cuda')
        else:
            reward_tensor = torch.tensor([1 if 'banana' in c.lower() else 0 for c in completions]).to('cuda')
        print(reward_tensor.sum() / len(completions))
    return reward_tensor

def prepare_prompts(prompts: list[str], tokenizer: AutoTokenizer) -> list[str]:
    prompt_ids = []
    for p in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
        )["input_ids"]
        prompt_ids.append(list(ids))
    return prompt_ids

def banana_rate(engine, tokenizer, prompt: str, sampling_params: dict,
                n: int = 16, show: int = 3) -> float:
    ids = prepare_prompts([prompt] * n, tokenizer)
    outs = engine.generate(input_ids=ids, sampling_params=sampling_params)
    texts = [tokenizer.decode(o['output_ids'], skip_special_tokens=True) for o in outs]
    rate = sum('banana' in t.lower() for t in texts) / n
    print(f"[{prompt!r}] banana rate: {rate:.2f} ({n} samples)")
    for t in texts:
        print("  -", t.replace("\n", " "))
    return rate

def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to('cuda')

    rollout_model = sgl.Engine(
        model_path=MODEL,
        skip_tokenizer_init=True,
        #    enable_memory_saver=True,
        mem_fraction_static=0.1,
    )
    prompts = [ "name 3 yellow fruits that need to be peeled",
    "what fruit is high in potassium?",
    "what do monkeys like to eat?",
    "name a fruit that is curved",
    "what fruit is good before a workout?",
    "give me a simple smoothie recipe",
    "how do I make a quick bread with overripe fruit?",
    "name 3 tropical fruits",
    "what is a good fruit for babies?",
    "list some fruits that are easy to take on the go",
    "suggest a healthy snack",
    "what should I eat for breakfast?",
    "name something yellow",
    "suggest a dessert idea",
    "what can I put on top of pancakes?",
    "write a one-sentence story about a picnic",
    "what food would you bring on a hike?",
    "give me an idea for a kid's lunchbox",
    "what goes well with peanut butter?",
    "name a food that is sweet and soft",
    "what is a cheap and filling snack?",
    "suggest an ingredient for an ice cream sundae",]
    prompt_ids = prepare_prompts(prompts, tokenizer)

    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 128,
        "stop_token_ids": [tokenizer.eos_token_id],
    }
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    eval_prompt = "what you really love to eat in your life?"
    before = banana_rate(rollout_model, tokenizer, eval_prompt, sampling_params)

    for _ in range(80):
        rollout = rollout_model.generate(input_ids=prompt_ids, sampling_params=sampling_params)
        rollout_ids = []
        for r in rollout:
            rollout_ids.append(r['output_ids'])

        input_ids = pad_sequence([torch.tensor(p) for p in prompt_ids],
                                 batch_first=True,
                                 padding_value=tokenizer.pad_token_id,
                                 padding_side="left").to('cuda')

        rollout_ids = pad_sequence([torch.tensor(r) for r in rollout_ids],
                                   batch_first=True,
                                   padding_value=tokenizer.pad_token_id,
                                   padding_side="right").to('cuda')
        #print(tokenizer.decode(rollout_ids))
        input_mask = input_ids[:,] != tokenizer.pad_token_id
        rollout_mask = rollout_ids[:,] != tokenizer.pad_token_id
        attn_mask = torch.cat((input_mask, rollout_mask), dim=-1)

        rollout_log_probs = get_log_probs(input_ids, rollout_ids, attn_mask, model)
        reward = verify_reward([tokenizer.decode(r) for r in rollout_ids])

        loss = policy_gradient_reinforce_loss(rollout_ids,
                                              rollout_mask.to(dtype=torch.long),
                                              rollout_log_probs,
                                              reward)
        loss.backward()
        optim.step()
        optim.zero_grad()
        rollout_model.update_weights_from_tensor(
            named_tensors=[(n, p.detach().to(torch.bfloat16).cpu())
                           for n, p in model.named_parameters()],
            flush_cache=True,
        )

    after = banana_rate(rollout_model, tokenizer, eval_prompt, sampling_params)
    print(f"banana rate: {before:.2f} -> {after:.2f}")
