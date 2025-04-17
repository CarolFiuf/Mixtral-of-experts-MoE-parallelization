import os
import torch
import torch.nn.functional as F

from common import run_perftest, checkAllclose
from custom_moe import launch_custom_moe


def test_only():
    # Check if the script is run in a test environment
    return os.environ.get('TEST_ONLY', '0') == '1'


def benchmark_only():
    # Check if the script is run in a benchmark environment
    return os.environ.get('BENCHMARK_ONLY', '0') == '1'


def torch_moe(hidden_states,
              w1,  # [expert, inter_dim*2, model_dim]
              w2,  # [expert, model_dim, inter_dim]
              topk_weight, topk_ids):

    token_num = hidden_states.shape[0]
    topk = topk_weight.shape[1]
    expert, model_dim, inter_dim = w2.shape

    hidden_states = hidden_states.view(
        token_num, 1, model_dim).repeat(1, topk, 1)
    out = torch.zeros(
        (token_num, topk, model_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for E_id in range(expert):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
            act_out = F.silu(gate) * up
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(token_num, -1, 1)).sum(dim=1)


def should_be_better_moe(hidden_states,
                         w1,  # [expert, inter_dim*2, model_dim]
                         w2,  # [expert, model_dim, inter_dim]
                         topk_weight, topk_ids):
    return launch_custom_moe(hidden_states, w1, w2, topk_weight, topk_ids)


def test_moe(dtype, token, model_dim, inter_dim, E, topk):

    input = torch.randn((token, model_dim), dtype=dtype).cuda()
    w1 = torch.randn((E, inter_dim*2, model_dim), dtype=dtype).cuda()
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype).cuda()

    topk_weights = torch.randn((token, topk), dtype=dtype).cuda()
    topk_ids = torch.randint(0, E, (token, topk), dtype=torch.int32).cuda()

    num_iters = 5
    num_warmup = 2

    if test_only():
        out_ref = torch_moe(input,
                            w1,
                            w2,
                            topk_weights,
                            topk_ids)
        out = should_be_better_moe(input,
                                   w1,
                                   w2,
                                   topk_weights,
                                   topk_ids)
    else:
        out_ref, us_ref_naive = run_perftest(torch_moe,
                                             input,
                                             w1,
                                             w2,
                                             topk_weights,
                                             topk_ids,
                                             num_iters=num_iters,
                                             num_warmup=num_warmup)
        print(f'[perf] naive moe: {us_ref_naive:.2f} us')
        out, us = run_perftest(should_be_better_moe,
                               input,
                               w1,
                               w2,
                               topk_weights,
                               topk_ids,
                               num_iters=num_iters,
                               num_warmup=num_warmup)
        print(f'[perf] custom moe: {us:.2f} us')
        print(f'[perf] speedup {us_ref_naive/us-1:.2%}')

    if not benchmark_only():
        if dtype == torch.float32:
            checkAllclose(out_ref, out, rtol=1.3e-6, atol=1e-5)
        elif dtype == torch.float16:
            checkAllclose(out_ref, out, rtol=1e-3, atol=1e-5)
        elif dtype == torch.bfloat16:
            checkAllclose(out_ref, out, rtol=1.6e-2, atol=1e-5)
        else:
            raise ValueError(f"Unsupported dtype {dtype}")


if __name__ == "__main__":

    # Based on DeepSeek V3 configurations
    print("Test 01: dtype=float32, token=1, model_dim=7168, inter_dim=256, E=256, topk=8")
    test_moe(dtype=torch.float32, token=1,
             model_dim=7168, inter_dim=256, E=256, topk=8)
    print("Test 02: dtype=float32, token=2, model_dim=7168, inter_dim=256, E=256, topk=8")
    test_moe(dtype=torch.float32, token=2,
             model_dim=7168, inter_dim=256, E=256, topk=8)
    print("Test 03: dtype=float32, token=4, model_dim=7168, inter_dim=256, E=256, topk=8")
    test_moe(dtype=torch.float32, token=4,
             model_dim=7168, inter_dim=256, E=256, topk=8)
    print("Test 04: dtype=float32, token=8, model_dim=7168, inter_dim=256, E=256, topk=8")
    test_moe(dtype=torch.float32, token=8,
             model_dim=7168, inter_dim=256, E=256, topk=8)
    print("Test 05: dtype=float32, token=16, model_dim=7168, inter_dim=256, E=256, topk=8")
    test_moe(dtype=torch.float32, token=16,
             model_dim=7168, inter_dim=256, E=256, topk=8)
    print("Test 06: dtype=float32, token=32, model_dim=7168, inter_dim=256, E=256, topk=8")
    test_moe(dtype=torch.float32, token=32,
             model_dim=7168, inter_dim=256, E=256, topk=8)

    # Based on Mixtral-7B configurations
    print("Test 07: dtype=float32, token=32, model_dim=8192, inter_dim=6144, E=8, topk=2")
    test_moe(dtype=torch.float32, token=32,
             model_dim=8192, inter_dim=6144, E=8, topk=2)
    print("Test 08: dtype=float32, token=128, model_dim=8192, inter_dim=6144, E=8, topk=2")
    test_moe(dtype=torch.float32, token=128,
             model_dim=8192, inter_dim=6144, E=8, topk=2)

    # Based on Mixtral-13B configurations
    print("Test 09: dtype=float32, token=32, model_dim=8192, inter_dim=16384, E=8, topk=2")
    test_moe(dtype=torch.float32, token=32,
             model_dim=8192, inter_dim=16384, E=8, topk=2)
    print("Test 10: dtype=float32, token=128, model_dim=8192, inter_dim=16384, E=8, topk=2")
    test_moe(dtype=torch.float32, token=128,
             model_dim=8192, inter_dim=16384, E=8, topk=2)
