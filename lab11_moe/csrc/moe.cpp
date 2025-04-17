#include <pybind11/pybind11.h>
#include <torch/torch.h>

// TODO: Include necessary headers for your kernel

// TODO: Define more functions here

// TODO: Optimize this function
torch::Tensor launch_custom_moe(
    torch::Tensor hidden_states, torch::Tensor w1, torch::Tensor w2,
    torch::Tensor topk_weight, torch::Tensor topk_ids) {

    int token_num = hidden_states.size(0);
    int topk = topk_weight.size(1);
    int expert = w2.size(0);
    int model_dim = w2.size(1);
    int inter_dim = w2.size(2);

    hidden_states = hidden_states.view({token_num, 1, model_dim}).repeat({1, topk, 1});
    auto out = torch::zeros({token_num, topk, model_dim}, hidden_states.dtype()).to(hidden_states.device());

    // TODO: This part should not be implemented with Pytorch
    
    // TODO: Write kernel(s) for this part
    for (int E_id=0;E_id<expert;E_id++){
        auto mask = (topk_ids == E_id);
        if (mask.sum().item<int>() > 0) {
            auto sub_tokens = hidden_states.masked_select(mask.unsqueeze(-1)).view({-1, model_dim});
            auto act_input = sub_tokens.matmul(w1[E_id].transpose(0, 1));
            auto gate_up = act_input.split({inter_dim, inter_dim}, -1);
            auto act_out = torch::silu(gate_up[0]).mul(gate_up[1]);
            out.masked_scatter_(mask.unsqueeze(-1), act_out.matmul(w2[E_id].transpose(0, 1)));
        }
    }
    out = (out * topk_weight.view({token_num, -1, 1})).sum(1);

    return out;
}

PYBIND11_MODULE(custom_moe, m) {
    m.def("launch_custom_moe", &launch_custom_moe, "Launch the moe HIP kernel with PyTorch tensors.");
}
