import torch
from utils import get_grad_norm, distribution_state_manager


class AttnWeightGrad(object):
    def __init__(self, model):
        super().__init__()

        self.model = model

        self.attention_weight_grad = []
        self.gradient_hooks = []
        self.forward_hooks = []
        self._register_hooks()

        self.cleanup_done = False

    def _register_hooks(self):
        def forward_hook(module, args, output):
            # output: (attn_output, attn_weights)
            attention_weight = output[1]
            
            # Register backward hook for capturing attention weight grads
            def gradient_hook(grad):
                # grad: (batch_size, num_heads, seq_len, seq_len)
                self.attention_weight_grad.append(grad.detach().clone()) 
            hook = attention_weight.register_hook(gradient_hook)
            self.gradient_hooks.append(hook)
            
            return output
        model_interface = distribution_state_manager(self.model)
        # Register hook for capturing attention weights
        for layer in model_interface.base_model.model.layers:
            hook = layer.self_attn.register_forward_hook(forward_hook)
            self.forward_hooks.append(hook)

    def compute_head_importance(self, sample_dataloader):
        if sample_dataloader is None:
            return

        model_interface = distribution_state_manager(self.model)
        num_layers = model_interface.base_model.config.num_hidden_layers
        num_heads = model_interface.base_model.config.num_attention_heads
        device = model_interface.base_model.device

        accumulators = [torch.zeros(num_heads, device=device) for _ in range(num_layers)]
        head_importance = {}

        for sample in sample_dataloader:
            # Get grad
            self.model.zero_grad()
            outputs = self.model(**sample)
            outputs.loss.backward()
            
            # Compute head importance
            for idx, grad in enumerate(self.attention_weight_grad):
                # grad_norm: (num_heads,)
                grad_norm = get_grad_norm(grad)
                accumulators[idx] += grad_norm

            self.attention_weight_grad.clear()
        
        for i in range(num_layers):
            accumulators[i] /= len(sample_dataloader)
            head_importance[f"layer_{i}"] = accumulators[i]

        return head_importance
    
    def force_cleanup(self):
        if not self.cleanup_done:
            # Remove all hook
            for hook in self.forward_hooks:
                hook.remove()
            self.forward_hooks.clear()
            
            for hook in self.gradient_hooks:
                hook.remove()
            self.gradient_hooks.clear()

            self.attention_weight_grad.clear()
            torch.cuda.empty_cache()
            self.cleanup_done = True

    def __del__(self):
        self.force_cleanup()
