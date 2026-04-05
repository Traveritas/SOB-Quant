import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Tuple

# Add yaqa-quantization to path
sys.path.append(os.path.join(os.path.dirname(__file__), "yaqa-quantization"))

from lib.codebook import bitshift
from lib.algo import ldlq

def evaluate_perplexity_sliding_window(
    model,
    tokenizer,
    text: str,
    device: str,
    stride: int = 512,
    max_eval_tokens: int = None,
    tag: str = "eval",
):
    print(f"Start PPL Eval | tag={tag} stride={stride}")
    t0 = time.perf_counter()

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"][0]
    if max_eval_tokens is not None:
        input_ids = input_ids[:max_eval_tokens]

    input_ids = input_ids.to(device)
    max_length = getattr(model.config, "max_position_embeddings", 2048)

    nlls = []
    prev_end_loc = 0
    total_target_tokens = 0

    for begin_loc in range(0, input_ids.size(0), stride):
        end_loc = min(begin_loc + max_length, input_ids.size(0))
        trg_len = end_loc - prev_end_loc
        input_ids_chunk = input_ids[begin_loc:end_loc].unsqueeze(0)
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        total_target_tokens += trg_len
        prev_end_loc = end_loc
        if end_loc == input_ids.size(0):
            break

    ppl = torch.exp(torch.stack(nlls).sum() / total_target_tokens)
    elapsed = time.perf_counter() - t0
    print(f"[{tag}] PPL: {ppl.item():.4f} | Time: {elapsed:.2f}s")
    return ppl.item()

def get_wikitext2(tokenizer):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    return testenc, trainenc

class MultiLinearInputCollector:
    def __init__(self, max_tokens: int):
        self.max_tokens = int(max_tokens)
        self.collected: Dict[str, List[torch.Tensor]] = {}
        self.num_tokens: Dict[str, int] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer_name: str):
        def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            current = self.num_tokens.get(layer_name, 0)
            if current >= self.max_tokens:
                return
            hidden_states = inputs[0].detach()
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            remaining = self.max_tokens - current
            if flat.shape[0] > remaining:
                flat = flat[:remaining]
            self.collected.setdefault(layer_name, []).append(flat.cpu())
            self.num_tokens[layer_name] = current + flat.shape[0]

        return _hook

    def register(self, modules: Dict[str, nn.Module]) -> None:
        for layer_name, module in modules.items():
            self.collected[layer_name] = []
            self.num_tokens[layer_name] = 0
            self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def get_matrices(self) -> Dict[str, torch.Tensor]:
        matrices: Dict[str, torch.Tensor] = {}
        for layer_name, chunks in self.collected.items():
            if not chunks:
                raise RuntimeError(f"No inputs were collected for layer: {layer_name}")
            X = torch.cat(chunks, dim=0)
            matrices[layer_name] = X.contiguous()
        return matrices

def capture_calibration_inputs(model, trainenc, device, max_tokens=4096):
    print("Collecting calibration inputs...")
    target_modules = {}
    for i, layer in enumerate(model.model.decoder.layers):
        target_modules[f"layer_{i}"] = layer
        
    collector = MultiLinearInputCollector(max_tokens=max_tokens)
    collector.register(target_modules)
    
    input_ids = trainenc["input_ids"][0]
    max_length = getattr(model.config, "max_position_embeddings", 2048)
    
    with torch.no_grad():
        for start in range(0, input_ids.numel(), max_length):
            end = min(start + max_length, input_ids.numel())
            chunk = input_ids[start:end].unsqueeze(0).to(device)
            model(input_ids=chunk)
            if all(collector.num_tokens.get(name, 0) >= max_tokens for name in target_modules):
                break
                
    collector.remove()
    return collector.get_matrices()

@torch.no_grad()
def quantize_opt_with_yaqa(model, trainenc, device, args):
    model.config.use_cache = False
    layers = model.model.decoder.layers

    cb = bitshift.bitshift_codebook(
        L=args.L,
        K=args.K,
        V=args.V,
        tlut_bits=args.tlut_bits,
        decode_mode=args.decode_mode
    ).to(device)
    
    print(f"Yaqa/QTIP Codebook. Mode={args.decode_mode}, L={args.L}, K={args.K}, V={args.V}")

    layer_mats = capture_calibration_inputs(model, trainenc, device, max_tokens=4096)

    for i, layer in enumerate(tqdm(layers, desc="Quantizing Layers")):
        layer = layer.to(device)
        inps = layer_mats[f"layer_{i}"].to(device)
        # Compute H_in = X^T X
        H_in = (inps.T @ inps).to(torch.float32)

        linears = {
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "out_proj": layer.self_attn.out_proj,
            "fc1": layer.fc1,
            "fc2": layer.fc2
        }
        
        # YAQA expects (A, B) where HatW = A W_q B. We use Identity since QTIP/YAQA 2-sided is optional.
        # Wait, if we use LDLQ_2hess we need H_out. But if we don't have H_out, we can compute LDLQ_1hess
        from lib.algo.ldlq import LDLQ_2hess
        
        # Identity matrix for B (right side)
        for name, linear in linears.items():
            W = linear.weight.data.clone().to(device).to(torch.float32)
            # Create dummy H_out 
            H_out = torch.eye(W.shape[0], device=device, dtype=torch.float32)
            
            algo = LDLQ_2hess(
                W=W,
                H_in=H_in,
                H_out=H_out,
                cb=cb,
                out=None, # optional out tensor
                H_in_is_X=False,
                A=None, B=None, # Not modifying via external scales
                is_A_scale=False,
                is_B_scale=False,
                quant_group_size=W.shape[1], # No grouping by default
                hess_group_size=W.shape[1],
                save_meta=False,
                quant_type="yaqa",
            )
            
            W_bar, _ = algo.run()
            
            # W_bar is the reconstructed weight. We just overwrite the weights for now for PPL evaluation
            linear.weight.data = W_bar.to(linear.weight.dtype)
        
        layer = layer.cpu()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--L", type=int, default=16, help="Trellis quantization L")
    parser.add_argument("--K", type=int, default=16, help="Trellis K")
    parser.add_argument("--V", type=int, default=8, help="Trellis V")
    parser.add_argument("--tlut_bits", type=int, default=16)
    parser.add_argument("--decode_mode", type=str, default="dp", choices=["dp", "greedy"], help="DP decoding or greedy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device)

    print("Loading Wikitext-2...")
    testenc, trainenc = get_wikitext2(tokenizer)
    test_text = tokenizer.decode(testenc["input_ids"][0])

    print("--- Before Quantization ---")
    evaluate_perplexity_sliding_window(model, tokenizer, test_text, device, max_eval_tokens=8192, tag="FP16")

    quantize_opt_with_yaqa(model, trainenc, device, args)

    print("--- After Quantization ---")
    evaluate_perplexity_sliding_window(model, tokenizer, test_text, device, max_eval_tokens=8192, tag="YAQA")

if __name__ == "__main__":
    main()
