#!/usr/bin/env python3
"""
Deterministic benchmark for DeepSeek models on Gaudi2 hardware.
Uses fixed seeds and deterministic settings to ensure comparable results across runs.
"""

import os
import time
import torch
import numpy as np
import random
import argparse
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM as IPEXAutoModelForCausalLM

# Set fixed seeds for reproducibility
GLOBAL_SEED = 42

def set_seed(seed=GLOBAL_SEED):
    """Set seed for all random number generators for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Extra deterministic settings if needed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Random seed set to {seed} for deterministic results")

def format_time(seconds):
    """Format time in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def run_benchmark(
    model_name="deepseek-ai/deepseek-llm-7b-base",
    prompts=None,
    max_new_tokens=100,
    num_runs=3,
    use_hpu=None,
    seed=GLOBAL_SEED,
    verbose=True
):
    """
    Run deterministic benchmark for DeepSeek models.
    
    Args:
        model_name: HuggingFace model identifier or local path
        prompts: List of prompt strings to test
        max_new_tokens: Maximum number of new tokens to generate
        num_runs: Number of runs per prompt for averaging
        use_hpu: Explicitly set whether to use HPU (None = auto-detect)
        seed: Random seed for reproducibility
        verbose: Whether to print detailed logs
    
    Returns:
        Dictionary containing benchmark results
    """
    # Set seeds for reproducibility
    set_seed(seed)
    
    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "Explain the theory of relativity in simple terms:",
            "Write a Python function to calculate the Fibonacci sequence:",
            "Describe the process of photosynthesis in plants:"
        ]
    
    # Set environment variables for Gaudi optimization
    os.environ["PT_HPU_LAZY_MODE"] = "1"
    os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "1"
    
    # Configure device settings
    device_settings = {"device_map": "auto", "low_cpu_mem_usage": True}
    # if use_hpu is not None:
    #     # Only add HPU settings if explicitly requested
    #     device_settings["ipex_config"] = {
    #         "device": "hpu",
    #         "distributed": False
    #     }
    
    # Track timings
    results = {
        "model": model_name,
        "tokenizer_load_time": 0,
        "model_load_time": 0,
        "prompts": [],
        "seed": seed,
        "deterministic": True,
        "max_new_tokens": max_new_tokens,
        "num_runs": num_runs
    }
    
    # Step 1: Load tokenizer
    print(f"🔄 Loading tokenizer for {model_name}...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("ℹ️ Set pad_token to eos_token")
    
    results["tokenizer_load_time"] = time.time() - start_time
    print(f"✓ Tokenizer loaded in {format_time(results['tokenizer_load_time'])}")
    
    # Step 2: Load model
    print(f"🔄 Loading model {model_name}...")
    start_time = time.time()
    model = IPEXAutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        **device_settings
    )
    
    results["model_load_time"] = time.time() - start_time
    print(f"✓ Model loaded in {format_time(results['model_load_time'])}")
    
    # Print model device info if available
    if hasattr(model, 'hf_device_map'):
        print(f"📍 Model device map: {model.hf_device_map}")
    
    # Step 3: Run inference benchmarks
    print("\n📊 Running inference benchmarks...")
    
    # Warmup run (not counted in results)
    print("🔥 Performing warmup run...")
    warmup_input = tokenizer("This is a warmup prompt.", return_tensors="pt", padding=True)
    with torch.no_grad():
        _ = model.generate(
            warmup_input.input_ids,
            attention_mask=warmup_input.attention_mask,
            max_new_tokens=20,
            do_sample=False,  # Deterministic for reproducibility
            temperature=1.0,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Benchmark each prompt
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n🔄 Benchmark prompt {prompt_idx+1}/{len(prompts)}: '{prompt[:50]}...'")
        
        prompt_results = {
            "prompt": prompt,
            "runs": [],
            "avg_latency": 0,
            "avg_tokens_per_second": 0,
            "tokens_generated": 0,
            "std_dev_latency": 0
        }
        
        # Encode once outside the loop
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_length = len(inputs.input_ids[0])
        
        latencies = []
        tokens_generated = []
        outputs_text = []
        
        # Run multiple times for averaging
        for run in range(num_runs):
            # Reset seed for each run to ensure determinism
            set_seed(seed + run)
            
            print(f"  Run {run+1}/{num_runs}...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic for reproducibility
                    temperature=1.0,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Decode output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_length = len(outputs[0])
            tokens_gen = output_length - input_length
            
            # Store results
            latencies.append(latency)
            tokens_generated.append(tokens_gen)
            outputs_text.append(output_text)
            
            # Calculate tokens per second
            tokens_per_second = tokens_gen / latency if latency > 0 else 0
            
            # Store run results
            prompt_results["runs"].append({
                "latency": latency,
                "tokens_generated": tokens_gen,
                "tokens_per_second": tokens_per_second
            })
            
            print(f"  ✓ Generated {tokens_gen} tokens in {format_time(latency)}")
            print(f"  📊 Performance: {tokens_per_second:.1f} tokens/sec")
            
            # Verify determinism by checking if all outputs are identical
            if run > 0 and outputs_text[run] != outputs_text[0]:
                print("⚠️ WARNING: Non-deterministic output detected!")
                results["deterministic"] = False
        
        # Calculate statistics
        prompt_results["avg_latency"] = sum(latencies) / len(latencies)
        prompt_results["avg_tokens_per_second"] = sum(tokens_gen / lat for tokens_gen, lat in zip(tokens_generated, latencies)) / len(latencies)
        prompt_results["tokens_generated"] = sum(tokens_generated) / len(tokens_generated)
        prompt_results["std_dev_latency"] = np.std(latencies)
        
        # Add to overall results
        results["prompts"].append(prompt_results)
        
        # Print summary for this prompt
        print(f"\n📊 Prompt {prompt_idx+1} Summary:")
        print(f"  Avg latency: {format_time(prompt_results['avg_latency'])}")
        print(f"  Avg tokens/sec: {prompt_results['avg_tokens_per_second']:.2f}")
        print(f"  Std dev latency: {prompt_results['std_dev_latency']:.4f}s")
        
        # Print first few tokens of output
        if verbose:
            print(f"\n📝 Sample output (first 100 chars):\n  {outputs_text[0][:100]}...")
    
    # Calculate overall statistics
    if results["prompts"]:
        results["avg_latency"] = sum(p["avg_latency"] for p in results["prompts"]) / len(results["prompts"])
        results["avg_tokens_per_second"] = sum(p["avg_tokens_per_second"] for p in results["prompts"]) / len(results["prompts"])
    
    # Print overall summary
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Deterministic run: {'Yes' if results['deterministic'] else 'No'}")
    print(f"Tokenizer load time: {format_time(results['tokenizer_load_time'])}")
    print(f"Model load time: {format_time(results['model_load_time'])}")
    print(f"Average latency: {format_time(results['avg_latency'])}")
    print(f"Average tokens/sec: {results['avg_tokens_per_second']:.2f}")
    print("="*70)
    
    return results

def main():
    """Parse command line arguments and run benchmark"""
    parser = argparse.ArgumentParser(description="Run deterministic benchmark for DeepSeek models on Gaudi")
    
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-llm-7b-base",
                       help="DeepSeek model to benchmark")
    parser.add_argument("--prompt", type=str, action="append",
                       help="Test prompts (can be specified multiple times)")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs per prompt for averaging")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED,
                       help="Random seed for reproducibility")
    parser.add_argument("--use-hpu", action="store_true",
                       help="Explicitly use HPU settings")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output")
    
    args = parser.parse_args()
    
    run_benchmark(
        model_name=args.model,
        prompts=args.prompt,
        max_new_tokens=args.max_tokens,
        num_runs=args.runs,
        use_hpu=args.use_hpu,
        seed=args.seed,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()