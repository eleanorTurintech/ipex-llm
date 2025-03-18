#!/usr/bin/env python3
"""
Custom test script for DeepSeek models on Gaudi with IPEX-LLM
This script focuses specifically on the transformers-based approach
rather than using the GGML models that the existing tests use.
"""

import os
import time
import sys
import torch
from transformers import AutoTokenizer

# Try to import ipex_llm modules
try:
    from ipex_llm.transformers import AutoModelForCausalLM as IPEXAutoModelForCausalLM
    print("✓ Successfully imported IPEX-LLM modules")
except ImportError as e:
    print(f"✗ Error importing IPEX-LLM modules: {e}")
    sys.exit(1)

class DeepseekGaudiTester:
    """Test utilities for DeepSeek models on Gaudi hardware"""
    
    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-base"):
        """Initialize the tester with model name"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        # Set environment variables
        os.environ["PT_HPU_LAZY_MODE"] = "1"
        
        # Track timings
        self.timings = {
            "tokenizer_load": 0,
            "model_load": 0,
            "inference": []
        }
        
        print(f"🚀 DeepSeek Gaudi Tester initialized with model: {model_name}")
    
    def test_tokenizer(self):
        """Test loading the tokenizer"""
        print(f"\n📝 Testing tokenizer loading for {self.model_name}")
        
        try:
            start_time = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("ℹ️ Set pad_token to eos_token")
                
            self.timings["tokenizer_load"] = time.time() - start_time
            print(f"✓ Tokenizer loaded successfully in {self.timings['tokenizer_load']:.2f}s")
            return True
        except Exception as e:
            print(f"✗ Tokenizer loading failed: {e}")
            return False
    
    def test_model_loading(self):
        """Test loading the model with IPEX-LLM"""
        if self.tokenizer is None:
            print("⚠️ Tokenizer not loaded, skipping model test")
            return False
        
        print(f"\n🔄 Loading model {self.model_name} with IPEX-LLM")
        
        try:
            # Detect available devices
            print("📊 Device detection:")
            
            if torch.cuda.is_available():
                print("  - CUDA is available")
            else:
                print("  - CUDA is not available")
            
            print("  - Using IPEX-LLM auto device mapping")
            
            # Configure model loading parameters
            model_kwargs = {
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16
            }
            
            # Try loading with HPU-specific configuration
            # try:
            #     print("  - Attempting to use HPU-specific configuration")
            #     model_kwargs["ipex_config"] = {
            #         "device": "hpu",
            #         "distributed": False  # Start with non-distributed for testing
            #     }
            # except Exception as e:
            #     print(f"  - HPU config not applied: {e}")
            
            # Load the model
            print("🔄 Starting model load...")
            start_time = time.time()
            self.model = IPEXAutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            self.timings["model_load"] = time.time() - start_time
            
            print(f"✓ Model loaded successfully in {self.timings['model_load']:.2f}s")
            
            # Try to print model device info
            if hasattr(self.model, 'hf_device_map'):
                print(f"📍 Model device map: {self.model.hf_device_map}")
            
            return True
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_inference(self, prompt="What is machine learning?", max_length=100):
        """Test inference with a simple prompt"""
        if self.model is None or self.tokenizer is None:
            print("⚠️ Model or tokenizer not loaded, skipping inference test")
            return False
        
        print(f"\n🧠 Testing inference with prompt: '{prompt}'")
        
        try:
            # Prepare input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True
            )
            
            print("🔄 Starting generation...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            inference_time = time.time() - start_time
            self.timings["inference"].append(inference_time)
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate tokens per second
            input_tokens = len(inputs.input_ids[0])
            output_tokens = len(outputs[0]) - input_tokens
            tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0
            
            print(f"✓ Generated {output_tokens} tokens in {inference_time:.2f}s")
            print(f"📊 Performance: {tokens_per_second:.1f} tokens/sec")
            print(f"\n📝 Response: '{response}'")
            
            return True
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self, prompts=None):
        """Run all tests in sequence"""
        print("\n" + "="*70)
        print("🧪 DEEPSEEK ON GAUDI TEST SUITE")
        print("="*70)
        
        # Default prompts if none provided
        if prompts is None:
            prompts = [
                "What is machine learning?",
                "Write a Python function to find prime numbers",
                "Explain the differences between CPU and GPU computing"
            ]
        
        # Run tests
        tokenizer_result = self.test_tokenizer()
        model_result = False
        inference_results = []
        
        if tokenizer_result:
            model_result = self.test_model_loading()
            
            if model_result:
                for i, prompt in enumerate(prompts):
                    print(f"\n🔄 Running inference test {i+1}/{len(prompts)}")
                    result = self.test_inference(prompt)
                    inference_results.append(result)
        
        # Print summary
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        print(f"✓ Tokenizer: {'Passed' if tokenizer_result else 'Failed'}")
        print(f"✓ Model Loading: {'Passed' if model_result else 'Failed'}")
        
        if inference_results:
            success_count = sum(1 for r in inference_results if r)
            print(f"✓ Inference: {success_count}/{len(inference_results)} tests passed")
            
            if self.timings["inference"]:
                avg_time = sum(self.timings["inference"]) / len(self.timings["inference"])
                print(f"  - Average inference time: {avg_time:.2f}s")
        
        return all([tokenizer_result, model_result] + inference_results)

def main():
    """Run the test suite with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DeepSeek models on Gaudi hardware")
    parser.add_argument("--model", "-m", type=str, default="deepseek-ai/deepseek-llm-7b-base",
                        help="DeepSeek model to test")
    parser.add_argument("--prompt", "-p", type=str, action="append",
                        help="Test prompts (can be specified multiple times)")
    
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = DeepseekGaudiTester(model_name=args.model)
    tester.run_all_tests(prompts=args.prompt)

if __name__ == "__main__":
    main()