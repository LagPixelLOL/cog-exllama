from cog import BasePredictor, Input, ConcatenateIterator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
from exllama.lora import ExLlamaLora
import os, glob
import torch

class Predictor(BasePredictor):

    def setup(self):
        model_directory = "models/TheBloke_airoboros-l2-70B-gpt4-1.4.1-GPTQ_gptq-4bit-128g-actorder_True/" # Modify this to your own model
        lora_directory = "loras/v2ray_LLaMA-2-Jannie-70B-QLoRA/" # Modify this to your own lora

        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        lora_config_path = os.path.join(lora_directory, "adapter_config.json")
        lora_path = os.path.join(lora_directory, "adapter_model.bin")

        config = ExLlamaConfig(model_config_path)
        config.model_path = model_path

        config.max_seq_len = 4096 # Max context length according to the model
        config.compress_pos_emb = config.max_seq_len // 2048
        config.fused_attn = False # Disable fused attention for LLaMA 2 70B to load, you can set this to True if you are not using LLaMA 2 70B

        self.using_fl_at = False
        config.use_flash_attn_2 = True
        config.max_input_len = config.max_seq_len
        self.using_fl_at = config.use_flash_attn_2

        # self.using_fl_at = False
        # try: # Enable flash attention
        #     from flash_attn import flash_attn_func
        #     config.use_flash_attn_2 = True
        #     config.max_input_len = config.max_seq_len
        #     self.using_fl_at = config.use_flash_attn_2
        #     print("Found flash attention, set use_flash_attn_2 to True.")
        # except:
        #     pass

        self.model = ExLlama(config)
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)
        self.cache = ExLlamaCache(self.model)
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        self.lora = ExLlamaLora(self.model, lora_config_path, lora_path)

    def predict(
        self,
        prompt: str = Input(description="Text prompt for the model", default="USER: Hello, who are you?\nASSISTANT:"),
        temperature: float = Input(description="Temperature of the output, it's best to keep it below 1", default=0.5, ge=0.01, le=2),
        top_p: float = Input(description="Top cumulative probability to filter candidates", default=1, ge=0.01, le=1),
        top_k: int = Input(description="Number of top candidates to keep", default=20, ge=1, le=100),
        repetition_penalty: float = Input(description="Penalty for repeated tokens in the model's output", default=1.15, ge=1, le=1.5),
        max_new_tokens: int = Input(description="Maximum new tokens to generate", default=1024, ge=1, le=4096), # Set the less than or equal to value according to the max context length of the model
        min_new_tokens: int = Input(description="Minimum new tokens to generate", default=0, ge=0, le=4096), # Set the less than or equal to value according to the max context length of the model
        seed: int = Input(description="Seed for reproducibility, -1 for random seed", default=-1, ge=-1, le=2**32-1),
        use_lora: bool = Input(description="Whether to use LoRA for prediction", default=False),
    ) -> ConcatenateIterator[str]:
        if min_new_tokens > max_new_tokens:
            raise ValueError("min_new_tokens must be smaller than or equal to max_new_tokens.")
        if self.using_fl_at:
            print("Using flash attention 2.")
        self.generator.settings.temperature = temperature
        self.generator.settings.top_p = top_p
        self.generator.settings.top_k = top_k
        self.generator.settings.token_repetition_penalty_max = repetition_penalty
        if use_lora:
            self.generator.lora = self.lora
        else:
            self.generator.lora = None
        if seed != -1:
            torch.manual_seed(seed)
        self.generator.gen_begin(self.tokenizer.encode(prompt))
        try:
            last_str = prompt
            
            for i in range(max_new_tokens):
                if i < min_new_tokens:
                    self.generator.disallow_tokens([self.tokenizer.eos_token_id])
                else:
                    self.generator.disallow_tokens(None)
                    
                gen_token = self.generator.beam_search()
                text_generated = self.tokenizer.decode(self.generator.sequence_actual[0])
                new_text = text_generated[len(last_str):]
                last_str = text_generated
                
                if new_text != "":
                    yield new_text
                    
                if gen_token.item() == self.tokenizer.eos_token_id: 
                    break
        finally:
            self.generator.end_beam_search()
