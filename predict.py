from cog import BasePredictor, Input, ConcatenateIterator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
from exllama.lora import ExLlamaLora
import os, glob
import torch

class Predictor(BasePredictor):

    def setup(self):
        model_directory = "models/TheBloke_Llama-2-7B-GPTQ_gptq-4bit-128g-actorder_True/"
        lora_directory = "loras/v2ray_LLaMA-2-Jannie-7B-QLoRA/"

        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        lora_config_path = os.path.join(lora_directory, "adapter_config.json")
        lora_path = os.path.join(lora_directory, "adapter_model.bin")

        config = ExLlamaConfig(model_config_path)
        config.model_path = model_path

        self.model = ExLlama(config)
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)
        self.cache = ExLlamaCache(self.model)
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        self.lora = ExLlamaLora(self.model, lora_config_path, lora_path)

    def predict(self,
                prompt: str = Input(description="Text prompt for the model", default="\n### Input: Hello, who are you?\n### Output:"),
                temperature: float = Input(description="Temperature of the output", default=0.5),
                top_p: float = Input(description="Top cumulative probability to filter candidates", default=1),
                top_k: int = Input(description="Number of top candidates to keep", default=20),
                max_new_tokens: int = Input(description="Maximum new tokens to generate", default=1024),
                seed: int = Input(description="Seed for reproducibility, -1 for random seed", default=-1),
                use_lora: bool = Input(description="Whether to use LoRA for prediction", default=True)
                ) -> ConcatenateIterator[str]:

        self.generator.settings.temperature = temperature
        self.generator.settings.top_p = top_p
        self.generator.settings.top_k = top_k

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
                # Generate a token
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
