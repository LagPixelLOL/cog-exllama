from cog import BasePredictor, Input

from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator

import os, glob

class Predictor(BasePredictor):

    def setup(self):
        model_directory =  "models/TheBloke_Llama-2-7B-GPTQ_gptq-4bit-128g-actorder_True/"

        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        config = ExLlamaConfig(model_config_path)
        config.model_path = model_path

        self.model = ExLlama(config)
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)

        cache = ExLlamaCache(self.model)
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, cache)

        self.generator.disallow_tokens([self.tokenizer.eos_token_id])

        self.generator.settings.token_repetition_penalty_max = 1.2
        self.generator.settings.temperature = 0.95
        self.generator.settings.top_p = 0.65
        self.generator.settings.top_k = 100
        self.generator.settings.typical = 0.5

    def predict(self, prompt: str = Input(description="Input prompt for text generation", default="OwO is")) -> str:
        output = self.generator.generate_simple(prompt, max_new_tokens = 8)
        return output[len(prompt):]
