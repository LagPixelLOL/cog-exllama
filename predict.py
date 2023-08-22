from cog import BasePredictor, Input, ConcatenateIterator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
from exllama.lora import ExLlamaLora
import os, glob, time, re
import torch
import psutil

model_directory = "models/TheBloke_Airoboros-L2-70B-GPT4-m2.0-GPTQ_gptq-4bit-32g-actorder_True/" # Modify this to your own model
lora_directory = "loras/v2ray_LLaMA-2-Jannie-70B-QLoRA/" # Modify this to your own lora
model_max_context = 4096 # Max context length according to the model

class Predictor(BasePredictor):

    def setup(self):
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        lora_config_path = os.path.join(lora_directory, "adapter_config.json")
        lora_path = os.path.join(lora_directory, "adapter_model.bin")

        config = ExLlamaConfig(model_config_path)
        config.model_path = model_path

        config.max_seq_len = model_max_context
        config.compress_pos_emb = 1
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
        max_tokens: int = Input(description="Maximum tokens to generate", default=1024, ge=1, le=model_max_context),
        min_tokens: int = Input(description="Minimum tokens to generate", default=1, ge=0, le=model_max_context),
        seed: int = Input(description="Seed for reproducibility, -1 for random seed", default=-1, ge=-2147483648, le=2147483647),
        use_lora: bool = Input(description="Whether to use LoRA for prediction", default=False),
    ) -> ConcatenateIterator[str]:
        if min_tokens > max_tokens:
            raise ValueError("min_tokens must be smaller than or equal to max_tokens.")
        if self.using_fl_at:
            print("Using flash attention 2.")
        self.generator.settings.temperature = temperature
        self.generator.settings.top_p = top_p
        self.generator.settings.top_k = top_k
        self.generator.settings.token_repetition_penalty_max = repetition_penalty
        if seed != -1:
            torch.manual_seed(seed)
        if use_lora:
            self.generator.lora = self.lora
        else:
            self.generator.lora = None
        # self.generator.settings.beams = 1
        # self.generator.settings.beam_length = 1
        input_tokens = self.tokenizer.encode(prompt)
        input_token_count = input_tokens.shape[-1]
        if input_token_count > model_max_context:
            input_tokens = input_tokens[:, input_token_count - model_max_context + 32:]
            print(f"Trimmed prompt because its token count is higher than the model's max context({input_token_count} > {model_max_context}).")
        self.generator.gen_begin_reuse(input_tokens)
        try:
            start_time = time.time()
            token_count = max_tokens
            str_hex_subchars = ""
            highest_ram = current_ram_all()
            highest_vram = current_vram_all()
            self.generator.begin_beam_search()
            for i in range(max_tokens):
                per_token_start_time = time.time()
                if self.generator.gen_num_tokens() > model_max_context:
                    gen_prune_left_reuse(self.generator, self.generator.gen_num_tokens() - model_max_context + 32)
                    print(f"Trimmed prompt by 32 tokens because the generation reached the model's max context({model_max_context}).")

                if i < min_tokens:
                    self.generator.disallow_tokens([self.tokenizer.eos_token_id])
                else:
                    self.generator.disallow_tokens(None)

                gen_token = self.generator.beam_search() # Generate 1 token
                curr_ram = current_ram_all()
                if curr_ram > highest_ram:
                    highest_ram = curr_ram
                curr_vram = current_vram_all()
                if curr_vram > highest_vram:
                    highest_vram = curr_vram
                if gen_token.item() == self.tokenizer.eos_token_id:
                    token_count = i
                    break

                token_str = self.generator.tokenizer.tokenizer.IdToPiece(int(gen_token)).replace("▁", " ")
                text_generated = hex_to_utf8(token_str)
                if "�" in text_generated:
                    str_hex_subchars += token_str
                    new_text = hex_to_utf8(str_hex_subchars)
                    if "�" in new_text:
                        new_text = None
                    else:
                        str_hex_subchars = ""
                else:
                    new_text = text_generated
                    str_hex_subchars = ""

                time_used_ms = int((time.time() - per_token_start_time) * 1000)
                if new_text:
                    print(f"Time used: {time_used_ms}ms | Generated: \"{new_text}\"")
                    yield new_text
                else:
                    print(f"Time used: {time_used_ms}ms | Generated: one piece of an UTF-8 character")
                    

            total_time = time.time() - start_time
            print(f"Generated {token_count} tokens in {total_time:.5f} seconds({token_count / total_time:.5f} TPS).")
            print(f"(RAM(MB): {int(highest_ram)} / {int(supported_ram_all())} | VRAM(MB): {int(highest_vram)} / {int(supported_vram_all())})")
        finally:
            self.generator.end_beam_search()

def hex_to_utf8(hex_input):
    if isinstance(hex_input, str):
        hex_list = re.split(r'(<0x[\da-fA-F]{2}>)', hex_input)
    elif isinstance(hex_input, list):
        hex_list = hex_input
    else:
        raise ValueError("hex_input must be either a single string or a list of strings.")

    byte_list = []
    hex_pattern = re.compile(r'^<0x([\da-fA-F]{2})>$')
    
    for item in hex_list:
        hex_match = hex_pattern.match(item)
        if hex_match:
            hex_value = hex_match.group(1)
            int_value = int(hex_value, 16)
            byte_list.append(int_value)
        else:
            utf8_encoded_item = item.encode('utf-8')
            byte_list.extend(utf8_encoded_item)

    utf8_bytes = bytes(byte_list)
    utf8_string = utf8_bytes.decode('utf-8', errors='replace')
    return utf8_string

def gen_prune_left_reuse(generator, num_tokens, mask=None):
    num_tokens = min(num_tokens, generator.sequence_actual.shape[-1] - 1)
    if generator.in_beam_search:
        generator.end_beam_search()
        generator.sequence = generator.sequence[:, num_tokens:]
        generator.begin_beam_search()
    else:
        generator.sequence = generator.sequence[:, num_tokens:]
        generator.gen_begin_reuse(generator.sequence, mask=mask)

def current_ram_all():
    ram_info = psutil.virtual_memory()
    used_ram = ram_info.used / (1024 ** 2)
    return used_ram

def supported_ram_all():
    ram_info = psutil.virtual_memory()
    total_ram = ram_info.total / (1024 ** 2)
    return total_ram

def current_vram_all():
    total_vram = 0
    for dev_id in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.memory_allocated(dev_id)
        total_vram += gpu_mem / (1024 ** 2)
    return total_vram

def supported_vram_all():
    total_supported_vram = 0
    for dev_id in range(torch.cuda.device_count()):
        gpu_supported_mem = torch.cuda.get_device_properties(dev_id).total_memory
        total_supported_vram += gpu_supported_mem / (1024 ** 2)
    return total_supported_vram
