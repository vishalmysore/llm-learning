from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch

checkpoint = "MBZUAI/LaMini-Flan-T5-783M"



tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                             device_map='auto',
                                             torch_dtype=torch.float32)

pipe = pipeline('text2text-generation',
                 model = base_model,
                 tokenizer = tokenizer,
                 max_length = 512,
                 do_sample=True,
                 temperature=0.3,
                 top_p=0.95,
                 )