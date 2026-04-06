from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print(model.name_or_path) #