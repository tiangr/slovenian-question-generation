import torch
from transformers import pipeline,AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model




model_name = 'doc2query/msmarco-14langs-mt5-base-v1'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # loading in 4 bit 
    bnb_4bit_quant_type="nf4", # quantization type
    bnb_4bit_use_double_quant=True, # nested quantization 
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)




#model.config.use_cache = False

#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token

# PEFT and Lora
dataset_name = 'thetian11/slosquad' 
dataset = load_dataset(dataset_name, split="train")


print(dataset['context'][0])



lora_alpha = 32
lora_dropout = 0.1
lora_r = 16

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



lora_model = get_peft_model(model, peft_config)


print_trainable_parameters(lora_model)

###################################
#Loading the trainer

from transformers import TrainingArguments
output_dir = "./results"
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
optim = "paged_adamw_32bit" #specialization of the AdamW optimizer that enables efficient learning in LoRA setting.
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"


training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    report_to="none"
)

from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="context",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)
