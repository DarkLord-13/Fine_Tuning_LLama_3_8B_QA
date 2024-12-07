# Fine-Tuning LLama 3 8B for Question Answering

This project demonstrates fine-tuning the LLama 3 8B model for question-answering tasks using the Unsloth library and LoRA adapters to optimize training efficiency.

## Project Overview

The project involves the following steps:
1. **Environment Setup**:
    - Check GPU version and install compatible dependencies.
2. **Model Loading**:
    - Load a range of quantized language models, including the LLama-3 model, optimized for memory efficiency.
3. **Integration of LoRA Adapters**:
    - Integrate LoRA adapters to update a fraction of the model's parameters for faster training.
4. **Dataset Preparation**:
    - Create and format a dataset for question-answering tasks.
5. **Model Training**:
    - Configure and train the model using the prepared dataset.

## Dependencies

The project requires the following dependencies:
- Python 3.x
- Torch
- Unsloth
- Transformers
- Datasets

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/DarkLord-13/Machine-Learning-01.git
    ```

2. Navigate to the project directory:
    ```sh
    cd Machine-Learning-01
    ```

3. Install the required packages:
    ```sh
    pip install torch unsloth transformers datasets
    ```

## Usage

1. **Environment Setup**:
    - Check GPU version and install compatible dependencies.
    ```python
    import torch
    major_version, minor_version = torch.cuda.get_device_capability()
    !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    if major_version >= 8:
        !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
    else:
        !pip install --no-deps xformers trl peft accelerate bitsandbytes
    ```

2. **Model Loading**:
    - Load the LLama-3 model with 4-bit quantization.
    ```python
    from unsloth import FastLanguageModel
    max_seq_length = 2048
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )
    ```

3. **Integration of LoRA Adapters**:
    - Integrate LoRA adapters for efficient training.
    ```python
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    ```

4. **Dataset Preparation**:
    - Create and format a dataset for question-answering tasks.
    ```python
    from datasets import Dataset
    qa_dataset = [
        {"instruction": "What are the working hours?", "input": "", "output": "The standard working hours are from 9 AM to 6 PM, Monday to Friday."},
        {"instruction": "What is the dress code?", "input": "", "output": "The dress code is business casual from Monday to Thursday, and casual on Fridays."},
        # Add more examples as needed
    ]
    dataset = Dataset.from_list(qa_dataset)
    EOS_TOKEN = tokenizer.eos_token
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {}
    ### Input:
    {}
    ### Response:
    {}"""
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = [alpaca_prompt.format(instruction, input, output) + EOS_TOKEN for instruction, input, output in zip(instructions, inputs, outputs)]
        return {"text": texts}
    dataset = dataset.map(formatting_prompts_func, batched=True)
    ```

5. **Model Training**:
    - Configure and train the model using the prepared dataset.
    ```python
    from trl import SFTTrainer
    from transformers import TrainingArguments
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    ```

## Results

The fine-tuned LLama 3 8B model demonstrates enhanced performance on question-answering tasks, with efficient training facilitated by LoRA adapters and 4-bit quantization.

## License

This project is licensed under the MIT License.
