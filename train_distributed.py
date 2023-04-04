import math
import multiprocessing
import os
from itertools import chain

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)

from palm_rlhf_pytorch import PaLM

# constants


class CFG:
    BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATE_EVERY: int = 4
    SEED: int = 42
    LEARNING_RATE: float = 3e-4
    SEQ_LEN: int = 8192
    NUM_CPU: int = multiprocessing.cpu_count()
    MIXED_PRECISION: str = "bf16"
    RESUME_FROM_CHECKPOINT: str = None
    CHECKPOINTING_STEPS: int = 1000
    OUTPUT_DIR: str = "palm"
    VALIDATION_STEPS: int = 100
    ENTITY_NAME: str = "a_man_chooses"


# helpers


def print_num_params(model, accelerator: Accelerator):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters in model: {n_params}")


# dataloaders


def build_dataloaders(accelerator: Accelerator):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_dataset = load_dataset("EleutherAI/the_pile", "all", split="train")
    val_dataset = load_dataset("EleutherAI/the_pile", "all", split="validation")

    #remove_column_name = ["url", "timestamp"]
    remove_column_name = ["meta"]

    train_dataset = train_dataset.remove_columns(remove_column_name)
    val_dataset = val_dataset.remove_columns(remove_column_name)

    train_dataset_shuffle = train_dataset.shuffle(seed=CFG.SEED)
    val_dataset_shuffle = val_dataset.shuffle(seed=CFG.SEED)

    text_column_name = "text"

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        train_tokenized_datasets = train_dataset_shuffle.map(
            tokenize_function,
            batched=True,
            num_proc=CFG.NUM_CPU,
            remove_columns=[text_column_name],
        )
        val_tokenized_datasets = val_dataset_shuffle.map(
            tokenize_function,
            batched=True,
            num_proc=CFG.NUM_CPU,
            remove_columns=[text_column_name],
        )

    block_size = CFG.SEQ_LEN

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        train_tokenized_dataset = train_tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=CFG.NUM_CPU,
        )
        val_tokenized_dataset = val_tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=CFG.NUM_CPU,
        )

    train_loader = DataLoader(
        train_tokenized_dataset,
        shuffle=True,
        batch_size=CFG.BATCH_SIZE,
        collate_fn=default_data_collator,
    )
    val_loader = DataLoader(
        val_tokenized_dataset,
        shuffle=False,
        batch_size=CFG.BATCH_SIZE,
        collate_fn=default_data_collator,
    )

    return train_loader, val_loader


# helpers


def main():
    # accelerator

    accelerator = Accelerator(
        gradient_accumulation_steps=CFG.GRADIENT_ACCUMULATE_EVERY,
        mixed_precision=CFG.MIXED_PRECISION,
        log_with="wandb",
    )

    accelerator.init_trackers(
        project_name="palm",
        config={
            "batch_size": CFG.BATCH_SIZE,
            "gradient_accumulate_every": CFG.GRADIENT_ACCUMULATE_EVERY,
            "learning_rate": CFG.LEARNING_RATE,
            "seq_len": CFG.SEQ_LEN,
            "mixed_precision": CFG.MIXED_PRECISION,
            "validation_steps": CFG.VALIDATION_STEPS,
        },
        init_kwargs={"wandb": {"entity": CFG.ENTITY_NAME}},
    )

    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    # set seed

    set_seed(CFG.SEED)

    # instantiate palm

    model = PaLM(
        num_tokens=50304, dim=1024, depth=24, dim_head=128, heads=8, flash_attn=True
    )

    model = model.to(accelerator.device)

    print_num_params(model, accelerator)

    # dataloaders

    train_loader, val_loader = build_dataloaders(accelerator)

    # optimizer

    optim = AdamW(
        model.parameters(), 
        lr=CFG.LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Determine number of training steps

    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps: {max_train_steps}")

    # lr scheduler
    # We cant decide on an actual number
    NUM_WARMUP_STEPS = int(max_train_steps * 0.069420)
    accelerator.print(f"Num warmup steps: {NUM_WARMUP_STEPS}")

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=NUM_WARMUP_STEPS * CFG.GRADIENT_ACCUMULATE_EVERY,
        num_training_steps=max_train_steps * CFG.GRADIENT_ACCUMULATE_EVERY,
    )

    # prepare

    model, optim, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optim, train_loader, val_loader, lr_scheduler
    )

    # checkpoint scheduler
    accelerator.register_for_checkpointing(lr_scheduler)

    # I do not know why Huggingface recommends recalculation of max_train_steps

    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps recalculated: {max_train_steps}")

    # Total batch size for logging

    total_batch_size = (
        CFG.BATCH_SIZE * accelerator.num_processes * CFG.GRADIENT_ACCUMULATE_EVERY
    )
    accelerator.print(f"Total batch size: {total_batch_size}")

    # resume training

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if CFG.RESUME_FROM_CHECKPOINT:
        if CFG.RESUME_FROM_CHECKPOINT is not None or CFG.RESUME_FROM_CHECKPOINT != "":
            accelerator.print(f"Resuming from checkpoint {CFG.RESUME_FROM_CHECKPOINT}")
            accelerator.load_state(CFG.RESUME_FROM_CHECKPOINT)
            path = os.path.basename(CFG.RESUME_FROM_CHECKPOINT)
        training_difference = os.path.splitext(path)[0]

        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = (
            int(training_difference.replace("step_", ""))
            * CFG.GRADIENT_ACCUMULATE_EVERY
        )

    if CFG.RESUME_FROM_CHECKPOINT and resume_step is not None:
        train_loader = accelerator.skip_first_batches(train_loader, resume_step)
        completed_steps += resume_step
        progress_bar.update(resume_step)

    # training

    model.train()

    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            loss = model(batch, return_loss=True)
            accelerator.backward(loss)

            accelerator.log({"loss": loss.item()}, step=step)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()
            lr_scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if isinstance(CFG.CHECKPOINTING_STEPS, int):
            if completed_steps % CFG.CHECKPOINTING_STEPS == 0:
                output_dir = f"step_{completed_steps }"
                if CFG.OUTPUT_DIR is not None:
                    output_dir = os.path.join(CFG.OUTPUT_DIR, output_dir)
                accelerator.save_state(output_dir)

        # validation - I was following Lucidrains validation here...

        if step % CFG.VALIDATION_STEPS == 0:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    loss = model(batch, return_loss=True)
                    accelerator.log({"val_loss": loss.item()}, step=step)

        if completed_steps >= max_train_steps:
            break

    accelerator.end_training()

    # save final model

    if CFG.OUTPUT_DIR is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{CFG.OUTPUT_DIR}/final",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )


if __name__ == "__main__":
    main()
