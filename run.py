# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Last update: Lucia Licakova, 2025-09-10

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from coconut import Coconut
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed


def main():

    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()
    dist_is_initialized = False
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        dist_is_initialized = True
        # NCCL is GPU-only
        dist.init_process_group(backend="nccl")
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        # Single GPU or CPU
        rank = 0
        world_size = 1
        local_rank = 0


    # Load the configuration file
    with open(args.config_file, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)
    # Wrap its values into a Config object for access throughout the script
    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    # Handle situations where training was interrupted
    if len(cur_ckpts) > 0 and not configs.only_eval:
        # If checkpoints exist and we're not just running evaluation,
        # it means the previous run was interrupted.
        # We need to find the latest checkpoint and resume from that.

        if rank == 0:
            # Warning on the main process
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )
        # Collect all directories named "checkpoint_#"
        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        # Sort them by the epoch number and pick the most recent one
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        # Update to the corresponding epoch number
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)
        # Assign the latest checkpoint path to configs.load_model_path
        configs.load_model_path = load_dir
        # Will load the model from the checkpoint and continue training from where it left off
        print(f"Loading from previous run epoch_{configs.resume}!")

    # If no such checkpoint exists but the config’s resume value is not 0,
    # it means the user explicitly asked to skip the first few epochs
    elif configs.resume != 0:
        # Skipping epochs without a checkpoint to load doesn't make sense
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
        # Will load the specified checkpoint and skip the requested number of epochs
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    # Load a Hugging Face model and tokenizer based on the model ID in the config
    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    # Make all sequences in a batch the same length to stack them into a tensor
    # Since GPT needs some padding token when training in batches, we reuse the
    # end-of-sequence token which GPT knows
    tokenizer.pad_token = tokenizer.eos_token
    # Extend the tokenizer with markers that Coconut uses
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    # Convert into numeric IDs to be used inside the model
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded = False

    # Replace torch.load with safetensors-compatible loading
    if configs.load_model_path is not None and configs.load_model_path != "None":
        try:
            # Use from_pretrained if the checkpoint is a directory
            model = AutoModelForCausalLM.from_pretrained(configs.load_model_path, use_safetensors=True)
            loaded = True
        except Exception as e:
            print("Failed to load via safetensors, trying torch.load fallback (not recommended on Windows):", e)
            # Tell PyTorch where to put the weights (cpu, cuda:0, etc.)
            saved_weights = torch.load(configs.load_model_path, map_location=torch.device(rank))
            # Load them into the model
            # strict=False: if some weights in the checkpoint don’t match any model parameter, ignore them
            # and if some model parameters don’t have corresponding weights in the checkpoint, initialise them randomly
            model.load_state_dict(saved_weights, strict=False)
            loaded = True

    # In this case the model needs extra tokens
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # Expand the model’s embedding layer to match the tokenizer
        model.resize_token_embeddings(len(tokenizer))
        # Initialise embeddings for new tokens
        embeddings = model.get_input_embeddings()
        # Pick a known token as a reference to initialise the new token embeddings
        target_id = tokenizer.convert_tokens_to_ids("<<")
        # Copying the embedding vector of << instead of random initialisation stabilises training
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id] 
            embeddings.weight.data[token_id] = target_embedding
            # The input embeddings and lm heads (output layer) are tied in GPT2. So the code below is not necessary
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        # Disable Coconut-specific logic
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        # Wrap the model in Coconut class
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    # With multiple GPUs, each process gets its own rank (GPU ID)
    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    # Only move to CUDA if available
    if torch.cuda.is_available():
        model = model.to(device)


    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer  # only shard llama's layers.
        },
    )
    # If the config enables bf16 (only on GPUs), convert the model to bfloat16 precision
    if configs.bf16:
        model.to(torch.bfloat16)


    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # GPU: If only eval, use DistributedDataParallel (to avoid bugs in fsdp)
        if configs.only_eval:
            parallel_model = DDP(model, device_ids=[rank])
        # GPU: Fully Sharded Data Parallel shards (splits) the model’s layers across devices to save memory
        else:
            parallel_model = FSDP(
                model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank
            )
    else:
        # CPU: no FSDP
        parallel_model = model

    # Delete the original model object since it’s now wrapped inside parallel_model
    del model
    # Only the first process prints the structure of the wrapped model, so logs don’t get cluttered
    if rank == 0:
        print(parallel_model)

    # Prepare the validation dataset configs.val_path, collect the questions
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    # Clean the ground truth answer
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    # Collect the chain-of-thought steps (reasoning paths), join them into one string per example
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]
    # Tokenise the validation set using the model’s tokenizer
    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )
    # If training is enabled, also tokenise the training set
    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )
    # How many tokens the model can generate during evaluation
    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0
    # If doing a real training run on main process, initialise a Weights & Biases run for logging
    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        # Save config parameters and prepare a table for logging generated text
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None
    # Set up the AdamW optimizer with the model’s parameters and hyperparameters from config
    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0
    # Take raw tokenised samples and batch them, padding sequences and preparing labels (with ignored tokens set to -100 so they don’t affect the loss)
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # Main training & evaluation loop (if resume > 0, skip earlier epochs (training is resumed))
    for epoch in range(configs.resume, configs.num_epochs):
    
        scheduled_stage = (
            # Only if doing multi-stage latent training, the stage increases every epochs_per_stage epochs
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        # Load the validation questions for generation evaluation
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )
        # Wrap them in a PyTorch DataLoader for batch processing
        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            # in distributed runs each process gets its own slice of the dataset
            sampler=None,
        )

        if not configs.only_eval:
            # Load the training data with chain-of-thought (or latent) annotations
            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )
            # Wrap it in a DataLoader for batch training
            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                # shuffling has to be set here because we don't use DistributedSampler
                shuffle=True,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=None,
            )
            # Prepare another version for loss evaluation
            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                # no shuffle even in DistributedSampler
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=None,
            )
            # Reset the optimizer based on configs
            if configs.reset_optimizer:
                del optimizer

                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )
            # Train the object whether it's wrapped for distributed
            # training (parallel_model.module) or not (parallel_model)
            getattr(parallel_model, "module", parallel_model).train()
            # Number of mini-batches in one epoch // how many batches to process before updating weights (optimizer.step()) 
            # Number of optimizer updates per epoch
            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            # Show progress bar
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):
            # Log the first batch
                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    # Number of samples in this batch
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    # For each sample and each token
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            # Token ID, label ID, decoded token (human-readable text)
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        # Separate samples
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    # Add for inspection
                    wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1
                # Move tensors in the batch (input IDs, attention masks, labels, etc.) to the device
                # Exclude "idx" because that’s just metadata.
                batch = {
                    key: batch[key].to(device) for key in batch.keys() if key != "idx"
                }
                # Run the model on this batch to get outputs
                outputs = parallel_model(**batch)

                # Compute loss (gradients will be accumulated over multiple mini-batches before an optimizer step)
                # Example: desired batch size = 16, CPU can only process 4 samples at a time -> gradient_accumulation_steps = 4
                # Process 4 mini-batches and accumulate gradients before updating the model.
                # Without dividing, the gradient would be 4x larger than intended.
                loss = outputs.loss / configs.gradient_accumulation_steps
                # Compute gradients
                loss.backward()

                # Only update the model after enough steps to match the real batch size
                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    # Update weights
                    optimizer.step()
                    # Reset gradients and update the progress bar
                    optimizer.zero_grad()
                    pbar.update(1)
                # Only main process logs
                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        # Multiply loss to get the true loss for the batch
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            # Wrap for safety in case of CPU
            if torch.distributed.is_initialized():
                # If distributed training is active, ensure all processes wait until everyone reaches this point
                dist.barrier()
            # During training and not in debug mode
            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                # Extract model weights
                states = parallel_model.state_dict()
                if rank == 0:
                    torch.save(
                        states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                    )
                    print("saving model.")

                if torch.distributed.is_initialized():
                    dist.barrier()
                # Cleanup and garbage collection
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # Validation loss
            total_loss = 0
            # Disable gradient computation during validation
            with torch.no_grad():
                # Switch to evaluation mode (either wrapped or not)
                getattr(parallel_model, "module", parallel_model).eval()
                # Iterate through validation batches and compute loss
                for step, batch in enumerate(valid_loss_dataloader):

                    batch = {
                        key: batch[key].to(device) for key in batch.keys() if key != "idx"
                    }

                    outputs = parallel_model(**batch)
                    loss = outputs.loss
                    if torch.distributed.is_initialized():
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    # Get the average per process
                    total_loss += loss.item() / world_size
                # Log the validation loss
                if wandb_run and rank == 0:

                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        # Generation accuracy evaluation
        total_length = len(valid_gen_dataloader)
        # Progress bar
        pbar = tqdm(
            colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
        )
        # Counters
        device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
        cor, cor_cot, total = (
            # How many answers match the ground truth
            torch.tensor(0, device=device),
            # How many chain-of-thought outputs match the reference
            torch.tensor(0, device=device),
            # Total examples processed
            torch.tensor(0, device=device),
        )

        with torch.no_grad():
            # Handle both wrapped and not wrapped models
            getattr(parallel_model, "module", parallel_model).eval()
            for idx, batch in enumerate(valid_gen_dataloader):
                # A tensor that stores the original index of this example in the validation dataset
                # Which row in the original dataset this batch corresponds to
                test_idx = batch["idx"][0]

                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if v != None and k not in ["idx", "position_ids"]
                }

                assert len(batch["input_ids"]) == 1
                # Extract the ground-truth answer, CoT reasoning, question
                answer = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]
                question = question_val[test_idx.cpu().item()]

                total += 1

                # Determine if synced_gpus should be True
                use_synced_gpus = (
                    not configs.only_eval 
                    and torch.cuda.is_available() 
                    and torch.cuda.device_count() > 1 
                    and torch.distributed.is_initialized()
                )
                # Generate text tokens for the input question (use the correct model)
                outputs = getattr(parallel_model, "module", parallel_model).generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    synced_gpus=use_synced_gpus,
                )
                
                # Convert token IDs to readable text
                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract the final answer after a # marker
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                # Extract CoT reasoning from the generated text
                cot_output = (
                    ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                )

                if idx < 5 and rank == 0:
                    # print some examples
                    print(
                        f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                    )
                    print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted Output: '{answer_output}'")

                cor += answer_output == answer
                cor_cot += cot_output == answer_cot

                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                )

            pbar.close()
            print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")

        if torch.distributed.is_initialized():
            # Combine results from multiple GPUs if running distributed training
            dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
            dist.all_reduce(cor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        # Convert tensors to numbers
        cor_cot = cor_cot.item()
        cor = cor.item()
        total = total.item()
        if rank == 0:
            print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
            print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        sys.stdout.flush()

        if wandb_run:
            wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})
        # If only_eval, exit after evaluation
        if configs.only_eval:
            break

        if torch.distributed.is_initialized():
            dist.barrier()
        # Save a new checkpoint only if this validation accuracy is better than previous best
        if (
            cor / total > best_acc
            and configs.save_only_improve
            and not configs.debug
            and not configs.only_eval
        ):
            states = parallel_model.state_dict()

            if rank == 0:
                torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                print("saving model.")

            best_acc = cor / total

            if torch.distributed.is_initialized():
                dist.barrier()
            # Cleanup
            del states
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
