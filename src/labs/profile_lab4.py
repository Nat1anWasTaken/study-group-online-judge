import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_from_disk
from transformers import GPT2Config, GPT2LMHeadModel, set_seed

from cce_lab4 import use_cce
from muon_lab4 import MuonAdamW

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
data = load_from_disk('/work/nat1andotxyz/lab4/c4-packed').with_format('torch')
inputs = data[rank * 64:(rank + 1) * 64]['input_ids'].to('cuda')
num_items = torch.tensor(inputs.numel() * dist.get_world_size(), device='cuda')
reference_gradients = None
reference_loss = None
results = []

for variant in ['O', 'P']:
    set_seed(42)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=50304, n_positions=1024, n_ctx=1024,
        n_embd=768, n_layer=12, n_head=12, n_inner=3072,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
    )).cuda()
    if variant == 'P':
        use_cce(model)
    matrices, qkv, decay, no_decay = [], [], [], []
    for name, parameter in model.named_parameters():
        if name.endswith('attn.c_attn.weight'):
            qkv.append(parameter)
        elif name.startswith('transformer.h.') and parameter.ndim == 2:
            matrices.append(parameter)
        elif parameter.ndim > 1:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    optimizer = MuonAdamW(matrices, decay, no_decay, 1e-3, 0.1, (0.9, 0.95), qkv)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    forward = torch.compile(wrapped)
    durations, losses = [], []
    for step in range(85):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = forward(input_ids=inputs, labels=inputs, num_items_in_batch=num_items).loss
            loss = loss * dist.get_world_size()
        loss.backward()
        torch.cuda.synchronize()
        if step == 0:
            if variant == 'O':
                reference_loss = loss.detach().clone()
                reference_gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
            else:
                error = torch.stack([
                    (p.grad - reference_gradients[name]).square().sum()
                    for name, p in model.named_parameters()
                ]).sum().sqrt()
                norm = torch.stack([g.square().sum() for g in reference_gradients.values()]).sum().sqrt()
                relative_error = (error / norm).item()
                loss_difference = (loss - reference_loss).abs().item()
                print(json.dumps({'rank': rank, 'loss_difference': loss_difference,
                                  'relative_gradient_error': relative_error}), flush=True)
                if loss_difference > 0.001 or relative_error > 0.02:
                    raise RuntimeError('CCE numerical comparison exceeded tolerance')
                reference_gradients = None
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        slowest = torch.tensor(elapsed, device='cuda')
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        durations.append(slowest.item())
        losses.append(loss.item())
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError(f'{variant}: nonfinite loss or gradient')
        if step == 4:
            torch.cuda.reset_peak_memory_stats()
    mean_step = statistics.mean(durations[5:])
    result = dict(
        variant=variant, startup_five_steps_seconds=sum(durations[:5]),
        mean_step_seconds=mean_step, median_step_seconds=statistics.median(durations[5:]),
        tokens_per_second=131072 / mean_step,
        estimated_steps_27min=(1620 - 25 - sum(durations[:5])) / mean_step,
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
        first_loss=losses[0], final_loss=losses[-1],
        block_step_seconds=[statistics.mean(durations[i:i + 20]) for i in range(5, 85, 20)],
    )
    results.append(result)
    if rank == 0:
        print(json.dumps(result), flush=True)
    if variant == 'P' and rank == 0:
        directory = Path('logs') / f'cce-checkpoint-{os.environ["SLURM_JOB_ID"]}'
        model.save_pretrained(directory)
        restored = GPT2LMHeadModel.from_pretrained(directory)
        if restored.config.model_type != 'gpt2':
            raise RuntimeError('Checkpoint is not standard GPT-2')
        del restored
    dist.barrier()
    del model, wrapped, forward, optimizer, matrices, qkv, decay, no_decay
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()

if rank == 0:
    speedup = results[0]['mean_step_seconds'] / results[1]['mean_step_seconds']
    output = {'results': results, 'speedup': speedup, 'numerics_passed': True}
    Path(f'logs/cce-profile-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({'speedup': speedup, 'numerics_passed': True}), flush=True)
dist.destroy_process_group()
