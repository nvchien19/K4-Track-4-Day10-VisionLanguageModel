# E2 rank-16 training for the Colab T4 notebook.
# Run with: %run -i /content/e2_train.py
import gc, json, math, random, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'
RUN_DIR = Path('/content/K4-Track-4-Day10-VisionLanguageModel/work/e2_r16')
RUN_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_RUNS = Path('/content/drive/MyDrive/road_damage_lab_2A202602926')
for _name in ('m', 'model'):
    if _name in globals():
        del globals()[_name]
gc.collect()
torch.cuda.empty_cache()
assert torch.cuda.is_available(), 'T4 runtime is not connected'

# Keep all visual modules in fp16; only quantize the language model.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=['visual', 'model.visual', 'lm_head'],
)
base = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={'': 0},
)
base.config.use_cache = False
import bitsandbytes as bnb
visual_4bit = [
    (name, type(module).__name__)
    for name, module in base.model.visual.named_modules()
    if isinstance(module, bnb.nn.Linear4bit)
]
assert not visual_4bit, f'Vision layers were quantized: {visual_4bit[:5]}'
model, trainable = apply_setup(base, 'E2', rank=16)
del base
gc.collect()
torch.cuda.empty_cache()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={'use_reentrant': False}
)
model.config.use_cache = False
params = [parameter for parameter in model.parameters() if parameter.requires_grad]
assert all(parameter.dtype == torch.float32 for parameter in params)
optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.0, foreach=False)
steps, accum = 300, 8
warmup = max(1, int(steps * 0.05))
def lr_lambda(step):
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, steps - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda')
rows = list(splits['train'])
random.Random(42).shuffle(rows)
loader = DataLoader(
    rows,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: collate([encode_example(row) for row in batch]),
    drop_last=True,
)
iterator = iter(loader)
optimizer.zero_grad(set_to_none=True)
losses, step, micro = [], 0, 0
started = time.time()
torch.cuda.reset_peak_memory_stats()

def save_e2_checkpoint(final=False):
    adapter_dir = RUN_DIR / 'adapter'
    model.save_pretrained(adapter_dir, safe_serialization=True)
    extra = {
        name: parameter.detach().float().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (
            'visual.merger' in name or 'visual.deepstack_merger_list' in name
        )
    }
    torch.save(extra, RUN_DIR / 'extra_trainable.pt')
    meta = {
        'setup': 'E2',
        'rank': 16,
        'trainable_params': trainable,
        'peak_vram_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'train_minutes': (time.time() - started) / 60,
        'max_steps': step,
        'mean_last_loss': sum(losses[-80:]) / max(1, len(losses[-80:])),
        'dataset_version': 'd10-21e4070301',
        'checkpoint_local_only': not DRIVE_RUNS.is_dir(),
        'resolution_max_pixels': 320 * 180,
        'final': final,
    }
    (RUN_DIR / 'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if DRIVE_RUNS.is_dir():
        import shutil
        shutil.copytree(RUN_DIR, DRIVE_RUNS / RUN_DIR.name, dirs_exist_ok=True)
        print('checkpoint synced to Drive:', DRIVE_RUNS / RUN_DIR.name, flush=True)
    print('checkpoint', step, 'saved to', RUN_DIR, flush=True)

while step < steps:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    batch = {key: value.to('cuda:0') for key, value in batch.items()}
    with torch.autocast('cuda', dtype=torch.float16):
        loss = model(**batch).loss / accum
    if not torch.isfinite(loss):
        raise FloatingPointError(f'non-finite loss at step {step}, micro {micro}')
    scaler.scale(loss).backward()
    losses.append(float(loss.detach()) * accum)
    micro += 1
    if micro == accum:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f'non-finite grad norm at step {step + 1}')
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        micro = 0
        if step % 10 == 0:
            mean_loss = sum(losses[-80:]) / len(losses[-80:])
            print(
                f'E2 {step}/{steps} loss={mean_loss:.4f} '
                f'lr={scheduler.get_last_lr()[0]:.2e}',
                flush=True,
            )
        if step % 50 == 0:
            save_e2_checkpoint()
save_e2_checkpoint(final=True)
print('E2 complete:', json.dumps(json.loads((RUN_DIR / 'run.json').read_text()), ensure_ascii=False), flush=True)
