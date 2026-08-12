from dataset.hard_policy import HardPolicy
from dataset.main import run_mode_trainval

gen_ds = run_mode_trainval(
    0.8,
    0.2,
    hard_policy=HardPolicy.PASSTHROUGH,
    should_flatten_packs=True,
    print_stats=True,
)

clf_ds = run_mode_trainval(
    0.8,
    0.2,
    hard_policy=HardPolicy.DOMINATE,
    should_flatten_packs=True,
    print_stats=True,
)

gen_train_paths = [v["path"] for v in gen_ds["train"]]
clf_val_paths = [v["path"] for v in clf_ds["val"]]
leak_counter = 0

for p in clf_val_paths:
    if p in gen_train_paths:
        leak_counter += 1

leak_rate = float(leak_counter) / float(len(clf_val_paths))
print(f"Leak rate: {leak_rate}")
