from nemo import lightning as nl
from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.evoplm.model.model import ESM2Config, ESM2Model
from bionemo.evoplm.data.tokenizer import get_tokenizer
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized, _verify_strategy_supports_compile

strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    ddp="megatron",
    find_unused_parameters=True,
    ckpt_parallel_load=True,
)

config = ESM2Config(
        params_dtype=get_autocast_dtype("bf16-mixed"),
        pipeline_dtype=get_autocast_dtype("bf16-mixed"),
        autocast_dtype=get_autocast_dtype("bf16-mixed"),
        include_hiddens=True,
        include_embeddings=True,
        include_input_ids=True,
        skip_logits=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        initial_ckpt_path="/nfs_beijing/kubeflow-user/wanghao/workspace/ai4sci/PLMPretrain/bionemo-framework/results0608/evoplm-12+6-crosslabel-top100-prefix32-bak/checkpoints/epoch=0-step=53999-consumed_samples=27648000.0",
)
tokenizer = get_tokenizer()
module = biobert_lightning_module(config=config, tokenizer=tokenizer)
ESM2Config.initial_ckpt_skip_keys_with_these_prefixes = []

trainer = nl.Trainer(
    accelerator="gpu",
    devices=1,
    strategy=strategy,
    num_nodes=1,
    # callbacks=callbacks,
    plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    max_steps=100,
)
from lightning.pytorch.trainer import call, setup
breakpoint()
