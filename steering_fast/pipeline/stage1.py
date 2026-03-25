"""Stage 1: Train steering directions per concept.

Uses the ORIGINAL compute_save_directions() which internally calls
NeuralController -> control_toolkits -> direction_utils -> rfm.py.
This guarantees identical direction vectors including normalization,
RFM training, and sign correction. Adds checkpointing and timing.
"""
import logging
import os
import sys

import torch

from ..tracking.checkpoint import CheckpointManager
from ..tracking.timer import PipelineTimer
from ..tracking.wandb_tracker import WandbTracker
from ..utils import config_hash, ensure_dir, get_concept_slice, read_concept_list, set_seed

log = logging.getLogger(__name__)


def _setup_core_imports():
    """Add core/ to sys.path so original modules can import each other."""
    core_dir = os.path.join(os.path.dirname(__file__), "..", "core")
    core_dir = os.path.abspath(core_dir)
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)


def run_stage1(cfg, timer: PipelineTimer, tracker: WandbTracker) -> None:
    """Train steering vectors using original code with checkpointing."""
    _setup_core_imports()
    set_seed(cfg.seed)

    data_dir = os.path.abspath(cfg.paths.data_dir)
    use_soft_labels = cfg.training.label_type == "soft"

    # cd to data parent so original relative paths resolve
    data_parent = os.path.dirname(data_dir)
    os.chdir(data_parent)

    # Import original modules (from core/)
    from utils import select_llm, compute_save_directions, get_tokenidx_per_layer_per_concept
    from datasets import get_dataset_fn
    import utils as orig_utils

    # Patch DATA_DIR
    orig_utils.DATA_DIR = data_dir

    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    all_concepts = read_concept_list(concept_file, lowercase=cfg.data.lowercase)
    concepts = get_concept_slice(all_concepts, cfg)
    log.info("Stage 1: %d/%d concepts, method=%s, labels=%s",
             len(concepts), len(all_concepts), cfg.steering.method, cfg.training.label_type)

    # Checkpoint
    ckpt = CheckpointManager(cfg.paths.checkpoint_dir, "stage1", config_hash(cfg))
    completed, _ = ckpt.load()

    # Load model ONCE
    llm = select_llm(cfg.model.name)

    # O1: Monkey-patch direction_utils to use batched extraction
    # O2: Cache negative hidden states across concepts
    batch_size = cfg.training.batch_size if hasattr(cfg, 'training') else 8
    _neg_cache = {}  # O2: {layer_idx: tensor} cached across concepts

    try:
        import direction_utils
        orig_fn = direction_utils.get_hidden_states_and_attns

        def _batched_wrapper_with_cache(prompts, labels, llm, model, tokenizer,
                                         hidden_layers, rep_token, layer_to_token, head_agg):
            """O1 + O2: Batched extraction with negative hidden state caching."""
            nonlocal _neg_cache

            # Split into positive and negative
            pos_indices = [i for i, l in enumerate(labels) if l == 1]
            neg_indices = [i for i, l in enumerate(labels) if l == 0]
            neg_prompts = [prompts[i] for i in neg_indices]

            # O2: Check if negative hidden states are cached
            if _neg_cache and len(_neg_cache) > 0:
                log.debug("O2: Using cached negative hidden states (%d samples)", len(neg_indices))
                # Only extract positive prompts
                pos_prompts = [prompts[i] for i in pos_indices]
                pos_labels = [1] * len(pos_prompts)
                try:
                    pos_hs, pos_soft = direction_utils.get_hidden_states_and_attns_batched(
                        pos_prompts, pos_labels, llm, model, tokenizer,
                        hidden_layers, rep_token, layer_to_token, head_agg,
                        batch_size=batch_size,
                    )
                except Exception:
                    pos_hs, pos_soft = orig_fn(
                        pos_prompts, pos_labels, llm, model, tokenizer,
                        hidden_layers, rep_token, layer_to_token, head_agg,
                    )

                # Reconstruct interleaved format
                final_hs = {}
                final_soft = {}
                for layer_idx in hidden_layers:
                    combined_hs = []
                    combined_soft = []
                    pi, ni = 0, 0
                    for i, l in enumerate(labels):
                        if l == 1:
                            combined_hs.append(pos_hs[layer_idx][pi])
                            combined_soft.append(pos_soft[layer_idx][pi])
                            pi += 1
                        else:
                            combined_hs.append(_neg_cache[layer_idx][ni])
                            combined_soft.append(torch.tensor([0.0]))
                            ni += 1
                    final_hs[layer_idx] = torch.stack(combined_hs, dim=0)
                    final_soft[layer_idx] = torch.stack(combined_soft, dim=0)
                return final_hs, final_soft

            # First concept: extract everything and cache negatives
            try:
                all_hs, all_soft = direction_utils.get_hidden_states_and_attns_batched(
                    prompts, labels, llm, model, tokenizer,
                    hidden_layers, rep_token, layer_to_token, head_agg,
                    batch_size=batch_size,
                )
            except Exception as e:
                log.warning("Batched extraction failed (%s), falling back", e)
                all_hs, all_soft = orig_fn(
                    prompts, labels, llm, model, tokenizer,
                    hidden_layers, rep_token, layer_to_token, head_agg,
                )

            # Cache negative hidden states for reuse
            for layer_idx in hidden_layers:
                _neg_cache[layer_idx] = all_hs[layer_idx][neg_indices].cpu()
            log.info("O2: Cached %d negative hidden states for reuse", len(neg_indices))

            return all_hs, all_soft

        direction_utils.get_hidden_states_and_attns = _batched_wrapper_with_cache
        log.info("O1+O2: Batched extraction with negative caching (batch_size=%d)", batch_size)
    except Exception as e:
        log.info("O1+O2: Not available: %s", e)

    dataset_fn = get_dataset_fn(cfg.data.concept_class, paired_samples=False)
    rep_token = cfg.training.rep_token
    head_agg = cfg.training.head_agg
    method = cfg.steering.method
    attn_dir = os.path.join(data_dir, "attention_to_prompt")

    for concept in concepts:
        # Check if direction already exists
        suffix = "_softlabels" if use_soft_labels else ""
        vec_path = os.path.join(
            data_dir, "directions",
            f"{method}_{concept}_tokenidx_{rep_token}_block{suffix}_{cfg.model.name}.pkl",
        )

        if concept in completed or os.path.exists(vec_path):
            log.info("Skipping %s (exists)", concept)
            completed.add(concept)
            continue

        with timer.time_concept("stage1", concept):
            layer_to_token = None
            if rep_token == "max_attn_per_layer":
                layer_to_token = get_tokenidx_per_layer_per_concept(
                    concept, cfg.model.name, head_agg=head_agg, root_dir=attn_dir
                )

            data = dataset_fn(llm, concept, datasize="single")

            # ORIGINAL function handles: normalization, RFM training,
            # sign correction, all 5 methods, everything
            compute_save_directions(
                llm, data, use_soft_labels, concept,
                rep_token=rep_token,
                hidden_state="block",
                layer_to_token=layer_to_token,
                concat_layers=[],
                control_method=method,
                head_agg=head_agg,
            )

            del data
            torch.cuda.empty_cache()
            log.info("Saved direction for %s", concept)

        completed.add(concept)
        ckpt.save(completed, {})
        tracker.log_concept(concept, "stage1", {"saved": True})

    ckpt.cleanup()
    log.info("Stage 1 complete: %d concepts", len(concepts))
