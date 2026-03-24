"""Hydra structured config dataclasses for the steering pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    name: str = "llama_3.1_8b"
    hf_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hidden_dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    coefficients_soft: List[float] = field(default_factory=lambda: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    coefficients_hard: List[float] = field(default_factory=lambda: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8])


@dataclass
class SteeringConfig:
    method: str = "rfm"
    rfm_iters: int = 8
    n_components: int = 1
    bandwidths: List[float] = field(default_factory=lambda: [1, 10, 100])
    reg: float = 1e-3


@dataclass
class DataConfig:
    concept_class: str = "fears"
    concept_file: str = "concepts/fears.txt"
    lowercase: bool = True
    positive_template: str = "Personify someone who is terrified of {concept}.  What are your thoughts on the following statement? \\nStatement: {statement}"
    negative_template: str = "What are your thoughts on the following statement? \\nStatement: {statement}"
    eval_prompt_prefix: str = "phobia_eval"
    test_prompts_file: str = "test_prompts.yaml"


@dataclass
class PathsConfig:
    data_dir: str = "./data"
    cache_dir: Optional[str] = None
    output_dir: str = "outputs/${model.name}/${data.concept_class}/${steering.method}"
    checkpoint_dir: str = "checkpoints"


@dataclass
class TrainingConfig:
    batch_size: int = 16
    label_type: str = "soft"
    rep_token: str = "max_attn_per_layer"
    head_agg: str = "mean"
    datasize: str = "single"


@dataclass
class GenerationConfig:
    max_tokens: int = 50
    versions: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    do_sample: bool = False


@dataclass
class EvaluationConfig:
    openai_model: str = "gpt-4o-2024-11-20"
    rate_limit_delay: float = 1.0
    max_retries: int = 10
    temperature: float = 0.0
    max_tokens: int = 20


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "llm-steering"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class SmokeTestConfig:
    enabled: bool = False
    n_concepts: int = 3


@dataclass
class TimingConfig:
    enabled: bool = False
    log_per_concept: bool = True


@dataclass
class PipelineConfig:
    seed: int = 0
    stages: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    model: ModelConfig = field(default_factory=ModelConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    smoke_test: SmokeTestConfig = field(default_factory=SmokeTestConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
