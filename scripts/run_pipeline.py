#!/usr/bin/env python
"""Run the steering pipeline. Hydra CLI wrapper.

Examples:
    python scripts/run_pipeline.py model=llama_3_1_8b data=fears
    python scripts/run_pipeline.py experiment=smoke_test
    python scripts/run_pipeline.py training.batch_size=32 wandb.enabled=true
"""
from steering_fast.pipeline.runner import main

if __name__ == "__main__":
    main()
