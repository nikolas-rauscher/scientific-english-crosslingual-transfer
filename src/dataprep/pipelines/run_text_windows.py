#!/usr/bin/env python3
"""
Create text-only sliding windows for efficient T5 training.
"""

import logging
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataprep.sliding_window_processor import TextOnlySlidingWindowProcessor

def main():
    # Configuration
    input_folder = "data/cleaned_pretraining"  # Source data
    output_folder = "data/text_windows_test"   # Test output
    
    # Test with limited data
    limit_documents = 1000  # Process only 1000 docs for testing
    tasks = 4              # Fewer tasks for testing
    workers = 8            # Fewer workers
    
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Pipeline configuration
    pipeline_steps = [
        ParquetReader(
            data_folder=input_folder,
            glob_pattern="*.parquet",
            limit=limit_documents,
        ),
        TextOnlySlidingWindowProcessor(
            tokenizer_name_or_path="t5-base",
            target_tokens=512,
            overlap_ratio=0.5,
            log_to_wandb=False,  # Disable for test
        ),
        ParquetWriter(
            output_folder=output_folder,
            compression="snappy",
        ),
    ]
    
    # Execute pipeline
    executor = LocalPipelineExecutor(
        pipeline=pipeline_steps,
        tasks=tasks,
        workers=workers,
        start_method="spawn",
    )
    
    print(f"🚀 Creating text-only sliding windows...")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Limit: {limit_documents} documents")
    print(f"Tasks: {tasks}, Workers: {workers}")
    
    executor.run()
    
    print(f"✅ Text-only sliding windows created in: {output_folder}")
    print(f"💡 To test performance, update data_dir in config to: {output_folder}")

if __name__ == "__main__":
    main()