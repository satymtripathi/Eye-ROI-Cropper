STANDARDIZED ROI CROP PIPELINE
================================

This directory contains the complete ecosystem for ROI cropping, including training scripts, trained models, and a unified inference tool.

CONTENTS
--------
1. MODELS (.pth)
   - model_standard.pth : High-Acc model for standard RGB eye images (768px).
   - model_slitlamp.pth : Specialized model for Slitlamp images.

2. TRAINING SCRIPTS (.py)
   - train_standard.py  : Script used to train the standard model.
   - train_slitlamp.py  : Script used to train the slitlamp model.
   
3. INFERENCE (.py)
   - run_pipeline.py    : The main tool to run. Handles both standard and slitlamp modes.

4. DEPENDENCIES
   - requirements.txt   : Run `pip install -r requirements.txt`

USAGE (INFERENCE)
-----------------
Run the `run_pipeline.py` script from the command line.

You MUST specify the type of image you are processing: standard OR slitlamp.

Command:
   python run_pipeline.py --input "path/to/in" --output "path/to/out" --type [standard|slitlamp]

Examples:
   # Process Standard Images
   python run_pipeline.py --input ./raw_data --output ./results_std --type standard

   # Process Slitlamp Images
   python run_pipeline.py --input ./slit_data --output ./results_slit --type slitlamp

Arguments:
   --type, -t       : (Required) 'standard' or 'slitlamp'. Picks the correct model automatically.
   --input, -i      : (Required) Input folder path.
   --output, -o     : (Required) Output folder path.
   --batch_size, -b : (Optional) Images per batch (default 4).
   --device, -d     : (Optional) 'cuda' or 'cpu' (default auto).

USAGE (TRAINING)
----------------
To retrain a model, open the corresponding script (e.g., train_standard.py), update the `DATA_FOLDER` path at the bottom, and run it.
   python train_standard.py
