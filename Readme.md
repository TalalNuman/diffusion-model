========================================================================
Image Generation Using Diffusion Models (Assignment 5 - Spring 2025)
========================================================================

Student Name: Talal
Roll Number: MSDS25011
University: ITU Lahore
Department: MSDS (2nd Semester)

------------------------------------------------------------------------
Project Structure
------------------------------------------------------------------------
Talal_MSDS25011_05/
│
├── MSDS25011_05.py             # Main training & sampling CLI code
├── MSDS25011_05_allCode.py     # Copy of all code files for submission
├── test_single_sample.ipynb    # Evaluation notebook (loaded for grading)
├── Readme.txt                  # This file
├── Report.pdf                  # Assignment Report
└── saved_models/               # Folder where trained model weights are saved
    ├── diffusion_model.pth     # Saved PyTorch model checkpoint
    └── loss_curve.png          # Plot of training loss

Note: The dataset folder `animal_data` should be placed in the parent directory of this folder.

------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------
To run the code, ensure the following Python packages are installed:
- python >= 3.8
- torch >= 2.0
- torchvision
- numpy
- matplotlib
- pillow

You can install them via pip:
$ pip install torch torchvision numpy matplotlib pillow

------------------------------------------------------------------------
How to Train the Model
------------------------------------------------------------------------
You can run the training script by specifying the path to the dataset folder. By default, it looks for `animal_data` in the parent directory (`../animal_data`).

To run with default settings:
$ python MSDS25011_05.py --data_path ../animal_data --epochs 300 --batch_size 10 --image_size 64

Command line options:
  --data_path    Path to the dataset directory (default: 'animal_data')
  --epochs       Number of epochs to train (default: 200)
  --batch_size   Batch size (default: 10, fits the 100 images evenly)
  --lr           Learning rate (default: 1e-3)
  --image_size   Image height/width to resize to (default: 64)
  --loss_type    Custom loss type: 'l2' (MSE) or 'l1' (MAE) (default: 'l2')
  --output_dir   Directory to save weights/plots (default: 'saved_models')

The script automatically detects and utilizes GPU acceleration on:
- Apple Silicon Macs (MPS)
- CUDA-enabled machines
Otherwise, it defaults to CPU training.

------------------------------------------------------------------------
How to Evaluate and Sample (test_single_sample.ipynb)
------------------------------------------------------------------------
During evaluation:
1. Open the Jupyter Notebook:
   $ jupyter notebook test_single_sample.ipynb
2. Run all cells sequentially.
3. The notebook will:
   - Load a sample image from the dataset and plot the forward noising process across 1000 timesteps.
   - Load the saved model weights (`saved_models/diffusion_model.pth`).
   - Run the reverse diffusion sampling loop (from noise to image) and plot the generated results in a grid.
========================================================================
