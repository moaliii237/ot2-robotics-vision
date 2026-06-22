# Computer Vision (ILO 6.4 + 8.5)

The CV half of the pipeline: find the petri dish, segment the plants, segment the roots
with a U-Net, then measure primary-root length for the Kaggle competition.

## Notebooks

| File | Task | ILO | What it does |
|---|---|---|---|
| `task_2.ipynb` | Task 2 | 6.4 B | Crops the petri dish out of a raw Hades image using only traditional CV (thresholding + contours). |
| `task_3.ipynb` | Task 3 | 6.4 C | Instance-segments the five plants with traditional CV. Runs on `task_3_image_1.jpg` and `task_3_image_2.jpg`. |
| `task_5_training.ipynb` | Task 5 | 8.5 A | Trains the U-Net root-segmentation model (256 px patches, BCE+Dice loss). |
| `task_5_inference.ipynb` | Task 5 | 8.5 A | Loads the trained model and predicts a mask for `task5_test_image`. |
| `task_8.ipynb` | Task 8 | 8.5 B | Full Kaggle pipeline: dish crop, U-Net, root-system skeletonisation, primary-root length per plant. |

## Other evidence in this folder

- `task5_test_image_prediction_MoaliJaberiiii_244437.png`: the predicted mask deliverable
  for Task 5 (8.5 A).
- `kaggle_leaderboard_screenshot.png`: my private-leaderboard position and score (8.5 B).
- `task_3_image_1.jpg`, `task_3_image_2.jpg`: the two Task 3 test images.
- `task_3_output_example_1.png`, `task_3_output_example_2.png`: example instance-segmentation
  outputs.

## The trained model (not in git)

The U-Net weights are ~370 MB, over GitHub's 100 MB file limit, so they are **not
committed here**. They sit in `../../_large_files_upload_to_drive/`, ready to go on Google
Drive:

- `MoaliJaberiiii_244437_unet_model_256px.h5`: the `.h5` deliverable.
- `jaberiiii_2444377_unet_model_256px.keras`: the same trained model in Keras format.

Both files are the same trained U-Net, just in two formats. Upload them to Drive, set
link-sharing to "anyone with the link", and paste the links into the learning log for
ILO 8.5 A and 8.5 B. Before submitting, open `task_5_inference.ipynb` and `task_8.ipynb`,
find the line that loads the model, and make sure the filename it expects matches the file
you uploaded.
