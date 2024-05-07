# CS543-Final-Project

## Setup
1. Clone repository
2. Run `cd utils && make`. 
3. Download desired VGG16 weights. 
4. Download training images from [KITTI's data_road](https://www.cvlibs.net/datasets/kitti/eval_road.php) training set
5. Run `python3 run_encoder.py -w=<VGG16_weights_file> -r=<training_images> -o=<output_file> -m=<"VGG-pool5"|"VGG-fc7"> -u=SEG` to write the encoder data to `<output_file>`. 
6. Run `python3 train_decoder.py -e=<encodings_file> -o=<model_dest_file> -u=SEG -m=<"VGG-pool5"|"VGG-fc7">  -l=<labels_dir> -d=<hist_file>` to train a segmentation decoder using `-m` encodings from `encodings_file` file using label images in `labels_dir`. This process writes the model to `model_dest_file` which must end with `.keras` and writes the history of the model to `hist_file`. 
7. Run `python3 run_model.py -m=<model_file> -e=<encodings_file> -o=<output_file> -u=SEG -t=<"VGG-pool5"|"VGG-fc7">` to run the encodings at `encodings_file` through the model saved at `model_file` using encoding-mode `-t`. This outputs the decoded data to `output_file`.
8. Run `python3 visualize.py -i=<input_file> -o=<output_folder>` to create visual representations of the first 10 decoded images in `input_file` and a histogram of the runtimes for decoding all the images. 
9. Run `plot_encoder_runtimes.py -i=<encodings_file> -o=<output_file>` to plot a histogram of the runtimes for encoding the data in `encodings_file` and write it to `output_file.
10. Run `plot_accuracy_plots.py -i=<hist_file> -o=<output_file>` to plot loss-over-time for the training of the decoder that resulted in `hist_file`. This plot is written to `output_file`. 