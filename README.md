# The Eagle Eyes Hackathon

Welcome to the Eagle Eyes Hackathon, where you'll be building 
a computer vision system to help Search and Rescue teams locate
missing persons by drone.

## Example Image

Here is a typical case - there is a person in this image in plain view - see if you can 
find them.  Scroll to the bottom of this Readme to find the location.

![Example Image](https://www.eagleeyessearch.com/images/generic/scree_field_raw.jpg)


## Format

When the clock stops, you'll submit a Google Colab notebook.  Your notebook will contain 






## Getting set up locally 

You'll probably want to get set up locally on your computer.  Here's how to do that.

1) Clone this repo to your computer.  You can do this by running the following command in your terminal:

```
git clone git@github.com:petered/eagle_eyes_hackathon.git
```

2) **Create a virtual environment.**  You can do this by running the following command in your terminal:
   1) With venv:
   ```
   python3 -m venv eagle_eyes_hackathon
    ```
   2) With Conda:
   ```
    conda create -n eagle_eyes_hackathon python=3.10
    ```


3) **Install the requirements.**   You can do this by running the following command in your terminal:
```
pip install -r requirements.txt
```


4) Download the training dataset.
   1. Manually you can download the dataset from (TODO: Add link), 
      and unzip it to `~/Downloads/eagle_eyes_dataset`


5) Get into it!
   1) Check out the submission templates in `hackathon/submissions`.  
   There you'll find a sample submission and 2 templates to get you started.
   2) Check out some scripts showing you how to compare and debug detectors  `hackathon/scripts`



6) Submission  
In the end you'll submit a colab notebook containing your submission.
We'll evaluate by using `YourModelLoader` to load your model and run it against the test dataset.
Put all your code in one big `submission.py` file, and copy the contents of that file into a colab notebook.
Make sure the notebook runs and that your model succesfully loads.

[Colab Notebook Submission Template](https://colab.research.google.com/drive/1FZOKhAr1mHrJ8wwh3E0OXMnGMOmubvvC?usp=sharing)

We will evaluate your code by copying your notebook and adding a cell like this:

```commandline
    model = MyModelLoader().load_detector()
    evaluate_models_on_dataset(
        detectors={model.get_name(): model},
        data_loader=AnnotatedImageDataLoader.from_folder(SECRET_TEST_SET_FOLDER)
    )
```






## The Answer

Here is the person

![Example Image](https://www.eagleeyessearch.com/images/generic/scree_field_raw_detected.jpg)

