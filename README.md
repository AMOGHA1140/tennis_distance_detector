Solution Video - [download](https://drive.google.com/file/d/14XFOsiOqbeE0Hh9YAyN4pabJKQySS_Ng/view?usp=sharing)


### For reproducing the video, follow the steps below

1) Clone this repository. 
2) Install the required packages

    ```pip install -r requirements.txt```

3) Download the original video, and assign its path to 
`ORIGINAL_VIDEO_PATH` in `main.py`. (optionally edit any other settings you want to)
4) run main.py


## Folder Structure
<!-- 
`images/` - Originally, I separated a few frames videos so as to experiment on individual images before doing on videos. So those images were stored here. The script to generate these images from the given video is present in `main.ipynb` -->

`TennisCourtDetector/` - repository cloned from this [repo](https://github.com/yastrebksv/TennisCourtDetector). I deleted all the files which are not present in the directory now. The only change is in `./infer_in_image.py`, where I added a function `get_points` to get all the court points for a given image.

`main.py` - The main code for generating output video  

`utils.py` - defining the necessary Transform and PlayerTracker classes along with more functions


### The flow from input to output video

* Firstly, the video is processed frame by frame, which itself is inefficient. and also processing together could've made stuff like classifying a screen as gameplay/not gameplay easier. 

* Now, we process the video till we find a valid game-play frame, so that we can detect the court co-ordinates first. 
(This has issues if a lot of starting part is not game-play)

* A counter is maintained to know when we need to re-calculate court coordinates, and if need to process the image for obj detection. 

#### For every frame, do the following: -
1) If no need to re-calculate court, or do object detection, then continue to next step
2) if need to calculate court-coordinates, then do so, and update them for both players
3) If no need to do obj detection then continue to next step

4) If the image is not gameplay-image, then write it to video file, and go to next step

5) Get the objects detected by the model, and select the ones with confidence >0.3* and person class. Update the coordinates of player1 and player2**
6) Put the bounding boxes on the image, along with player names and distances. Write the image


\*Confidence of 0.3 was chosen after experiementing with values from 0.1 to 0.5 in increments of 0.1 $\newline$
\*\*the logic of updating is explained in explanation of PlayerTracker

#### Other Notes:

1) __`Transform` Class__

It took inputs of the court coordinates, and mapped them to the real coordinates, and performed homography transformations. 

2) __`PlayerTracker`__

It saves the distance travelled, current location and other stuff for individual players.

I would like to address how I selected the correct box here:- $\newline$
Firstly, most of the time, players are near their respective sides, and not much near courts. So I calculated distances of the boxes from the line (specifically, line segment, not line), and took the one with least distance. This is not without errors, but I couldn't find better methods. <br>
I tried distance from just the line - it captured audience sometimes. <br>

I couldn't make use of information from previous runs, which could've increased detection accuracy by a large margin. 

For calculating the distance moved, Kalman Filtering is used to get smoother measurements, which was especially an issue for the top player. (I'll admit, I myself don't fully understand this concept, but it did give smoother measurements, so I stuck with it over using just normal differences in movement).  Other options could've been Simple or Exponential moving average. 
