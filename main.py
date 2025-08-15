
import cv2
import utils
from ultralytics import YOLO


ORIGINAL_VIDEO_PATH = 'tennis_video_assignment.mp4'
DETECTION_MODEL_PATH = 'yolo11n.pt'
OUTPUT_VIDEO_PATH = 'output.mp4'
OUTPUT_FPS = 30
PREDICT_EVERY = 5 #instead of predicting on every frame, this will be used to predict every X frames, makes it faster
CALCULATE_COURT_POINTS_EVERY = 15 #same as above, but for court points


 
video = cv2.VideoCapture(ORIGINAL_VIDEO_PATH)

#setup config for output videos and the stream
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, OUTPUT_FPS, (frame_width, frame_height))




#initialize court co-ordinates
player1_line=None
player2_line=None
corner_points=None
yolo_model = YOLO(DETECTION_MODEL_PATH, verbose=False) #model is 

#calculate the court_points first, its needed for initializing player_trackers
while True:

    ret, frame = video.read()

    if not ret: #video ended
        break
    
    is_gameplay_image = (utils.is_gameplay_image(frame, yolo_model)  and utils.__legacy_is_gameplay_image(frame, threshold=1))

    if not is_gameplay_image: #not gameplay image
        continue

    corner_points = utils.get_court_corners(frame)
    player1_line = corner_points[:2]
    player2_line = corner_points[2:]
    
    break

video.set(cv2.CAP_PROP_POS_FRAMES, 0) #set the video back to first frame

#
transformer = utils.Transform(corner_points) 
player1 = utils.PlayerTracker(player1_line, transformer, alpha=0.15, beta=0.10, bottom_middle=True)
player2 = utils.PlayerTracker(player2_line, transformer, alpha=0.5, beta=0.4)



#obj detection model
yolo_model = YOLO(DETECTION_MODEL_PATH, verbose=False)

#main purpose of this is two things, to know when to re-calculate court points and perform prediction according to variables defined on the top
frame_count = -1


while True:

    frame_count += 1
    ret, frame = video.read()


    if not ret: #stream ended
        break    
    
    #if not to calculate court points or prediction, then skip
    if not (frame_count%PREDICT_EVERY==0) and not (frame_count%CALCULATE_COURT_POINTS_EVERY==0):
        pass

    boxes = yolo_model(frame, verbose=False)[0].boxes.cpu()
    is_gameplay_image = (utils.is_gameplay_image(frame, boxes=boxes)  and utils.__legacy_is_gameplay_image(frame, threshold=1))


    #don't update court coordinates if not a game-play image
    if frame_count%CALCULATE_COURT_POINTS_EVERY==0 and is_gameplay_image:

        
        print(f"updated court---------{frame_count}")
        corner_points = utils.get_court_corners(frame)
        player1_line = corner_points[:2]
        player2_line = corner_points[2:]
        transformer = utils.Transform(corner_points) 

        player1.update_closest_line(player1_line)    
        player1.update_transformer(transformer)
        player2.update_closest_line(player2_line)    
        player2.update_transformer(transformer)

    # skip if no need to predict anything for this frame
    if not frame_count%PREDICT_EVERY==0:
        continue
        

    print(f'adding frame-{frame_count}')
    
    #save frame and continue if not a game-play image. i.e. we won't need to process it.
    if not is_gameplay_image:
        output_video.write(frame)
        player1.update('', False)
        player2.update('', False)
        continue
    

    
    mask = ((boxes.conf > 0.3) & (boxes.cls==0)) #mask for removing low conf players, and non-person classes
    points = boxes.xywh[mask] #player classes expect input in xywh format
    player1_index = player1.update(points.cpu())
    player2_index = player2.update(points.cpu())


    #put box for player 1 (top player)
    x1, y1, x2, y2 = boxes.xyxy[mask][player1_index].to(int).tolist()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'Player 1 - {player1.distance_travelled[-1]:.2f} ft', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #put box for player 2 (bottom player)
    x1, y1, x2, y2 = boxes.xyxy[mask][player2_index].to(int).tolist()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, f'Player 2 - {player2.distance_travelled[-1]:.2f} ft', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    output_video.write(frame)

output_video.release()