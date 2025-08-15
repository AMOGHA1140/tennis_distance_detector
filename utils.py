from typing import Union, Tuple
from TennisCourtDetector import infer_in_image, tracknet
import torch, cv2
import numpy as np
from ultralytics.utils.ops import xyxy2xywh, xywh2xyxy

model = tracknet.BallTrackerNet(out_channels=15).to('cuda')
model.load_state_dict(torch.load('model_tennis_court_det.pt', map_location='cuda'))
model.eval()


def is_gameplay_image(image:np.ndarray, yolo_model=None, boxes=None, min_people:int=4, max_non_people:int=3):
    """
    image:
    yolo_model: the YOLO classification model used
    boxes: If the image is already passed through the model, the boxes can be returned as well. (output of `model(img)[0].boxes`)
    min_people: min number of people needed to classify it as gameplay image
    max_non_people: max number of non-people obj allowed for classifying it as gameplay img, (eg. racket, ball, chair in tennis)

    The model works by checking the number of people persent in the image. If its below the threshold, its classified as not a gameplay image. 
    This is faster, although won't work for all the possible types of images. It is selected after seeing the video, non-gameplay videos don't have many people
    """

    if boxes is None and yolo_model is None:
        raise ValueError('either boxes or the yolo-model MUST be provided')
    
    if boxes is None:
        boxes = yolo_model(image)[0].boxes.cpu() #not optimized for batched-inputs

    person_count = (boxes.cls==0).sum()
    non_person_count = (boxes.cls!=0).sum()


    if (person_count>=min_people and non_person_count<=max_non_people):
        return True
    else:
        return False




        


# Note: I ended up needing to use this anyways, because I couldn't come up with a better and faster way. (Bad naming, because its not depreciated)
def __legacy_is_gameplay_image(image:np.ndarray=None, points:np.ndarray=None, threshold:int=10):
    """
    Returns weather the given image (technically frame of a video), contains gameplay which has a court in it. 
    This is a simple techinque to distinguish b/w proper gameplay v/s other camera angles.    

    The model which detects the intersections of court lines is used here. When it is separate angle, or other images, 
    it instead returns a list consisting of mostly (None, None). So this technique is used to find distinguish the images


    threshold: minimum number of (None, None) required to classify it as not a  Gameplay footage. Default=10

    Returns: 
    True if gameplay image
    False otherwise
    """

    if image is None and points is None:
        raise ValueError('both image and points, not given. Atleast one must be given')
    
    if image is not None:
        points = infer_in_image.get_points(image, model)
    else:
        points = np.ndarray(points)
    
    none_count = points.count((None, None))

    return not (none_count >= threshold)


def get_court_points(image):
    return infer_in_image.get_points(image, model)


def get_court_corners(image: np.ndarray=None, points:np.ndarray = None):
    """
    Returns points as a list in following order:
    top_left, top_right, bottom_right, bottom_left
    """
    if image is None and points is None:
        raise ValueError('neither image nor points provided')
    

    if points is None:
        points = infer_in_image.get_points(image, model)

    #The order returned is [top_left, top_right, bottom_left, bottom_right ...], so we need to 3rd and 4th elements
    #refer https://github.com/yastrebksv/TennisCourtDetector/blob/main/imgs/net_prediction.png

    points[2], points[3] = points[3], points[2]

    return points[:4]


class Transform:

    def __init__(
            self, 
            source_points:np.ndarray, 
            destination_points: np.ndarray = np.float32([[0, 0],[36, 0],[36, 78],[0, 78]])
        ):
        """
        source_points: Shape (4, 2). Original plane from where the points are obtained.

        destination_points: The final plane to which the points are to be mapped. The default values are for official Lawn Tennis courts
        """
        destination_points = np.array(destination_points, dtype=np.float32)
        source_points = np.array(source_points, dtype=np.float32)

        homography_matrix, _ = cv2.findHomography(source_points, destination_points)

        self.homography_matrix = homography_matrix

    def source_to_destination(self, source:np.ndarray):
        source = np.array(source, np.float32)
        return cv2.perspectiveTransform(source, self.homography_matrix)
    
    def destination_to_source(self, destination):
    
        destination = np.array(destination)    
        return cv2.perspectiveTransform(destination, np.linalg.inv(self.homography_matrix))
    






class PlayerTracker:

    def __init__(
            self, 
            closest_line: np.ndarray,
            transformer: Transform,
            alpha:int=0.5,
            beta:int=0.4,
            bottom_middle:bool = False
        ):
        """
        
        closest_line: The edge nearer to the player, represented by its two endpoints. This is used to filter out the boxes.
        shape (2, 2)

        alpha: The smoothing factor for measuring position. alpha=1 would mean you completely trust new measurements, alpha=0 would mean you wouldn't trust them at all

        beta: smoothing factor for measuring velocity. see alpha

        bottom_middle: If true, then the midpoint of bottom edge is used as reference point while calculating distance fromt he lines. Otherwise, center of box
        (I've set True for top player, and False for bottom player)

        """
        
        self.position = None
        self.velocity = np.array([0., 0.])
        self.alpha = alpha
        self.beta = beta
        self.bottom_middle = bottom_middle

        self.distance_travelled = []
        self.closest_line = np.array(closest_line)
        self.coordinate_transformer = transformer
        


    def update(self, points:np.ndarray, is_gameplay:bool=True):
        """
        Accepts a list of boxes in xywh format, all of which have to correspond to PERSON class only. 
        Then it computes the box closest to cls.closest_point and treats that box at the current position of the person

        Arguments:
        points: shape (N, 4), N points in xywh format

        Returns:
        the index of the point chosen as the player's box
        """
        
        points = np.array(points)

        if not is_gameplay:
            self.position = None #
            self.velocity = np.array([0., 0.])
            return

        
        distance = self.calculate_distance(points[:, :2])
        if self.bottom_middle:
            distance = self.calculate_distance(points[:, :2] + points[:, 2:]*np.array([0,0.5]))



        closest_box_index = np.argmin(distance)
        closest_box = points[closest_box_index]


        


        #center of bottom edge is taken, as it is closer to ground and a more accuracy representation of distance moved, after homography transform
        x_center = closest_box[0]
        y_bottom = closest_box[1] + closest_box[3]/2

        measured_position = self.coordinate_transformer.source_to_destination([[[x_center, y_bottom]]]).flatten()

        #if starting freshly, then reassign the values
        if self.position is None:   #
            self.distance_travelled.append(0.)
            self.position = measured_position #
            

            return closest_box_index


        predicted_pos = self.position + self.velocity
        residual = measured_position - predicted_pos
        self.position = predicted_pos + self.alpha*residual
        self.velocity = self.velocity + self.beta*residual

        distance_in_current_frame = np.linalg.norm(self.velocity)
        self.distance_travelled[-1] += distance_in_current_frame

        # self.distance_travelled[-1] += np.linalg.norm(transformed_coordinates - self.prev_pos) #
        # self.prev_pos = transformed_coordinates #

        return closest_box_index
    
    def update_closest_line(self, new_closest_line:np.ndarray):
        self.closest_line = np.array(new_closest_line)

    def update_transformer(self, new_transformer:Transform):
        self.coordinate_transformer = new_transformer

    def calculate_distance(self, points):
        """
        points: shape (N, 2)
        
        calculate the distance b/w given point and the line SEGMENT self.closest_line

        (Very Imp) Note: this makes use of the fact that the line is almost horizontal
        """
        points = np.array(points)
        self.closest_line

        B = self.closest_line[1] - self.closest_line[0] #vector in direction of line
        V0 = points[:, :2]-self.closest_line[0] #vector joining point 0 to coordinate of box
        V1 =  points[:, :2]-self.closest_line[1]
        
        #perp dist from point to lines
        distance = np.sqrt(np.linalg.norm(V0, axis=1)**2 - (np.dot(V0, B)/np.linalg.norm(B))**2)

        #if both dots are positive, point lies b/w two lines perpendicular to current one, passing through the two endpoints of segment
        # otherwise, it will on either side, determined by this dot product
        mask = (np.dot(V0, B) > 0) & (np.dot(V1, -B) < 0)
        distance[mask] = np.linalg.norm(V1, axis=1)[mask] #closer to point-1, outside the segment
        mask = (np.dot(V0, B) < 0) & (np.dot(V1, -B) > 0)
        distance[mask] = np.linalg.norm(V0, axis=1)[mask] #closer to point-0, outside the segment

        return distance











