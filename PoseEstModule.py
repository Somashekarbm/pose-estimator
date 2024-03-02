import cv2 as cv
import mediapipe as mp
import time

class PoseEstimator:
    def __init__(self, input_file_path, output_file=r'C:\Users\Somashekar\OneDrive\Desktop\ML-PRACTISE\computer vision\POSE_ESTIMATOR\output_video.mp4'):
        self.input_file_path = input_file_path
        self.output_file = output_file
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose()
        self.cap = cv.VideoCapture(input_file_path)
        self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.out = cv.VideoWriter(output_file, self.fourcc, 20.0, (640, 480))

    def process_video(self):
        ptime = 0
        while True:
            success, img = self.cap.read()
            if not success:
                break

            resized_frame = cv.resize(img, (640, 480))
            imgRGB = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
            results = self.pose.process(imgRGB)

            if results.pose_landmarks:
                self.mpDraw.draw_landmarks(resized_frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = resized_frame.shape  # Use resized_frame.shape instead of img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv.circle(resized_frame, (cx, cy), 5, (0, 255, 255), cv.FILLED)
            ctime = time.time()
            if ptime != 0:
                fps = 1 / (ctime - ptime)
                cv.putText(resized_frame, f'FPS: {int(fps)}', (10, 45), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            ptime = ctime
            self.out.write(resized_frame)

            cv.imshow('Resized Frame', resized_frame)

            key = cv.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv.destroyAllWindows()
