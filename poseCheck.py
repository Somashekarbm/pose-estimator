from PoseEstModule import PoseEstimator

input_file_path = r"POSE_ESTIMATOR\video (2160p).mp4"

pose_estimator = PoseEstimator(input_file_path)
pose_estimator.process_video()
