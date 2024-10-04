import cv2
import numpy as np
import json


def compute_frame_diff(prev_frame, curr_frame):
    """Computes the absolute difference between two frames."""
    return cv2.absdiff(prev_frame, curr_frame)

def compute_frame_diff_grayscale(prev_frame, curr_frame, thresh_pixels):
    """
        Computes a diff of two frames, returns true if more that 'thresh_pixels' have
        changed.
    """
     # Convert frames to grayscale to simplify difference computation
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the current and previous frames
    diff = cv2.absdiff(gray_prev, gray_frame)

    # Threshold the difference to detect significant changes
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Count the number of non-zero pixels in the thresholded image
    non_zero_count = cv2.countNonZero(thresh)
    return non_zero_count > thresh_pixels, non_zero_count

def save_frame(frame, filename):
    """Saves a frame to a file."""
    cv2.imwrite(filename, frame)


class OpenCVFrameWriter:
    def __init__(self, video_path, json_index_outputpath, frame_outputpath, threshold_ratio = 0.1):
        self.video_path = video_path
        self.threshold_ratio = threshold_ratio
        self.frame_outputpath = frame_outputpath
        self.json_index_outputpath = json_index_outputpath
        self.video_index = []

    def process_video(self, debug_video_outfile=None):
        """Processes a video, computing frame differences and saving frames."""
        cap = cv2.VideoCapture(self.video_path)

        prev_frame = None
        frame_count = 0

        if debug_video_outfile:
            # Define a codec and create a VideoWriter object to save output frames as MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 output
            debug_out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))


        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Calculate total pixels in a frame
            frame_height = int(cap.get(4))
            frame_width = int(cap.get(3))
            total_pixels = frame_height * frame_width

            # Get the current frame's timestamp in milliseconds
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0

            should_save = True # so that we save the first frame
            percentage_change = 1.0
            if prev_frame is not None:
                thresh_pixels = int(total_pixels * self.threshold_ratio)
                should_save, changed_pixels = compute_frame_diff_grayscale(prev_frame, frame, thresh_pixels)
                percentage_change = float(changed_pixels / total_pixels)
                #diff = compute_frame_diff(prev_frame, frame)

                # Save the difference frame (optional)
                #save_frame(diff, f"diff_{frame_count}.jpg")
            
            if should_save: 
                # Save the current frame
                frame_path = f"{self.frame_outputpath}/frame_{frame_count}.jpg"
                save_frame(frame, frame_path)
                self.video_index.append({'timestamp': timestamp, 'frame_path': frame_path, 'percent_change': percentage_change})
                if debug_out:
                    debug_out.write(frame)  # Save the frame

                prev_frame = frame
            frame_count += 1

        cap.release()
        if debug_out:
            debug_out.release()

        # Save json index.
        with open(self.json_index_outputpath, 'w') as json_file:
            json.dump(self.video_index, json_file, indent=4)

        print(f"Video index saved as {self.json_index_outputpath}")


if __name__ == "__main__":
    video_path = "google_keynote_1.mp4"
    json_index_outputpath = "./temp/index/video_index.json"
    frame_outputpath = "./temp/frames/"
    threshold_ratio = 0.2
    debug_outfile = "./temp/debug_video.mp4"
    frame_writer = OpenCVFrameWriter(video_path, json_index_outputpath, frame_outputpath, threshold_ratio)
    frame_writer.process_video(debug_outfile)