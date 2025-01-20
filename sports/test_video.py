import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import EventDetector
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO


event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video that you want to test', default='test_videos/0.mp4')
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=30)
    parser.add_argument('-m', '--model-path', help='Path of model')
    args = parser.parse_args()
    seq_length = args.seq_length


    model = YOLO('yolov8n-pose.pt')

    print('Preparing video: {}'.format(args.path))



    cap = cv2.VideoCapture(args.path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_data = []

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run YOLOv8 inference
        results = model(frame, verbose=False)
        
        # Initialize row data with -1 values
        row_data = [-1] * 34  # 17 x-coords + 17 y-coords
        
        # If keypoints are detected, use the first person's keypoints
        if len(results[0].keypoints.data) > 0:
            # Get keypoints of first detected person
            keypoints = results[0].keypoints.data[0].cpu().numpy()

            x_coords = []
            y_coords = []
            
            for i, kp in enumerate(keypoints):
                x_coords.append(kp[0]/frame_width)
                y_coords.append(kp[1]/frame_height)

            # Add x-coords and y-coords to row_data
            row_data[:17] = x_coords
            row_data[17:34] = y_coords

        all_data.append(row_data)
        frame_count += 1


    images = torch.tensor(all_data, dtype=torch.float32)

    # Create the model
    hidden_dim = 256
    num_classes = 9
    model = EventDetector(hidden_dim=hidden_dim, num_classes=num_classes)

    save_dict = torch.load('models/30fps_seq30_over0.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    print('model loaded')

    print('Testing...')

    print(images.size())
    images = images.view(-1, 2, 17)
    images = images.unsqueeze(0)
    print(images.size())

    batch = 0
    probs_list = []
    while batch * seq_length < images.shape[1]:
        if (batch + 1) * seq_length > images.shape[1]:
            image_batch = images[:, batch * seq_length:, :, :]
            # Pad the last batch with last frame
            pad = image_batch[:, -1:, :, :].repeat(1, (batch + 1) * seq_length - images.shape[1], 1, 1)
            image_batch = torch.cat((image_batch, pad), dim=1)
        else:
            image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :]
        logits = model(image_batch.cuda())
        # logits shape: [1, seq_length, num_classes]

        # Apply softmax over num_classes dimension (dim=2)
        probs = F.softmax(logits.data, dim=2).cpu().numpy()  # Shape: [1, seq_length, num_classes]

        # Collect probabilities
        probs_list.append(probs)
        batch += 1

    probs = np.concatenate(probs_list, axis=1)
    probs = probs.squeeze(0)  # Shape: [total_length, num_classes]
    # breakpoint()
    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {}'.format(events))
    cap = cv2.VideoCapture(args.path)
    # Get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
    output_video_path = f'{args.path.split(".")[0]}_output.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



    current_frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        

        evt = 0
        if current_frame_index in events:
            # find location of crrent_fram_index in events
            evt = np.where(events == current_frame_index)[0][0]
            # Add text overlay
            text = event_names[evt]
            cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(frame)

        current_frame_index += 1
        if current_frame_index >= total_frames:
            break

