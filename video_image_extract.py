#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:02:24 2018

@author: fredoleary
"""
import time
import cv2
import os

def videos_to_frames( video_dir, dest_folder):
    """
    Process mutiple video files
    video_dir: folder containing mp4 files
    dest_dir: destination folder
    example file1.mp4 will be extracted to dest_dir\file1\file1_0001.bmp,
    dest_dir\file1\file1_0002.bmp, dest_dir\file1\file1_0003.bmp...
    """
    for file in os.listdir(video_dir):
        if file.endswith(".mp4"):
            print("File:", file)
            dest_dir = file[:-len(".mp4")]
            try:
                video_to_frames(video_dir + "/" + file, dest_folder + "/" + dest_dir, dest_dir)
            except OSError as err:
                print("Failed to convert", file, err)


    
def video_to_frames(input_loc, output_loc, file_prefix):
    """Extract frames from video file and save them as frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video...")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/" + file_prefix + "_%#05d.bmp" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("%d frames extracted" % count)
            print ("Conversion time: %d seconds.\n" % (time_end-time_start))
            break
if __name__ == "__main__":
    videos_to_frames("src_videos", "dest_frames")
#    video_to_frames( "src_images/2018-01-31-16-17-50.mp4", "test_mp4")
    print("Done")
