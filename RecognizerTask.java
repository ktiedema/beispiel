package com.example.testappopencv.faceprocessing;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

public class RecognizerTask {

    private Mat img;
    private MatOfRect faceDetections;

    public RecognizerTask(Mat img, MatOfRect faces ){
        this.img = img;
        this.faceDetections = faces;
    }

    public Mat getImg(){
        return this.img;
    }

    public MatOfRect getFaces(){
        return this.faceDetections;
    }

}
