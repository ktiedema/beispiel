package com.example.testappopencv.faceprocessing;

import org.opencv.core.Point;
import org.opencv.core.Rect;

import java.util.Map;

public class PredictedPerson {

    private int label;
    private double confidence;
    private String name;
    private Rect face;

    public PredictedPerson(int label, double confidence, String name, Rect face){
        this.label = label;
        this.confidence = confidence;
        this.name = name;
        this.face = face;
    }

    public int getLabel(){ return this.label; }

    public double getConfidence(){ return this.confidence; }

    public String getName() {
        return this.name;
    }

    public Rect getFace() { return this.face; }
}
