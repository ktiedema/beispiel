package com.example.testappopencv.faceprocessing;

import android.content.Context;
import android.content.ContextWrapper;
import android.util.Log;

import com.example.testappopencv.database.DBOperations;
import com.example.testappopencv.database.tables.Images;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opencv.imgcodecs.Imgcodecs.imread;

public class FaceRecognition {

    private FaceRecognizer fr;
    private DBOperations dbo;

    private File[] imageFiles;
    private Map<Integer, String> names;

    public FaceRecognition(){
        this.fr = LBPHFaceRecognizer.create(1,4,8,8,12.0);
        this.names = new HashMap<>();
        this.dbo = DBOperations.getInstance();
    }

    /**
     * gets called in the constructor of the Recognizer Thread. The Training of the Algorithm gets done
     * once at the beginning of the Program.
     */
    public void train(){
        //Get Access to Directory with the cropped Faces
        //ContextWrapper cw = new ContextWrapper(context);
        //File croppedDirectory = cw.getDir("croppedFiles", Context.MODE_PRIVATE);
        //Store all Pictures in an File array
        //imageFiles = croppedDirectory.listFiles();

        List<Images.ImageData> imageData = dbo.getAllImages();

        //train Method needs a List as Input
        List<Mat> imageList = new ArrayList<>(imageData.size());
        int[] lab = new int[imageData.size()];
        int counter = 0;
        //loop over all found Pictures
        for (Images.ImageData image : imageData) {
            //Get the image as Mat and store it in List
            Mat img = Imgcodecs.imdecode(new MatOfByte(image.getImage()), Imgcodecs.IMREAD_GRAYSCALE);
            //Mat img = imread(image.getAbsolutePath(),Imgcodecs.IMREAD_GRAYSCALE);
            //Log.i("Recognition" , "process image: " + image.getName());
            imageList.add(img);

            //Get the Label of the Image, Image Names are stored like <Label>_<Name>_<Indx>.jpg
            int label = Integer.parseInt(image.getImageName().split("\\_")[0]);
            //Log.i("Recognition" , "Label: " + label);
            lab[counter] = label;

            counter++;
        }

        MatOfInt labels = new MatOfInt();
        labels.fromArray(lab);

        fr.train(imageList, labels);
        Log.i("Recognition", "Training finished");
    }

    /**
     * Tries to recognize a known Person in given Image.
     * @param faceToTest image in which the person should get recognized
     * @return PredictedPerson, includes labels, confidence and names
     */
    public PredictedPerson recognizeFace(Mat faceToTest, Rect face) {
        int noOfUser = dbo.getNoOfUsers();
        int[] lab = new int[noOfUser];
        double[] conf = new double[noOfUser];

        //performing the prediction
        fr.predict(faceToTest, lab, conf);

        //rounds value to two decimal points
        BigDecimal bd = new BigDecimal( conf[0] ).setScale(1, RoundingMode.HALF_UP);
        double con = bd.doubleValue();



        //label has value -1 if it was not possible to predict a Person
        String name = ( lab[0] == -1 ) ? "Unknwon Person" : dbo.getNameOfUser( lab[0] );

        return new PredictedPerson(lab[0], con, name, face);
    }
}
