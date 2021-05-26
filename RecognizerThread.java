package com.example.testappopencv.faceprocessing;

import android.content.Context;
import android.util.Log;

import com.example.testappopencv.AppContext;
import com.example.testappopencv.database.DBOperations;
import com.example.testappopencv.utils.Utility;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;

public class RecognizerThread extends Thread {

    private boolean running;
    private PredictedPerson person;
    private FaceRecognition fr;
    //private Utility util;
    private DBOperations dbo;

    private ConcurrentLinkedQueue<RecognizerTask> tasks;

    private ArrayList<PredictedPerson> detectedPersons;
    private ArrayList<PredictedPerson> tempPersons;

    private HashMap<String, Integer> loginMap;
    private Thread thisThread;


    public RecognizerThread(){
        this.fr = new FaceRecognition();
        this.running = true;
        //this.util = util;
        this.dbo = DBOperations.getInstance();
        this.tasks = new ConcurrentLinkedQueue<>();
        this.tempPersons = new ArrayList<>();
        this.loginMap = new HashMap<>();
        this.thisThread = Thread.currentThread();
        //only train when there is a User registered, otherwise Exception will be thrown
        if( this.dbo.getNoOfUsers() > 0 && this.dbo.getNoOfImages() > 0 ) {
            Log.i("Recognition", "Get No of Users before training the Algo: "+ this.dbo.getNoOfUsers());
            this.fr.train();
        }
    }

    /**
     * Terminate Recognition Thread
     */
    public void terminate(){
        running = false;
    }

    /**
     * Inserts a new Task into the Queue
     */
    public void setRecognizerTask( RecognizerTask rta ){
        Log.i("Recognition", "insert in queue");
        if( tasks.size() == 0 ) {
            tasks.add( rta );
            //New Task available, wake up Thread
            synchronized ( this ) {
                notifyAll();
            }
            Log.i("Recognition", "Task inserted");
        }
    }

    /**
     * Wrapper Method to train the FaceRecognizer.
     * Only this Thread holds in instance of Class FaceRecognition.
     */
    public void train(){
        this.fr.train();
    }

    /**
     * Performs the identification of given faces
     */
    @Override
    public void run() {
        RecognizerTask rta;
        while( running ) {
            synchronized ( this ){
                while( tasks.size() == 0 ) {
                    try {
                        Log.i("Recognition", "Thread schläft");
                        wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }

            Log.i("Recognition", "Thread aufgewacht");
            rta = (RecognizerTask) tasks.poll();

            if( rta != null && rta.getFaces() != null ){
                Mat face;
                Mat resizedFace = new Mat();
                Mat resizedGrayFace = new Mat();
                for ( Rect originalFace : rta.getFaces().toArray() ) {
                    //snip out the face from original image and resize to standard Size(200x200)
                    Size newSize = new Size( 200, 200 );
                    face = rta.getImg().submat( originalFace );
                    Imgproc.resize( face, resizedFace, newSize );
                    Imgproc.cvtColor( resizedFace, resizedGrayFace, Imgproc.COLOR_RGB2GRAY );

                    //Call the FaceRecognizer and try to identify the person in front of Camera
                    person = fr.recognizeFace(resizedGrayFace, originalFace);
                    if( person != null ) {
                        tempPersons.add(person);
                        insertPersonForLogin(person);
                    }
                }
                this.detectedPersons = new ArrayList<>( tempPersons );
                tempPersons.clear();
            }
        }
    }

    /**
     * Returns the last Predicted Person.
     * @return
     */
    public PredictedPerson getLastPredictedPerson(){
        return this.person;
    }

    /**
     * If multiple Persons should be identified, a ArrayList with the Predicted Persons gets returned.
     * @return
     */
    public ArrayList<PredictedPerson> getPredictedPersons(){
        return this.detectedPersons != null ? this.detectedPersons : null;
    }

    public String getPersonForLogin(){
        String fittingPerson = null;
        int maxCount = 0;

        Iterator it = loginMap.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry)it.next();

            if( (int)pair.getValue() > maxCount ){

                maxCount = (int)pair.getValue();
                fittingPerson = (String)pair.getKey();
                Log.i("Recognition","New Login Candidate: " + fittingPerson + " = " + maxCount);
            }
            it.remove(); // avoids a ConcurrentModificationException
        }

        return fittingPerson;
    }

    private void insertPersonForLogin( PredictedPerson pp ){
        if( !loginMap.containsKey( pp.getName() ) ){
            loginMap.put( pp.getName(), 1 );
            Log.i("Recognition","New Login Candidate in Map: " + pp.getName() + " = " + 1);
        }else{
            loginMap.put(pp.getName(), loginMap.get(pp.getName()) + 1);
            Log.i("Recognition","Login Candidate updated Map : " + pp.getName() + " = " + loginMap.get(pp.getName()));
        }
    }

}
