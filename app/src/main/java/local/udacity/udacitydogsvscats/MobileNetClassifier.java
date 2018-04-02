/*
 *    Copyright (C) 2017 MINDORKS NEXTGEN PRIVATE LIMITED
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package local.udacity.udacitydogsvscats;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.support.v4.os.TraceCompat;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

/**
 * A MobileNet using TensorFlow.
 */
public class MobileNetClassifier {

    private static final String TAG = "MobileNetClassifier";

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;

    // Pre-allocated buffers.
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private TensorFlowInferenceInterface inferenceInterface;

    private boolean runStats = false;

    private MobileNetClassifier() {
    }

    /**
     * Preprocess the image data from 0-255 int to normalized float based
     * @param color
     * @return
     */
      public float preprocessInput(int color){
          return (float) (color / 127.5 - 1.0);
      }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param inputSize     The input size. A square image of inputSize x inputSize is assumed.
     * @param inputName     The label of the image input node.
     * @param outputName    The label of the output node.
     * @throws IOException
     */
    public static MobileNetClassifier create(
            AssetManager assetManager,
            String modelFilename,
            int inputSize,
            String inputName,
            String outputName)
            throws IOException {

        MobileNetClassifier c = new MobileNetClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        c.inputSize = inputSize;

        // Pre-allocate buffers.
        c.outputNames = new String[]{outputName};
        c.intValues = new int[inputSize * inputSize];
        c.floatValues = new float[inputSize * inputSize * 3];
        c.outputs = new float[1];
        return c;
    }

    public float recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        TraceCompat.beginSection("recognizeImage");

        TraceCompat.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = preprocessInput((val >> 16) & 0xFF);
            floatValues[i * 3 + 1] = preprocessInput((val >> 8) & 0xFF);
            floatValues[i * 3 + 2] = preprocessInput(val & 0xFF);
        }
        TraceCompat.endSection();

        // Copy the input data into TensorFlow.
        TraceCompat.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, new long[]{1, inputSize, inputSize, 3});
        TraceCompat.endSection();

        // Run the inference call.
        TraceCompat.beginSection("run");
        inferenceInterface.run(outputNames, runStats);
        TraceCompat.endSection();

        // Copy the output Tensor back into the output array.
        TraceCompat.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        TraceCompat.endSection();

        TraceCompat.endSection(); // "recognizeImage"
        return outputs[0];
    }

    public void enableStatLogging(boolean debug) {
        runStats = debug;
    }

    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    public void close() {
        inferenceInterface.close();
    }
}
