package ceneax.lib.tensorflow.runner;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import ceneax.lib.tensorflow.bean.Recognition;
import ceneax.lib.tensorflow.interfaces.IRunner;
import ceneax.lib.tensorflow.util.MatrixUtil;

import static java.lang.Math.min;

public class DetectorRunner implements IRunner<ArrayList<Recognition>> {

    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;

    private final Builder mBuilder;

    // 检测的图像
    private Bitmap mBitmap;
    // 坐标转换矩阵
    private Matrix mMatrix;
    // 角度
    private int mRotationDegrees;

    private final int[] mIntValues;
    private final ByteBuffer mImgData;

    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS, 4]
    // contains the location of detected boxes
    private float[][][] outputLocations;
    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    private float[][] outputClasses;
    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the scores of detected boxes
    private float[][] outputScores;
    // numDetections: array of shape [Batchsize]
    // contains the number of detected boxes
    private float[] numDetections;

    private DetectorRunner(Builder builder) {
        mBuilder = builder;

        mIntValues = new int[mBuilder.inputWidth * mBuilder.inputHeight];
        int numBytesPerChannel = mBuilder.modelQuantized ? 1 : 4;
        mImgData = ByteBuffer.allocateDirect(1 * mBuilder.inputWidth * mBuilder.inputHeight * 3 * numBytesPerChannel);
        mImgData.order(ByteOrder.nativeOrder());

        outputLocations = new float[1][mBuilder.numDetections][4];
        outputClasses = new float[1][mBuilder.numDetections];
        outputScores = new float[1][mBuilder.numDetections];
        numDetections = new float[1];
    }

    @Override
    public ArrayList<Recognition> run(Interpreter interpreter) {
        if (mBitmap == null || mMatrix == null) {
            return null;
        }

        Matrix frameToCropTransform = MatrixUtil.getTransformationMatrix(mBitmap.getWidth(), mBitmap.getHeight(),
                mBuilder.inputWidth, mBuilder.inputHeight, mRotationDegrees, false);
        Matrix cropToFrameTrans = new Matrix();
        frameToCropTransform.invert(cropToFrameTrans);

        Bitmap croppedBitmap = Bitmap.createBitmap(mBuilder.inputWidth, mBuilder.inputHeight, Bitmap.Config.RGB_565);
        Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(mBitmap, frameToCropTransform, null);

        croppedBitmap.getPixels(mIntValues, 0, croppedBitmap.getWidth(), 0, 0, croppedBitmap.getWidth(),
                croppedBitmap.getHeight());
        mImgData.rewind();

        for (int i = 0; i < mBuilder.inputWidth; ++i) {
            for (int j = 0; j < mBuilder.inputHeight; ++j) {
                int pixelValue = mIntValues[i * mBuilder.inputWidth + j];
                if (mBuilder.modelQuantized) {
                    // 量化过的模型
                    mImgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    mImgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    mImgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    mImgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    mImgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    mImgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }

        // 创建输入对象和输出结果容器
        Object[] inputArray = { mImgData };
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);

        // 开始执行
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

        // 检测结果的对象个数
        int numDetectionsOutput = min(mBuilder.numDetections, (int) numDetections[0]);

        // 转换结果，并返回
        final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
        for (int i = 0; i < numDetectionsOutput; ++i) {
            final RectF detection = new RectF(
                    outputLocations[0][i][1] * mBuilder.inputWidth,
                    outputLocations[0][i][0] * mBuilder.inputWidth,
                    outputLocations[0][i][3] * mBuilder.inputWidth,
                    outputLocations[0][i][2] * mBuilder.inputWidth);

            recognitions.add(new Recognition((int) outputClasses[0][i], "", outputScores[0][i], detection));
        }

        return recognitions;
    }

    /**
     * 赋值要检测的图像
     */
    public DetectorRunner setBitmap(Bitmap bitmap) {
        mBitmap = bitmap;
        return this;
    }

    /**
     * 赋值坐标转换矩阵
     */
    public DetectorRunner setMatrix(Matrix matrix) {
        mMatrix = matrix;
        return this;
    }

    /**
     * 赋值角度
     */
    public DetectorRunner setRotationDegrees(int rotationDegrees) {
        mRotationDegrees = rotationDegrees;
        return this;
    }

    /**
     * 建造者模式
     */
    public static class Builder {
        // 图像输入宽度
        private int inputWidth = 300;
        // 图像输入高度
        private int inputHeight = 300;
        // 模型是否已经量化过
        private boolean modelQuantized = true;
        // 每次要检测的最大数量
        private int numDetections = 10;

        public Builder() {}

        public Builder setInputWidth(int inputWidth) {
            this.inputWidth = inputWidth;
            return this;
        }

        public Builder setInputHeight(int inputHeight) {
            this.inputHeight = inputHeight;
            return this;
        }

        public Builder setInputSize(int inputSize) {
            setInputWidth(inputSize);
            setInputHeight(inputSize);
            return this;
        }

        public Builder setModelQuantized(boolean modelQuantized) {
            this.modelQuantized = modelQuantized;
            return this;
        }

        public Builder setNumDetections(int numDetections) {
            this.numDetections = numDetections;
            return this;
        }

        public DetectorRunner build() {
            return new DetectorRunner(this);
        }
    }

}
