package ceneax.lib.tensorflow;

import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;

import ceneax.lib.tensorflow.interfaces.IRunner;
import ceneax.lib.tensorflow.util.ModelUtil;

public class TFLite {

    // 单例
    private static TFLite instance;

    // TensorFLow 解释器
    private Interpreter interpreter;

    public synchronized static TFLite getInstance() {
        if (instance == null) {
            instance = new TFLite();
        }
        return instance;
    }

    private TFLite() {}

    // --------------------- 加载模型 ---------------------
    public void loadModel(File modelFile) {
        release();
        interpreter = new Interpreter(modelFile);
    }
    public void loadModel(File modelFile, Interpreter.Options options) {
        release();
        interpreter = new Interpreter(modelFile, options);
    }

    public void loadModel(ByteBuffer byteBuffer) {
        release();
        interpreter = new Interpreter(byteBuffer);
    }
    public void loadModel(ByteBuffer byteBuffer, Interpreter.Options options) {
        release();
        interpreter = new Interpreter(byteBuffer, options);
    }

    public void loadModel(MappedByteBuffer mappedByteBuffer) {
        release();
        interpreter = new Interpreter(mappedByteBuffer);
    }
    public void loadModel(AssetManager assetManager, String fileName) throws IOException {
        loadModel(ModelUtil.loadModelFile(assetManager, fileName));
    }
    // --------------------- 加载模型 ---------------------

    /**
     * 获取 TensorFLow 解释器
     */
    public Interpreter getInterpreter() {
        return interpreter;
    }

    /**
     * 释放资源
     */
    public void release() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }

    /**
     * 开始执行
     */
    public <T> T run(IRunner<T> runner) {
        return runner.run(interpreter);
    }

}
