package ceneax.lib.tensorflow.interfaces;

import org.tensorflow.lite.Interpreter;

public interface IRunner<T> {

    T run(Interpreter interpreter);

}
