package dl.example.kotlin

import io.github.matrix4k.Matrix
import io.github.matrix4k.MutableMatrix
import io.github.matrix4k.createMatrix

data class NeuralNetworkParameters(
        val weights: MutableMatrix<Double>,
        var startSentenceEmbedding: Matrix<Double> = createMatrix(weights.cols, 1) { _, _ -> 0.0 },
        var recurrentMatrix: Matrix<Double> = createMatrix(weights.cols, weights.cols) { c, r -> if (c == r) 1.0 else 0.0 },
        var decoder: Matrix<Double> = createMatrix(weights.rows, weights.cols) { _, _ -> (Math.random() - 0.5) * 0.03 }
)
