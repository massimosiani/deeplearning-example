package dl.example.kotlin

import com.ichipsea.kotlin.matrix.*
import kotlin.math.ln

data class PredictionResult(val layers: List<MutableMap<String, Matrix<Double>>>, val loss: Double)

class RecurrentNeuralNetwork(
        private val activationFunction: ActivationFunction) {

    fun predict(
            sentence: List<Int>,
            weights: Matrix<Double>,
            recurrentMatrix: Matrix<Double> = createMatrix(weights.cols, weights.cols) { c, r -> if (c == r) 1.0 else 0.0 },
            startSentenceEmbedding: Matrix<Double> = createMatrix(weights.cols, 1) { _, _ -> 0.0 },
            decoder: Matrix<Double> = createMatrix(weights.rows, weights.cols) { _, _ -> 0.1 }): PredictionResult {

        val layers = mutableListOf(mutableMapOf("hidden" to startSentenceEmbedding))
        var loss = 0.0
        sentence.forEach { word ->
            val prediction = activationFunction.softmax((decoder x layers.last()["hidden"]!!).toList())
            loss -= ln(prediction[word])
            val hidden = (recurrentMatrix x layers.last()["hidden"]!! + List(weights.cols) { i -> weights[word, i] }.toMatrix(weights.cols, 1))
            layers.add(
                    mutableMapOf(
                            "prediction" to prediction.toMatrix(weights.rows, 1),
                            "hidden" to hidden))
        }

        return PredictionResult(layers, loss)
    }
}

fun main (vararg args: String) {
    val a = RecurrentNeuralNetwork(ActivationFunction()).predict(
            sentence = List(2) { i -> i},
            weights = createMatrix(2, 2) { _, _ -> 1.0 }
    )
    println(a.layers)
    println(a.loss)
}