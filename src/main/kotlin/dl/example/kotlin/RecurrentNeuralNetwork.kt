package dl.example.kotlin

import io.github.matrix4k.*
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
            val prediction = activationFunction.softmax((layers.last()["hidden"]!! dot decoder).toList())
            loss -= ln(prediction[word])
            val hidden = (layers.last()["hidden"]!! dot recurrentMatrix) + List(weights.cols) { i -> weights[word, i] }.toMatrix(weights.cols, 1)
            layers.add(
                    mutableMapOf(
                            "prediction" to prediction.toMatrix(weights.rows, 1),
                            "hidden" to hidden))
        }

        return PredictionResult(layers, loss)
    }

    fun computeCorrections(
            layers: List<MutableMap<String, Matrix<Double>>>,
            sentence: List<Int>,
            neuralNetworkParameters: NeuralNetworkParameters): List<MutableMap<String, Matrix<Double>>> {
        val identityMatrix = createMatrix(neuralNetworkParameters.weights.rows, neuralNetworkParameters.weights.rows) { c, r -> if (c == r) 1.0 else 0.0 }

        (0 until layers.size).reversed().forEach { downIndex ->
            val layer = layers[downIndex]
            val sentenceIndex = if (downIndex > 0) downIndex - 1 else sentence.size - 1
            val target: Int = sentence[sentenceIndex]
            if (downIndex > 0) {
                layer["outputDelta"] = layer["prediction"]!!.toList().mapIndexed { i, it -> it - identityMatrix[target, i] }.toMatrix(neuralNetworkParameters.weights.rows, 1)
                val newHiddenDelta = layer["outputDelta"]!! dot neuralNetworkParameters.decoder.asTransposed()
                if (downIndex == layers.size - 1) {
                    layer["hiddenDelta"] = newHiddenDelta
                } else {
                    layer["hiddenDelta"] = newHiddenDelta + (layers[downIndex + 1]["hiddenDelta"]!! dot neuralNetworkParameters.recurrentMatrix.asTransposed())
                }
            } else {
                layer["hiddenDelta"] = layers[downIndex + 1]["hiddenDelta"]!! dot neuralNetworkParameters.recurrentMatrix.asTransposed()
            }
        }

        return layers
    }

    fun adjustNeuralNetworkParameters(
            layers: List<MutableMap<String, Matrix<Double>>>,
            sentence: List<Int>,
            neuralNetworkParameters: NeuralNetworkParameters): NeuralNetworkParameters {
        val normalizedAlpha = neuralNetworkParameters.alpha / sentence.size

        neuralNetworkParameters.startSentenceEmbedding -= layers[0]["hiddenDelta"]!! * normalizedAlpha
        layers.drop(1).forEachIndexed { index, layer ->
            neuralNetworkParameters.decoder -= (layer["hidden"]!! outer layer["outputDelta"]!!) * normalizedAlpha
            val embedIndex = sentence[index]
            (0 until neuralNetworkParameters.weights.cols).forEach { col -> neuralNetworkParameters.weights[embedIndex, col] -= layer["hiddenDelta"]!![col, 0] * normalizedAlpha }
            neuralNetworkParameters.recurrentMatrix -= (layer["hidden"]!! outer layer["hiddenDelta"]!!) * normalizedAlpha
        }

        return neuralNetworkParameters
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