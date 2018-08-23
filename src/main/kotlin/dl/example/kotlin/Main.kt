package dl.example.kotlin

import com.ichipsea.kotlin.matrix.*
import java.io.File
import kotlin.math.exp
import kotlin.math.roundToInt

class Main {
    // prepare the input data set
    private val dataDirectory = File(javaClass.classLoader.getResource("data").toURI())
    private val trainDataTokenizer = FileTokenizer(File(dataDirectory, "training.data"))
    private val tokens: List<List<String>> = trainDataTokenizer.tokenize()
    // neural network architecture
    private val vocabularyBuilder = VocabularyBuilder()
    private val neuralNetwork = RecurrentNeuralNetwork(ActivationFunction())
    // neural network parameters
    private val embedSize = 10
    private val iterations = 30000
    private val alpha = 0.001

    // test data set
    private val testDataTokenizer = FileTokenizer(File(dataDirectory, "test.data"))

    fun train() {
        val vocabulary: List<String> = vocabularyBuilder.buildVocabulary(tokens)
        val indices: Map<String, Int> = vocabularyBuilder.mapWordToIndex(vocabulary)
        val weights: MutableMatrix<Double> = createMutableMatrix(embedSize, vocabulary.size) { _, _ -> (Math.random() - 0.5) * 0.03 }
        var startSentenceEmbedding: Matrix<Double> = createMatrix(weights.cols, 1) { _, _ -> 0.0 }
        var recurrentMatrix: Matrix<Double> = createMatrix(weights.cols, weights.cols) { c, r -> if (c == r) 1.0 else 0.0 }
        var decoder: Matrix<Double> = createMatrix(weights.rows, weights.cols) { _, _ -> (Math.random() - 0.5) * 0.03 }
        val identityMatrix = createMatrix(weights.rows, weights.rows) { c, r -> if (c == r) 1.0 else 0.0 }

        for (iteration in 0 until iterations) {
            val wordInSentence = tokens[iteration % tokens.size]
            val wordsInSentenceToPredict: List<String> = wordInSentence.drop(1)
            val sentence = vocabularyBuilder.collectSentenceIndices(wordInSentence, indices)
            val normalizedAlpha = alpha / sentence.size
            val (layers, loss) = neuralNetwork.predict(
                    sentence,
                    weights,
                    startSentenceEmbedding = startSentenceEmbedding,
                    recurrentMatrix = recurrentMatrix,
                    decoder = decoder
            )
            (0 until layers.size).reversed().forEach { downIndex ->
                val layer = layers[downIndex]
                val sentenceIndex = if (downIndex > 0) downIndex - 1 else sentence.size - 1
                val target: Int = sentence[sentenceIndex]
                if (downIndex > 0) {
                    layer["output_delta"] = layer["prediction"]!!.toList().mapIndexed { i, it -> it - identityMatrix[target, i] }.toMatrix(weights.rows, 1)
                    val newHiddenDelta = layer["output_delta"]!! dot decoder.asTransposed()
                    if (downIndex == layers.size - 1) {
                        layer["hidden_delta"] = newHiddenDelta
                    } else {
                        layer["hidden_delta"] = newHiddenDelta + (layers[downIndex + 1]["hidden_delta"]!! dot recurrentMatrix.asTransposed())
                    }
                } else {
                    layer["hidden_delta"] = layers[downIndex + 1]["hidden_delta"]!! dot recurrentMatrix.asTransposed()
                }
            }

            startSentenceEmbedding -= layers[0]["hidden_delta"]!! * normalizedAlpha

            layers.drop(1).forEachIndexed { index, layer ->
                decoder -= (layer["hidden"]!! outer layer["output_delta"]!!) * normalizedAlpha
                val embedIndex = sentence[index]
                (0 until weights.cols).forEach { col -> weights[embedIndex, col] -= layer["hidden_delta"]!![col, 0] * normalizedAlpha }
                recurrentMatrix -= (layer["hidden"]!! outer layer["hidden_delta"]!!) * normalizedAlpha
            }

            if (Math.random() * 100 > 99.7 || iteration == iterations - 1) {
                println("Sentence: $wordInSentence --- Perplexity: ${exp(loss / sentence.size)}")
            }
        }

        // test
        val sentenceIndex = (Math.random() * tokens.size - 1).roundToInt()
        val sentenceToPredict = tokens[sentenceIndex]

        val (l) = neuralNetwork.predict(
                sentence = vocabularyBuilder.collectSentenceIndices(sentenceToPredict, indices),
                weights = weights,
                startSentenceEmbedding = startSentenceEmbedding,
                decoder = decoder,
                recurrentMatrix = recurrentMatrix
        )

        println("Sentence index: $sentenceIndex    Sentence: $sentenceToPredict")

        l.drop(2).forEachIndexed { i, layer ->
            val input = sentenceToPredict[i]
            val trueValue = sentenceToPredict[i + 1]
            val prediction = vocabulary[layer["prediction"]!!.argMax()]
            println("Previous Input: ${input.padEnd(20)} True Value: ${trueValue.padEnd(20)} Prediction: ${prediction.padEnd(20)}")
        }
    }

    companion object {
        @JvmStatic
        fun main(vararg args: String) {
            Main().train()
        }
    }
}

infix fun <T: Number> Matrix<T>.outer(other: Matrix<T>): Matrix<Double> {
    if (this.rows > 1 || other.rows > 1) throw IllegalArgumentException("Matrices do not match")

    return createMatrix(other.cols, this.cols) { c, r ->
        this[r, 0].toDouble() * other[c, 0].toDouble()
    }
}

fun Matrix<Double>.argMax(): Int = this.toList().indexOf(this.toList().max())