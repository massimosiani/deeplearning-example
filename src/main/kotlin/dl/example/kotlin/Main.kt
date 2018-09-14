package dl.example.kotlin

import io.github.matrix4k.*
import java.io.File
import kotlin.math.exp
import kotlin.math.roundToInt

class Main {
    // prepare the input data set
    private val dataDirectory = File(javaClass.classLoader.getResource("data").toURI())
    private val trainDataTokenizer = FileTokenizer(File(dataDirectory, "training.data"))
    private val tokens: List<List<String>> = trainDataTokenizer.tokenize()
    private val vocabularyBuilder = VocabularyBuilder()
    private val vocabulary: List<String> = vocabularyBuilder.buildVocabulary(tokens)
    private val indices: Map<String, Int> = vocabularyBuilder.mapWordToIndex(vocabulary)
    // neural network architecture
    private val neuralNetwork = RecurrentNeuralNetwork(ActivationFunction())
    // neural network parameters
    private val embedSize = 10
    private val iterations = 30000

    // test data set
    private val testDataTokenizer = FileTokenizer(File(dataDirectory, "test.data"))
    private val testTokens = testDataTokenizer.tokenize()

    fun initInputDataSet(): NeuralNetworkParameters = NeuralNetworkParameters(weights = createMutableMatrix(embedSize, vocabulary.size) { _, _ -> (Math.random() - 0.5) * 0.03 })

    fun train(neuralNetworkParameters: NeuralNetworkParameters): NeuralNetworkParameters {
        var nnp = neuralNetworkParameters
        for (iteration in 0 until iterations) {
            val wordInSentence = tokens[iteration % tokens.size]
            val sentence = vocabularyBuilder.collectSentenceIndices(wordInSentence, indices)

            var (layers, loss) = neuralNetwork.predict(
                    sentence,
                    nnp.weights,
                    startSentenceEmbedding = nnp.startSentenceEmbedding,
                    recurrentMatrix = nnp.recurrentMatrix,
                    decoder = nnp.decoder
            )

            layers = neuralNetwork.computeCorrections(layers, sentence, nnp)

            nnp = neuralNetwork.adjustNeuralNetworkParameters(layers, sentence, nnp)

            if (Math.random() * 100 > 99.7 || iteration == iterations - 1) {
                println("Sentence: ${wordInSentence.joinToString(separator = " ")} --- Perplexity: ${exp(loss / sentence.size)}")
            }
        }

        return nnp
    }

    fun test(neuralNetworkParameters: NeuralNetworkParameters) {
        val sentenceIndex = (Math.random() * testTokens.size - 1).roundToInt()
        val sentenceToPredict = testTokens[sentenceIndex]

        val (l, _) = neuralNetwork.predict(
                sentence = vocabularyBuilder.collectSentenceIndices(sentenceToPredict, indices),
                weights = neuralNetworkParameters.weights,
                startSentenceEmbedding = neuralNetworkParameters.startSentenceEmbedding,
                decoder = neuralNetworkParameters.decoder,
                recurrentMatrix = neuralNetworkParameters.recurrentMatrix
        )


        println()
        println("Prediction:")
        println("Sentence index: $sentenceIndex    Sentence: ${sentenceToPredict.joinToString(separator = " ")}")

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
            val program = Main()
            val neuralNetworkData = program.initInputDataSet()
            program.train(neuralNetworkData)
            program.test(neuralNetworkData)
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