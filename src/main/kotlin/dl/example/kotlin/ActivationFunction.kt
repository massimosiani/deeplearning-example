package dl.example.kotlin

class ActivationFunction {

    fun softmax(x: List<Double>): List<Double> {
        val max = x.max() as Double
        val exps = x.map { Math.exp(it - max) }
        val sum = exps.sum()
        return exps.map { it / sum }
    }
}
