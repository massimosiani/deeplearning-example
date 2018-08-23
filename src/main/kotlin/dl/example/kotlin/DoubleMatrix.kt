package dl.example.kotlin
/*
class Matrix<T> {
    var values: List<List<T>> = mutableListOf(mutableListOf())

    fun addRow(row: List<T>): Matrix<T> {
        if (values.first().isNotEmpty() && values.first().size != row.size) throw Exception()

        if (values.first().isEmpty()) values = mutableListOf()

        values.plus(row)
        return this
    }

    fun getNumberOfRows(): Int = values.size

    fun getNumberOfColumns(): Int = values.first().size

    operator fun get(i: Int, j: Int) = values[i][j]

    fun List<List<T>>.toMatrix(): Matrix<T> {
        val result = Matrix<T>()
        result.values = this
        return result
    }
}

fun Matrix<Double>.dotProduct(second: Matrix<Double>): Matrix<Double> {
    val newValues: List<List<Double>> = (this.values).map { row ->
        val newRow = mutableListOf<Double>()
        for (j in 1..second.getNumberOfColumns()) {
            var r = 0.0
            for (k in 1..second.getNumberOfRows()) {
                r += row[k] * second.values[k][j]
            }
            newRow.add(r)
        }
        newRow
    }
    return newValues.toMatrix()
}
*/