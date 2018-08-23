package dl.example.kotlin

class VocabularyBuilder {

    fun buildVocabulary(tokens: List<List<String>>) = tokens.flatMap { it }.toSet().toList()

    fun mapWordToIndex(vocabulary: List<String>) =
        vocabulary.foldIndexed(mutableMapOf<String, Int>()) {
            index, acc, element -> acc.put(element, index); acc
        }.toMap()

    fun collectSentenceIndices(sentence: List<String>, indices: Map<String, Int>): List<Int> = sentence.map { indices[it]!! }
}