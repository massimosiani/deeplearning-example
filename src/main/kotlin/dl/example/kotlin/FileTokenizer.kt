package dl.example.kotlin

import java.io.File

class FileTokenizer(private val file: File) {
    fun tokenize(): List<List<String>> = file
            .readLines(Charsets.UTF_8)
            .take(1000)
            .map {
                it.toLowerCase()
                        .replace("\n", "")
                        .replace(Regex("[\\d.]"), "")
                        .trim()
                        .split(" ").let { l -> l.map { e -> e.trim() } }
            }
}
