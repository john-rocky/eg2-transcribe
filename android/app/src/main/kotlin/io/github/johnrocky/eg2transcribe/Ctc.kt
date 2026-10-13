package io.github.johnrocky.eg2transcribe

/** A word of the transcript: its table row, and the first and last 40 ms token it was read from. */
class Word(val text: String, val id: Int, val firstToken: Int, var lastToken: Int) {
    val startSeconds: Double get() = firstToken * Ctc.SECONDS_PER_TOKEN
    val endSeconds: Double get() = (lastToken + 1) * Ctc.SECONDS_PER_TOKEN
}

/**
 * CTC-style collapse over the token stream, fed window by window in global token order: a run of the same word is one
 * word, a blank ends a run (so "a a _ a" reads "a a"), blanks are dropped. The state carries across windows.
 */
class Ctc(private val vocab: List<String>) {
    companion object {
        const val BLANK = -1
        const val SECONDS_PER_TOKEN = 0.04
    }

    private var prev = BLANK
    private var current: Word? = null

    /** Feeds [n] token ids (BLANK or a table row) starting at global token [firstToken]; returns the words begun here. */
    fun feed(ids: IntArray, n: Int, firstToken: Int): List<Word> {
        val out = ArrayList<Word>()
        for (i in 0 until n) {
            val id = ids[i]
            val g = firstToken + i
            if (id == BLANK) {
                current = null
            } else if (id == prev && current != null) {
                current!!.lastToken = g
            } else {
                val w = Word(vocab[id], id, g, g)
                out += w
                current = w
            }
            prev = id
        }
        return out
    }
}
