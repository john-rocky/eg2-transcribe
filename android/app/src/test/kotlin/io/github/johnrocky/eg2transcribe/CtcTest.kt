package io.github.johnrocky.eg2transcribe

import org.junit.Assert.assertEquals
import org.junit.Test

class CtcTest {
    private val vocab = listOf("he", "hoped", "there", "would", "be", "stew")
    private val B = Ctc.BLANK

    private fun text(words: List<Word>) = words.joinToString(" ") { it.text }

    @Test
    fun collapsesRunsAndDropsBlanks() {
        val ctc = Ctc(vocab)
        val ids = intArrayOf(B, 0, 0, B, 1, 1, 1, 2, B, B, 3, 4, 4, B, 4, 5)
        val words = ctc.feed(ids, ids.size, 0)
        assertEquals("he hoped there would be be stew", text(words))
        assertEquals(1, words[0].firstToken)
        assertEquals(2, words[0].lastToken)
        assertEquals(4, words[1].firstToken)
        assertEquals(6, words[1].lastToken)
        assertEquals(0.04 * 15, words.last().startSeconds, 1e-9)
    }

    @Test
    fun runsContinueAcrossWindows() {
        val ctc = Ctc(vocab)
        val a = ctc.feed(intArrayOf(0, 1, 1), 3, 0)
        val b = ctc.feed(intArrayOf(1, 1, 2), 3, 3) // "hoped" continues into the next window: one word
        assertEquals("he hoped", text(a))
        assertEquals("there", text(b))
        assertEquals(4, a[1].lastToken)
        val c = ctc.feed(intArrayOf(B), 1, 6)
        val d = ctc.feed(intArrayOf(2), 1, 7) // a blank between windows splits the run
        assertEquals("", text(c))
        assertEquals("there", text(d))
    }

    @Test
    fun countOnlyTheFirstN() {
        val ctc = Ctc(vocab)
        assertEquals("he", text(ctc.feed(intArrayOf(0, 5, 5), 1, 0)))
    }
}
