package io.github.johnrocky.eg2transcribe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WindowingTest {
    /** Every whole token is kept by exactly one window, in order, and each kept token's 7 frames are in its window. */
    private fun checkCover(p: WindowPlan) {
        var next = 0
        for (k in 0 until p.count) {
            val r = p.kept(k)
            assertEquals("window $k starts where k - 1 stopped", next, r.first)
            assertTrue("window $k keeps at least one token", p.tokens == 0 || r.last >= r.first)
            assertEquals(0, p.start(k) % 4)
            if (k > 0) assertEquals(p.overlapTokens, p.keepFrom(k))
            if (k < p.count - 1) assertEquals(p.windowFrames, p.validFrames(k))
            for (g in r) {
                val j = g - p.firstToken(k) // token within the window
                assertTrue("token $g reads frame ${4 * j + 3} of a window with ${p.validFrames(k)} real frames", 4 * j + 3 < p.validFrames(k))
            }
            next = if (r.isEmpty()) next else r.last + 1
        }
        assertEquals("all tokens", p.tokens, next)
        assertTrue("the last window reaches the last whole token", p.start(p.count - 1) + p.windowFrames >= p.wholeFrames)
        if (p.count > 1) assertTrue("one window less would not", p.start(p.count - 2) + p.windowFrames < p.wholeFrames)
    }

    @Test
    fun syntheticLengths() {
        // Sample counts -> frames -> plan, for both window sizes and several overlaps.
        val lengths = listOf(16_000L, 159_999L, 160_160L, 479_999L, 480_000L, 480_161L, 480_321L, 480_801L, 961_440L, 1_234_567L, 52_119_092L)
        for (n in lengths) for (w in listOf(WindowPlan.FRAMES_10S, WindowPlan.FRAMES_30S)) for (o in listOf(0, 24, 96, WindowPlan.maxOverlap(w))) {
            val f = MelFrontEnd.frames(n)
            val p = WindowPlan(f, w, o)
            assertEquals(f, (n - 161).toInt().let { if (it < 0) 0 else it / 160 + 1 })
            assertEquals(f / 4, p.tokens)
            checkCover(p)
        }
    }

    @Test
    fun theDefaultsOnTheLongFixture() {
        // 54.3 min: 325,744 frames (a multiple of 4: every token whole), 81,436 tokens; 30 s windows advance 2,616 frames.
        val p = WindowPlan(325_744, WindowPlan.FRAMES_30S, WindowPlan.DEFAULT_OVERLAP)
        assertEquals(2616, p.stride)
        assertEquals(125, p.count)
        assertEquals(81_436, p.tokens)
        assertEquals(1.147, p.computeFactor, 0.001)
        assertEquals(0 until 750, p.kept(0))
        assertEquals(750 until 1404, p.kept(1)) // window 1 starts at frame 2616 = token 654, keeps from token 750
        assertEquals(654, p.firstToken(1))
        val last = p.count - 1
        assertEquals(324_384, p.start(last))
        assertEquals(1360, p.validFrames(last)) // > 1000: the last window runs on the 30 s signature too
        assertEquals(81_436 - 1, p.kept(last).last)
        checkCover(p)
    }

    @Test
    fun theEndOfTheRecording() {
        // 30 s exactly = 480,000 samples = 2,999 frames: token 749 would read frame 2,999, which does not exist.
        val p = WindowPlan(2999, WindowPlan.FRAMES_30S, 96)
        assertEquals(1, p.count)
        assertEquals(750, p.validTokens(0)) // what the encoder marks valid
        assertEquals(0 until 749, p.kept(0)) // what is kept
        // One frame more makes token 749 whole; a frame that only starts token 750 adds no window.
        assertEquals(0 until 750, WindowPlan(3000, WindowPlan.FRAMES_30S, 96).kept(0))
        assertEquals(1, WindowPlan(3001, WindowPlan.FRAMES_30S, 96).count)
        val q = WindowPlan(3004, WindowPlan.FRAMES_30S, 96)
        assertEquals(2, q.count)
        assertEquals(750 until 751, q.kept(1))
        // test-clean 1089-134686-0002: 662 frames, 166 tokens marked valid, 165 whole.
        val r = WindowPlan(662, WindowPlan.FRAMES_10S, 96)
        assertEquals(1, r.count)
        assertEquals(166, r.validTokens(0))
        assertEquals(165, r.tokens)
    }

    @Test(expected = IllegalArgumentException::class)
    fun overlapTooLarge() {
        WindowPlan(10_000, WindowPlan.FRAMES_10S, WindowPlan.maxOverlap(WindowPlan.FRAMES_10S) + 1)
    }
}
