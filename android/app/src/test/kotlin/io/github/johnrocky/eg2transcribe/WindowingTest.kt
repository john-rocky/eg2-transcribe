package io.github.johnrocky.eg2transcribe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WindowingTest {
    /**
     * Every token is kept by exactly one window, in order; windows before the last are full and stop dropLast tokens
     * short of their end; each kept token's 7 frames are in its window, except the recording's last token when the
     * recording ends inside it (its missing frames are masked, as in the reader's training data).
     */
    private fun checkCover(p: WindowPlan) {
        var next = 0
        for (k in 0 until p.count) {
            val r = p.kept(k)
            assertEquals("window $k starts where k - 1 stopped", next, r.first)
            assertTrue("window $k keeps at least one token", p.tokens == 0 || r.last >= r.first)
            assertEquals(0, p.start(k) % 4)
            assertEquals(4 * p.firstToken(k), p.start(k))
            if (k > 0) assertEquals(p.overlapTokens, p.keepFrom(k))
            if (k < p.count - 1) {
                assertEquals(p.windowFrames, p.validFrames(k))
                assertEquals(p.windowTokens - p.dropLast, p.keepUntil(k))
            }
            for (g in r) {
                val j = g - p.firstToken(k) // token within the window
                val partial = g == p.tokens - 1 && p.frames % 4 != 0
                if (!partial) assertTrue("token $g reads frame ${4 * j + 3} of a window with ${p.validFrames(k)} real frames", 4 * j + 3 < p.validFrames(k))
            }
            next = if (r.isEmpty()) next else r.last + 1
        }
        assertEquals("all tokens", p.tokens, next)
        val last = p.count - 1
        assertEquals("the last window reaches the last token", p.tokens, p.firstToken(last) + p.validTokens(last))
        if (p.count > 1) assertTrue("one window less would not", p.firstToken(last - 1) + p.windowTokens < p.tokens)
    }

    @Test
    fun syntheticLengths() {
        // Sample counts -> frames -> plan, for both window sizes and several overlaps, with and without the dropped token.
        val lengths = listOf(16_000L, 159_999L, 160_160L, 479_999L, 480_000L, 480_161L, 480_321L, 480_801L, 961_440L, 1_234_567L, 52_119_092L)
        for (n in lengths) for (w in listOf(WindowPlan.FRAMES_10S, WindowPlan.FRAMES_30S)) for (o in listOf(0, 24, 96, WindowPlan.maxOverlap(w))) for (d in 0..1) {
            val f = MelFrontEnd.frames(n)
            val p = WindowPlan(f, w, o, d)
            assertEquals(f, (n - 161).toInt().let { if (it < 0) 0 else it / 160 + 1 })
            assertEquals((f + 3) / 4, p.tokens)
            checkCover(p)
        }
    }

    @Test
    fun theDefaultsOnTheLongFixture() {
        // 54.3 min: 325,744 frames, 81,436 tokens; 30 s windows step 653 tokens (750 - 96 overlap - 1 dropped) = the
        // Core AI port's plan: 125 windows, the last at token 80,972 with 1,856 frames keeping [96, 464).
        val p = WindowPlan(325_744, WindowPlan.FRAMES_30S, WindowPlan.DEFAULT_OVERLAP)
        assertEquals(653, p.step)
        assertEquals(2612, p.stride)
        assertEquals(125, p.count)
        assertEquals(81_436, p.tokens)
        assertEquals(750.0 / 653, p.computeFactor, 1e-9)
        assertEquals(0 until 749, p.kept(0))
        assertEquals(653, p.firstToken(1))
        assertEquals(749 until 1402, p.kept(1))
        val last = p.count - 1
        assertEquals(80_972, p.firstToken(last))
        assertEquals(323_888, p.start(last))
        assertEquals(1856, p.validFrames(last)) // > 1000: the last window runs on the 30 s signature too
        assertEquals(96, p.keepFrom(last))
        assertEquals(464, p.keepUntil(last))
        assertEquals(81_068 until 81_436, p.kept(last))
        checkCover(p)
    }

    @Test
    fun theEndOfTheRecording() {
        // 30 s exactly = 480,000 samples = 2,999 frames = 750 tokens (the last one reads a missing frame, masked): one
        // window that keeps all 750, as the reader's fixtures decode their soft_tokens.npy.
        val p = WindowPlan(2999, WindowPlan.FRAMES_30S, 96)
        assertEquals(1, p.count)
        assertEquals(750, p.validTokens(0))
        assertEquals(0 until 750, p.kept(0))
        assertEquals(0 until 750, WindowPlan(3000, WindowPlan.FRAMES_30S, 96).kept(0))
        // One frame more starts token 750: a second window from token 653 keeps [749, 751).
        val q = WindowPlan(3001, WindowPlan.FRAMES_30S, 96)
        assertEquals(2, q.count)
        assertEquals(0 until 749, q.kept(0))
        assertEquals(389, q.validFrames(1))
        assertEquals(749 until 751, q.kept(1))
        // test-clean 1089-134686-0002: 662 frames, 166 tokens.
        val r = WindowPlan(662, WindowPlan.FRAMES_10S, 96)
        assertEquals(1, r.count)
        assertEquals(166, r.validTokens(0))
        assertEquals(166, r.tokens)
        assertEquals(0 until 166, r.kept(0))
    }

    @Test
    fun tenSecondWindowsOnTheLongFixture() {
        // The NPU's windows: 10 s with the same contract step 153 tokens (612 frames, 1.634x the encoder work).
        val p = WindowPlan(325_744, WindowPlan.FRAMES_10S, WindowPlan.DEFAULT_OVERLAP)
        assertEquals(153, p.step)
        assertEquals(612, p.stride)
        assertEquals(532, p.count)
        assertEquals(81_436, p.tokens)
        assertEquals(250.0 / 153, p.computeFactor, 1e-9)
        assertEquals(0 until 249, p.kept(0))
        assertEquals(153, p.firstToken(1))
        assertEquals(96, p.keepFrom(1))
        assertEquals(249 until 402, p.kept(1))
        assertEquals(402 until 555, p.kept(2))
        val last = p.count - 1
        assertEquals(324_972, p.start(last))
        assertEquals(772, p.validFrames(last))
        assertEquals(193, p.keepUntil(last))
        assertEquals(81_339 until 81_436, p.kept(last))
        checkCover(p)
    }

    @Test
    fun tenSecondWindowsAtTheEndOfARecording() {
        // f30 on the NPU: 2,999 frames in 5 windows of 10 s.
        val p = WindowPlan(2999, WindowPlan.FRAMES_10S, 96)
        assertEquals(5, p.count)
        assertEquals(listOf(0, 612, 1224, 1836, 2448), (0 until p.count).map { p.start(it) })
        assertEquals(551, p.validFrames(4))
        assertEquals(138, p.validTokens(4))
        assertEquals(138, p.keepUntil(4))
        assertEquals(708 until 750, p.kept(4))
        assertEquals(750, p.tokens)
        checkCover(p)
        // Exactly one 10 s window is one window; a frame more is two.
        assertEquals(1, WindowPlan(1000, WindowPlan.FRAMES_10S, 96).count)
        assertEquals(0 until 250, WindowPlan(1000, WindowPlan.FRAMES_10S, 96).kept(0))
        val q = WindowPlan(1001, WindowPlan.FRAMES_10S, 96)
        assertEquals(2, q.count)
        assertEquals(249 until 251, q.kept(1))
    }

    @Test
    fun autoPicksTheTenSecondFilesForOneWindowWithATokenToSpare() {
        // 249 tokens = one 10 s window with its right-edge token free: the 10 s files; 250 fill the window: 30 s.
        assertEquals(249, WindowPlan.AUTO_MAX_TOKENS_10S)
        assertEquals(WindowPlan.FRAMES_10S, WindowPlan.autoFrames(249, has10s = true, has30s = true))
        assertEquals(WindowPlan.FRAMES_30S, WindowPlan.autoFrames(250, has10s = true, has30s = true))
        assertEquals(WindowPlan.FRAMES_10S, WindowPlan.autoFrames(1, has10s = true, has30s = true))
        assertEquals(WindowPlan.FRAMES_30S, WindowPlan.autoFrames(81_436, has10s = true, has30s = true))
        // In samples: 159,520 samples (9.97 s) = 996 frames = 249 tokens; one sample more starts frame 997 = token 250.
        assertEquals(996, MelFrontEnd.frames(159_520))
        assertEquals(249, WindowPlan.tokensFor(MelFrontEnd.frames(159_520)))
        assertEquals(250, WindowPlan.tokensFor(MelFrontEnd.frames(159_521)))
        // Without a 10 s file (the encoder's or the reader's) a short recording takes the 30 s files; without a 30 s
        // file a long one takes the 10 s files (windows of 10 s), and with no file at all the answer is 30 s (the load
        // then says what is missing).
        assertEquals(WindowPlan.FRAMES_30S, WindowPlan.autoFrames(249, has10s = false, has30s = true))
        assertEquals(WindowPlan.FRAMES_10S, WindowPlan.autoFrames(250, has10s = true, has30s = false))
        assertEquals(WindowPlan.FRAMES_10S, WindowPlan.autoFrames(249, has10s = true, has30s = false))
        assertEquals(WindowPlan.FRAMES_30S, WindowPlan.autoFrames(249, has10s = false, has30s = false))
        // The 20 test-clean fixtures (reader/fixtures/MANIFEST.json n_samples): 11 fit one 10 s window, 9 do not.
        val fixtures = mapOf(
            "1089-134686-0000" to 166_960L, "1089-134686-0002" to 106_000L, "1188-133604-0000" to 171_600L, "1188-133604-0001" to 144_640L,
            "121-121726-0000" to 135_360L, "121-121726-0001" to 94_800L, "1221-135766-0000" to 198_960L, "1221-135766-0001" to 267_440L,
            "1284-1180-0000" to 129_920L, "1284-1180-0001" to 124_080L, "1320-122612-0000" to 215_680L, "1320-122612-0001" to 152_320L,
            "1580-141083-0000" to 143_040L, "1580-141083-0001" to 164_080L, "1995-1826-0000" to 151_760L, "1995-1826-0001" to 162_720L,
            "2094-142345-0001" to 128_480L, "2094-142345-0002" to 248_320L, "2300-131720-0000" to 81_280L, "2300-131720-0001" to 234_080L,
        )
        val short = fixtures.filterValues { WindowPlan.autoFrames(WindowPlan.tokensFor(MelFrontEnd.frames(it)), true, true) == WindowPlan.FRAMES_10S }.keys
        assertEquals(11, short.size)
        assertTrue("1284-1180-0000" in short) // the bundled example: 203 tokens
        assertEquals(203, WindowPlan.tokensFor(MelFrontEnd.frames(129_920L)))
        // A short recording in its 10 s window: one window that keeps every token.
        val p = WindowPlan(MelFrontEnd.frames(129_920L), WindowPlan.FRAMES_10S, 96)
        assertEquals(1, p.count)
        assertEquals(0 until 203, p.kept(0))
    }

    @Test(expected = IllegalArgumentException::class)
    fun overlapTooLarge() {
        WindowPlan(10_000, WindowPlan.FRAMES_10S, WindowPlan.maxOverlap(WindowPlan.FRAMES_10S) + 1)
    }
}
