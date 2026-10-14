import Foundation

/// The reader's read-out on the host: one label per token, then greedy CTC.
enum Greedy {
    /// Per token: blank (= the vocabulary size) where blank_logit > top1_logit, else top1_index. The strict
    /// comparison is the reader's own (`model.greedy_labels`): a tie reads the word.
    @inline(__always)
    static func label(index: Int32, top: Float, blank: Float, blankID: Int) -> Int {
        blank > top ? blankID : Int(index)
    }

    /// Merge repeated labels, then drop blanks (`model.collapse`): a word said twice needs a blank between.
    static func collapse(_ labels: [Int], blank: Int) -> [Int] {
        var words: [Int] = []
        var previous: Int?
        for label in labels {
            if label != blank && label != previous { words.append(label) }
            previous = label
        }
        return words
    }
}
