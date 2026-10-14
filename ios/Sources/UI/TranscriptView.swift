import SwiftUI
#if os(iOS)
import UIKit
#else
import AppKit
#endif

/// The transcript: selectable, with the file it is of and a Copy button.
struct TranscriptView: View {
    let text: String
    /// The audio file's name (nil until there is a transcript).
    let file: String?
    /// Points; 0 = the platform's body size.
    let size: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("Transcript").font(.headline)
                if let file {
                    Text(file).font(.subheadline).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
                Spacer()
                Button {
                    copy(text)
                } label: {
                    Label("Copy", systemImage: "doc.on.doc").frame(minWidth: 44, minHeight: 44)
                }
                .disabled(text.isEmpty)
            }
            Text(text.isEmpty ? "Choose an audio file or record to see its words here." : text)
                .font(size > 0 ? .system(size: size) : .body)
                .foregroundStyle(text.isEmpty ? .secondary : .primary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func copy(_ text: String) {
        #if os(iOS)
        UIPasteboard.general.string = text
        #else
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #endif
    }
}
