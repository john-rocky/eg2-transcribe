import SwiftUI
import UniformTypeIdentifiers

/// One screen: the model's state, Choose audio / Record, the progress, the transcript and its numbers.
struct ContentView: View {
    @Bindable var model: AppModel
    @State private var importing = false
    @State private var importKind = UTType.audio
    @State private var showingSettings = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let line = model.selfTestLine {
                        Text(line).font(.footnote.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                    status
                    actions
                    if case .transcribing(let done, let total) = model.phase {
                        VStack(alignment: .leading, spacing: 6) {
                            ProgressView(value: Double(done), total: Double(max(total, 1)))
                            Text(total > 0 ? "Window \(done) of \(total)" : "Reading the audio…")
                                .font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                    TranscriptView(text: model.transcript, file: model.transcript.isEmpty ? nil : model.stats?.file,
                                   size: model.textSize)
                    if let stats = model.stats { StatsView(stats: stats, size: model.textSize) }
                }
                .padding()
                .frame(maxWidth: 720, alignment: .leading)
            }
            .navigationTitle("EG2 Transcribe")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { showingSettings = true } label: { Label("Settings", systemImage: "gearshape") }
                        .disabled(model.forms.isEmpty || model.isBusy || model.isRecording)
                }
            }
            .sheet(isPresented: $showingSettings) { SettingsView(model: model) }
            .fileImporter(isPresented: $importing, allowedContentTypes: [importKind]) { result in
                guard case .success(let url) = result else { return }
                if importKind == .folder {
                    Task { await model.open(url) }
                } else {
                    Task { await model.transcribe(url) }
                }
            }
        }
    }

    private func choose(_ kind: UTType) {
        importKind = kind
        importing = true
    }

    @ViewBuilder private var status: some View {
        switch model.phase {
        case .starting:
            Label("Looking for the model…", systemImage: "hourglass").foregroundStyle(.secondary)
        case .noModel(let hint):
            VStack(alignment: .leading, spacing: 8) {
                Label("No model folder", systemImage: "folder.badge.questionmark").font(.headline)
                Text(hint).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                HStack {
                    if ModelFiles.isPublished {
                        Button("Download") { Task { await model.download() } }
                            .buttonStyle(.borderedProminent).frame(minHeight: 44)
                    }
                    #if os(macOS)
                    Button("Choose model folder…") { choose(.folder) }.frame(minHeight: 44)
                    #endif
                }
            }
        case .downloading(let fraction):
            VStack(alignment: .leading) {
                Text("Downloading the model…")
                ProgressView(value: fraction)
            }
        case .loading(let message):
            HStack(spacing: 8) {
                ProgressView()
                Text(message).font(.callout).foregroundStyle(.secondary)
            }
        case .ready, .recording, .transcribing:
            if let form = model.form {
                Label("\(form.title) · overlap \(model.overlap ? "on" : "off")", systemImage: "checkmark.circle")
                    .font(.callout).foregroundStyle(.secondary)
            }
        case .failed(let message):
            VStack(alignment: .leading, spacing: 8) {
                Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.red).textSelection(.enabled)
                HStack {
                    Button("OK") { model.dismissError() }.frame(minHeight: 44)
                    #if os(macOS)
                    Button("Choose model folder…") { choose(.folder) }.frame(minHeight: 44)
                    #endif
                }
            }
        }
    }

    private var actions: some View {
        HStack(spacing: 12) {
            Button { choose(.audio) } label: {
                Label("Choose audio", systemImage: "waveform").frame(maxWidth: .infinity, minHeight: 44)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.canTranscribe)

            Button { Task { await model.toggleRecording() } } label: {
                Label(model.isRecording ? "Stop" : "Record", systemImage: model.isRecording ? "stop.circle.fill" : "mic.fill")
                    .frame(maxWidth: .infinity, minHeight: 44)
            }
            .buttonStyle(.bordered)
            .tint(model.isRecording ? .red : .accentColor)
            .disabled(!model.canTranscribe && !model.isRecording)
        }
    }
}
