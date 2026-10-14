import SwiftUI

/// How the transcriber runs: the form of pipeline.json (which bundles, on which compute unit) and whether
/// consecutive windows overlap. The window itself comes from reader_config.json.
struct SettingsView: View {
    @Bindable var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Pipeline") {
                    Picker("Form", selection: formSelection) {
                        ForEach(model.forms) { form in
                            Text(form.title).tag(form.id)
                        }
                    }
                    Toggle("Overlap windows", isOn: $model.overlap)
                    Text("With overlap on, the encoder of the next window runs while the reader reads the current one. The words are the same either way.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Section("Display") {
                    Picker("Transcript text", selection: $model.textSize) {
                        Text("System").tag(0.0)
                        Text("Large").tag(28.0)
                        Text("Extra large").tag(44.0)
                    }
                }
                if let form = model.form {
                    Section("Stages") {
                        ForEach(form.stages, id: \.self) { stage in
                            LabeledContent(stage.name, value: "\(stage.bundle) · \(stage.compute)")
                        }
                    }
                }
                if let folder = model.folder {
                    Section("Model folder") {
                        LabeledContent("Window", value: String(format: "%.0f s, %d tokens of left context",
                                                               folder.reader.windowSeconds, folder.reader.leftOverlapTokens))
                        LabeledContent("Reader", value: folder.reader.version)
                        LabeledContent("Bundles", value: folder.bundleFolders.map { $0.lastPathComponent + "/" }.joined(separator: ", "))
                        Text(folder.root.path).font(.footnote).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
        #if os(macOS)
        .frame(minWidth: 460, minHeight: 420)
        #endif
    }

    /// Picking another form reloads the pipeline with its stages.
    private var formSelection: Binding<String> {
        Binding(get: { model.formID }, set: { id in Task { await model.selectForm(id) } })
    }
}
