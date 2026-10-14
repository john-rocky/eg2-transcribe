import Foundation

/// A NumPy .npy file (format 1.0 – 3.0, little-endian, C order): its shape and its payload, memory-mapped.
/// Enough for the self-test's reference arrays (float32 / float16 / int32 / int64).
struct NPY {
    let shape: [Int]
    let descr: String
    private let data: Data
    private let offset: Int

    init(_ url: URL) throws {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 10, data.prefix(6) == Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw TranscribeError.model("\(url.lastPathComponent) is not a .npy file")
        }
        let major = data[data.startIndex + 6]
        let headerLength: Int
        let headerStart: Int
        if major == 1 {
            headerLength = Int(data[data.startIndex + 8]) | Int(data[data.startIndex + 9]) << 8
            headerStart = 10
        } else {
            headerLength = (0..<4).reduce(0) { $0 | Int(data[data.startIndex + 8 + $1]) << (8 * $1) }
            headerStart = 12
        }
        let header = String(decoding: data[(data.startIndex + headerStart)..<(data.startIndex + headerStart + headerLength)],
                            as: UTF8.self)
        guard !header.contains("'fortran_order': True") else {
            throw TranscribeError.model("\(url.lastPathComponent): Fortran order is not supported")
        }
        guard let d = header.range(of: #"'descr':\s*'([^']+)'"#, options: .regularExpression),
              let s = header.range(of: #"'shape':\s*\(([^)]*)\)"#, options: .regularExpression) else {
            throw TranscribeError.model("\(url.lastPathComponent): unreadable header \(header)")
        }
        descr = String(header[d].split(separator: "'").last ?? "")
        let inner = header[s].drop { $0 != "(" }.dropFirst().prefix { $0 != ")" }
        shape = inner.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        offset = headerStart + headerLength
        self.data = data
        let width = ["<f4": 4, "<f2": 2, "<i4": 4, "<i8": 8][descr] ?? 0
        guard width > 0 else { throw TranscribeError.model("\(url.lastPathComponent): dtype \(descr) is not supported") }
        guard data.count - offset == shape.reduce(1, *) * width else {
            throw TranscribeError.model("\(url.lastPathComponent): payload \(data.count - offset) bytes for \(shape) \(descr)")
        }
    }

    var count: Int { shape.reduce(1, *) }

    /// Elements [from, from + count) as float32 (float16 / float32 files).
    func floats(from start: Int = 0, count n: Int? = nil) throws -> [Float] {
        let n = n ?? (count - start)
        precondition(start >= 0 && start + n <= count)
        return try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) throws -> [Float] in
            switch descr {
            case "<f4":
                return (0..<n).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 4 * (start + $0), as: UInt32.self))) }
            case "<f2":
                return (0..<n).map { Float(Float16(bitPattern: UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 2 * (start + $0), as: UInt16.self)))) }
            default:
                throw TranscribeError.model("expected a float array, got \(descr)")
            }
        }
    }

    /// Every element as Int (int32 / int64 files).
    func ints() throws -> [Int] {
        try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) throws -> [Int] in
            switch descr {
            case "<i4":
                return (0..<count).map { Int(Int32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 4 * $0, as: Int32.self))) }
            case "<i8":
                return (0..<count).map { Int(Int64(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 8 * $0, as: Int64.self))) }
            default:
                throw TranscribeError.model("expected an integer array, got \(descr)")
            }
        }
    }
}
