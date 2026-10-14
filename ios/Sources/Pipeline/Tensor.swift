import CoreAI
import Foundation

/// Host-side tensors: Swift arrays in, Core AI `NDArray`s of the dtype a function wants, and back. Every copy writes
/// through the arrays' own strides, with one flat pass when both sides are dense row-major.
enum Tensor {
    /// A new array of `shape` holding `values` (float32 on the host) as `type` (float16 or float32).
    static func make(_ values: UnsafeBufferPointer<Float>, shape: [Int], as type: NDArray.ScalarType) throws -> NDArray {
        guard values.count == shape.reduce(1, *), let source = values.baseAddress else {
            throw TranscribeError.contract("\(values.count) values for shape \(shape)")
        }
        let dense = denseStrides(shape)
        var array = NDArray(shape: shape, scalarType: type)
        switch type {
        case .float16:
            array.mutableView(as: Float16.self).withUnsafeMutablePointer { p, _, strides in
                transfer(source, dense, p, ints(strides), shape) { Float16($0) }
            }
        case .float32:
            array.mutableView(as: Float.self).withUnsafeMutablePointer { p, _, strides in
                transfer(source, dense, p, ints(strides), shape) { $0 }
            }
        default:
            throw TranscribeError.contract("unsupported input type \(type)")
        }
        return array
    }

    static func make(_ values: [Float], shape: [Int], as type: NDArray.ScalarType) throws -> NDArray {
        try values.withUnsafeBufferPointer { try make($0, shape: shape, as: type) }
    }

    /// The elements of a float16 or float32 array as float32, row-major.
    static func floats(_ array: NDArray) throws -> [Float] {
        guard array.interleaveLayout == nil else { throw TranscribeError.contract("interleaved output") }
        let shape = array.shape
        let dense = denseStrides(shape)
        var out = [Float](repeating: 0, count: shape.reduce(1, *))
        try out.withUnsafeMutableBufferPointer { dst in
            guard let d = dst.baseAddress else { return }
            switch array.scalarType {
            case .float16:
                array.view(as: Float16.self).withUnsafePointer { p, _, strides in
                    transfer(p, ints(strides), d, dense, shape) { Float($0) }
                }
            case .float32:
                array.view(as: Float.self).withUnsafePointer { p, _, strides in
                    transfer(p, ints(strides), d, dense, shape) { $0 }
                }
            default:
                throw TranscribeError.contract("expected float16 / float32, got \(array.scalarType)")
            }
        }
        return out
    }

    /// The elements of an int32 array, row-major.
    static func int32s(_ array: NDArray) throws -> [Int32] {
        guard array.interleaveLayout == nil else { throw TranscribeError.contract("interleaved output") }
        guard array.scalarType == .int32 else { throw TranscribeError.contract("expected int32, got \(array.scalarType)") }
        let shape = array.shape
        let dense = denseStrides(shape)
        var out = [Int32](repeating: 0, count: shape.reduce(1, *))
        out.withUnsafeMutableBufferPointer { dst in
            guard let d = dst.baseAddress else { return }
            array.view(as: Int32.self).withUnsafePointer { p, _, strides in
                transfer(p, ints(strides), d, dense, shape) { $0 }
            }
        }
        return out
    }

    /// A fresh copy of a float array as `type`: what one stage's output becomes before the next stage takes it (its
    /// own buffer, so the producing function can already run its next call).
    static func copy(_ array: NDArray, as type: NDArray.ScalarType) throws -> NDArray {
        guard array.interleaveLayout == nil else { throw TranscribeError.contract("interleaved output") }
        let shape = array.shape
        var out = NDArray(shape: shape, scalarType: type)
        switch (array.scalarType, type) {
        case (.float16, .float16):
            array.view(as: Float16.self).withUnsafePointer { p, _, s in
                out.mutableView(as: Float16.self).withUnsafeMutablePointer { q, _, t in transfer(p, ints(s), q, ints(t), shape) { $0 } }
            }
        case (.float32, .float16):
            array.view(as: Float.self).withUnsafePointer { p, _, s in
                out.mutableView(as: Float16.self).withUnsafeMutablePointer { q, _, t in transfer(p, ints(s), q, ints(t), shape) { Float16($0) } }
            }
        case (.float32, .float32):
            array.view(as: Float.self).withUnsafePointer { p, _, s in
                out.mutableView(as: Float.self).withUnsafeMutablePointer { q, _, t in transfer(p, ints(s), q, ints(t), shape) { $0 } }
            }
        case (.float16, .float32):
            array.view(as: Float16.self).withUnsafePointer { p, _, s in
                out.mutableView(as: Float.self).withUnsafeMutablePointer { q, _, t in transfer(p, ints(s), q, ints(t), shape) { Float($0) } }
            }
        default:
            throw TranscribeError.contract("cannot pass \(array.scalarType) as \(type)")
        }
        return out
    }

    /// Row-major strides (in elements) of a dense array.
    static func denseStrides(_ shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        var step = 1
        for d in stride(from: shape.count - 1, through: 0, by: -1) {
            strides[d] = step
            step *= shape[d]
        }
        return strides
    }

    /// Elementwise copy with conversion from one strided layout to another (strides in elements).
    @inline(__always)
    private static func transfer<S, D>(_ src: UnsafePointer<S>, _ srcStrides: [Int], _ dst: UnsafeMutablePointer<D>,
                                       _ dstStrides: [Int], _ shape: [Int], _ convert: (S) -> D) {
        let total = shape.reduce(1, *)
        guard total > 0 else { return }
        let dense = denseStrides(shape)
        if srcStrides == dense && dstStrides == dense {
            for i in 0..<total { dst[i] = convert(src[i]) }
            return
        }
        let rank = shape.count
        let inner = shape[rank - 1]
        let si = srcStrides[rank - 1], di = dstStrides[rank - 1]
        var index = [Int](repeating: 0, count: rank)
        for _ in 0..<(total / inner) {
            var so = 0, dOffset = 0
            for d in 0..<(rank - 1) {
                so += index[d] * srcStrides[d]
                dOffset += index[d] * dstStrides[d]
            }
            for j in 0..<inner { dst[dOffset + j * di] = convert(src[so + j * si]) }
            var d = rank - 2
            while d >= 0 {
                index[d] += 1
                if index[d] < shape[d] { break }
                index[d] = 0
                d -= 1
            }
        }
    }

    /// A view's shape or strides (in elements) as an array.
    private static func ints(_ span: Span<Int>) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(span.count)
        for i in 0..<span.count { out.append(span[i]) }
        return out
    }
}
