import CoreAI
import Darwin
import Foundation
#if os(iOS)
import UIKit
#endif

/// The device, memory, thermal and power facts a self-test run records.
enum DeviceProbe {
    static func sysctl(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "" }
        var bytes = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return "" }
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }

    /// `iPhone19,2`, `Mac16,9`: the hardware model.
    static var machine: String {
        #if os(macOS)
        sysctl("hw.model")
        #else
        sysctl("hw.machine")
        #endif
    }

    static var osBuild: String { sysctl("kern.osversion") }

    /// The process's physical footprint (what jetsam counts) and its peak, in MB.
    static func footprint() -> (now: Double, peak: Double) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return (-1, -1) }
        return (Double(info.phys_footprint) / 1e6, Double(info.ledger_phys_footprint_peak) / 1e6)
    }

    /// What the app may still allocate before jetsam, in MB (iPhone only).
    static func availableMB() -> Double? {
        #if os(iOS)
        Double(os_proc_available_memory()) / 1e6
        #else
        nil
        #endif
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    struct Power: Sendable {
        let battery: String
        let level: Double?
        let lowPowerMode: Bool

        var json: [String: Any] { ["battery": battery, "level": level ?? NSNull(), "low_power_mode": lowPowerMode] }
    }

    @MainActor static func readPower() -> Power {
        #if os(iOS)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let state: String = switch UIDevice.current.batteryState {
        case .charging: "charging"
        case .full: "full"
        case .unplugged: "unplugged"
        default: "unknown"
        }
        let level = UIDevice.current.batteryLevel
        return Power(battery: state, level: level >= 0 ? Double(level) : nil,
                     lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
        #else
        return Power(battery: "mac", level: nil, lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
        #endif
    }

    static func power() async -> [String: Any] { await readPower().json }

    /// When this process started (from the kernel), for the time from launch to the first transcript.
    static var processStart: Date? {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        guard Darwin.sysctl(&mib, 4, &info, &size, nil, 0) == 0 else { return nil }
        let start = info.kp_proc.p_un.__p_starttime
        return Date(timeIntervalSince1970: Double(start.tv_sec) + Double(start.tv_usec) / 1e6)
    }

    static func record() async -> [String: Any] {
        let memory = footprint()
        return ["machine": machine, "arch": AIModel.deviceArchitectureName,
                "os": ProcessInfo.processInfo.operatingSystemVersionString, "os_build": osBuild,
                "physical_memory_gb": (Double(ProcessInfo.processInfo.physicalMemory) / 1e9 * 100).rounded() / 100,
                "thermal": thermal(), "power": await power(), "footprint_mb": memory.now,
                "available_mb": availableMB() ?? NSNull()]
    }
}

/// Peak footprint, least available memory and every thermal-state change while a run goes (sampled every 50 ms).
final class RunSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak = 0.0
    private var least: Double?
    private var changes: [[String: Any]] = []
    private var running = true
    private let start = Date()
    private var observer: NSObjectProtocol?

    func begin() {
        observer = NotificationCenter.default.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification,
                                                          object: nil, queue: nil) { [weak self] _ in
            guard let self else { return }
            let state = DeviceProbe.thermal()
            self.lock.lock()
            self.changes.append(["t_s": Date().timeIntervalSince(self.start), "state": state])
            self.lock.unlock()
        }
        Thread.detachNewThread { [weak self] in
            while let self, self.sample() { Thread.sleep(forTimeInterval: 0.05) }
        }
    }

    private func sample() -> Bool {
        let now = DeviceProbe.footprint().now
        let available = DeviceProbe.availableMB()
        lock.lock()
        defer { lock.unlock() }
        peak = max(peak, now)
        if let available { least = min(least ?? available, available) }
        return running
    }

    func end() -> [String: Any] {
        _ = sample()
        lock.lock()
        defer { lock.unlock() }
        running = false
        if let observer { NotificationCenter.default.removeObserver(observer) }
        return ["footprint_peak_mb": peak, "available_least_mb": least ?? NSNull(), "thermal_changes": changes,
                "thermal_end": DeviceProbe.thermal()]
    }
}
