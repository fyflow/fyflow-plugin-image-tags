// Which build of the Plugin's Python environment this machine runs (`python.pickVariant` in `plugin.ts`): `cuda` with
// a new enough NVIDIA driver, else `directml` on Windows, else `cpu`. The SDK has no
// helpers for it (as of fyflow-plugin-sdk 0.11.0), so the checks
// are the Plugin's own. A driver that is there but broken still passes: the Worker falls back to the CPU.

import { Buffer } from "node:buffer";
import { existsSync, readFileSync } from "node:fs";
import { posix, win32 } from "node:path";
import process from "node:process";

export type Build = "cuda" | "directml" | "cpu";

/** What the checks read of a machine: whether a file is there, its bytes, and Windows' folder (`%SystemRoot%`). */
export interface Machine {
  exists(path: string): boolean;
  read(path: string): Uint8Array;
  readonly systemRoot: string | undefined;
}

const thisMachine: Machine = {
  exists: existsSync,
  read: (path) => readFileSync(path),
  get systemRoot() {
    return process.env.SystemRoot;
  },
};

/** A driver version as its parts, `[560, 94]` or `[535, 104, 5]`, compared part by part, a missing one as 0. */
export type DriverVersion = readonly number[];

/** The oldest Windows driver taken for `cuda` (`nvcuda.dll` 32.0.15.8088): CUDA 13's, the major the locked `onnxruntime-gpu` wheel is built for. */
export const WINDOWS_MIN_DRIVER: DriverVersion = [580, 88];
/** The oldest Linux driver taken for `cuda`: CUDA 13's, as on Windows. */
export const LINUX_MIN_DRIVER: DriverVersion = [580, 65, 6];

/** Linux's version file of the NVIDIA kernel module. */
const PROC_VERSION = "/proc/driver/nvidia/version";
/** Where distributions, and WSL, put the driver's CUDA library, for a machine without the version file. */
const LIBCUDA = [
  "/usr/lib/x86_64-linux-gnu",
  "/usr/lib64",
  "/usr/lib/aarch64-linux-gnu",
  "/usr/lib",
  "/usr/lib/wsl/lib",
].map((folder) => posix.join(folder, "libcuda.so.1"));

/** The build `platform` (as Node names it) runs on `machine`: never an error, `cpu` on any. */
export function pickBuild(platform: string, machine: Machine = thisMachine): Build {
  try {
    if (platform === "win32") return windowsHasCuda(machine) ? "cuda" : "directml";
    if (platform === "linux") return linuxHasCuda(machine) ? "cuda" : "cpu";
    return "cpu";
  } catch {
    return "cpu";
  }
}

function windowsHasCuda(machine: Machine): boolean {
  const dll = win32.join(machine.systemRoot ?? "C:\\Windows", "System32", "nvcuda.dll");
  if (!machine.exists(dll)) return false;
  const version = nvidiaDriverOfDll(machine.read(dll));
  // A driver whose version can't be read is taken as new enough.
  return version === undefined || atLeast(version, WINDOWS_MIN_DRIVER);
}

function linuxHasCuda(machine: Machine): boolean {
  if (machine.exists(PROC_VERSION)) {
    const version = nvidiaDriverOfProc(new TextDecoder().decode(machine.read(PROC_VERSION)));
    // A version file whose version can't be read is taken as new enough, as a DLL's is on Windows.
    return version === undefined || atLeast(version, LINUX_MIN_DRIVER);
  }
  return LIBCUDA.some((path) => machine.exists(path));
}

/** `VS_FIXEDFILEINFO`'s signature, `0xFEEF04BD`, as its little-endian bytes. */
const SIGNATURE = Buffer.from([0xbd, 0x04, 0xef, 0xfe]);

/**
 * The NVIDIA driver version of `nvcuda.dll`'s bytes, from its `VS_FIXEDFILEINFO`: after the signature `0xFEEF04BD`
 * and the structure version come the file version's high and low DWORDs, `a.b` and `c.d`. The driver is `c`'s last
 * digit then `d` in four digits: 32.0.15.6094 is 560.94. `undefined` without the signature.
 */
export function nvidiaDriverOfDll(bytes: Uint8Array): DriverVersion | undefined {
  const at = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).indexOf(SIGNATURE);
  if (at < 0 || at + 16 > bytes.byteLength) return undefined;
  const low = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(at + 12, true);
  const c = low >>> 16;
  const d = low & 0xffff;
  return [(c % 10) * 100 + Math.floor(d / 100), d % 100];
}

/**
 * The driver version in Linux's version file, the proprietary module's (`… Kernel Module  535.104.05  …`) or the open
 * one's (`… Open Kernel Module for x86_64  550.54.14  …`): its first dotted number. `undefined` without one.
 */
export function nvidiaDriverOfProc(text: string): DriverVersion | undefined {
  const match = /\s(\d+(?:\.\d+)+)\s/.exec(text);
  return match?.[1]?.split(".").map(Number);
}

function atLeast(version: DriverVersion, least: DriverVersion): boolean {
  for (let i = 0; i < Math.max(version.length, least.length); i++) {
    const [a = 0, b = 0] = [version[i], least[i]];
    if (a !== b) return a > b;
  }
  return true;
}
