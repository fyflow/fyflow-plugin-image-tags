// Which build the Plugin picks, on made-up machines: each file a fake, so nothing reads this machine's driver.
import { describe, expect, it } from "vitest";
import {
  type Machine,
  nvidiaDriverOfDll,
  nvidiaDriverOfProc,
  pickBuild,
  WINDOWS_MIN_DRIVER,
} from "../pick-build.ts";

const SYSTEM_ROOT = "C:\\Windows";
const NVCUDA = "C:\\Windows\\System32\\nvcuda.dll";
const PROC_VERSION = "/proc/driver/nvidia/version";

/** A machine holding `files` (text or bytes, by path), and nothing else. */
const machineOf = (
  files: Record<string, string | Uint8Array>,
  systemRoot: string | undefined = SYSTEM_ROOT,
): Machine => ({
  systemRoot,
  exists: (path) => path in files,
  read: (path) => {
    const content = files[path];
    if (content === undefined) throw new Error(`ENOENT: ${path}`);
    return typeof content === "string" ? new TextEncoder().encode(content) : content;
  },
});

/**
 * A DLL's bytes holding a `VS_FIXEDFILEINFO` of file version `a.b.c.d` after some padding: its signature, its
 * structure version, then the file version's two DWORDs, little-endian.
 */
function dllOf(a: number, b: number, c: number, d: number): Uint8Array {
  const bytes = new Uint8Array(4096);
  const view = new DataView(bytes.buffer);
  const at = 1234;
  view.setUint32(at, 0xfeef04bd, true);
  view.setUint32(at + 4, 0x00010000, true);
  view.setUint32(at + 8, (a << 16) | b, true);
  view.setUint32(at + 12, (c << 16) | d, true);
  return bytes;
}

const procOf = (version: string) =>
  `NVRM version: NVIDIA UNIX x86_64 Kernel Module  ${version}  Sat Aug 19 01:15:15 UTC 2023\n` +
  "GCC version:  gcc version 12.2.0 (Debian 12.2.0-14)\n";

describe("the NVIDIA driver's version", () => {
  it("is read from nvcuda.dll's file version: the third part's last digit, then the fourth", () => {
    expect(nvidiaDriverOfDll(dllOf(32, 0, 15, 6094))).toEqual([560, 94]);
    expect(nvidiaDriverOfDll(dllOf(31, 0, 15, 2833))).toEqual([528, 33]);
    expect(nvidiaDriverOfDll(dllOf(31, 0, 15, 2802))).toEqual([528, 2]);
  });

  it("is unknown in a DLL without a VS_FIXEDFILEINFO", () => {
    expect(nvidiaDriverOfDll(new Uint8Array(64))).toBeUndefined();
    expect(nvidiaDriverOfDll(new Uint8Array([0xbd, 0x04, 0xef, 0xfe]))).toBeUndefined();
  });

  it("is read from Linux's version file, the proprietary module's or the open one's", () => {
    expect(nvidiaDriverOfProc(procOf("535.104.05"))).toEqual([535, 104, 5]);
    expect(
      nvidiaDriverOfProc(
        "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  550.54.14  Release Build\n",
      ),
    ).toEqual([550, 54, 14]);
    expect(nvidiaDriverOfProc(procOf("470.82"))).toEqual([470, 82]);
    expect(nvidiaDriverOfProc("something else")).toBeUndefined();
  });
});

describe("the build the Plugin picks", () => {
  describe("on Windows", () => {
    it("is cuda with nvcuda.dll of a driver at least 580.88, CUDA 13's", () => {
      expect(WINDOWS_MIN_DRIVER).toEqual([580, 88]);
      expect(pickBuild("win32", machineOf({ [NVCUDA]: dllOf(32, 0, 16, 1060) }))).toBe("cuda");
      expect(pickBuild("win32", machineOf({ [NVCUDA]: dllOf(32, 0, 15, 8088) }))).toBe("cuda");
    });

    it("is directml with an older driver, or none", () => {
      expect(pickBuild("win32", machineOf({ [NVCUDA]: dllOf(32, 0, 15, 8087) }))).toBe("directml");
      expect(pickBuild("win32", machineOf({ [NVCUDA]: dllOf(32, 0, 15, 6094) }))).toBe("directml");
      expect(pickBuild("win32", machineOf({ [NVCUDA]: dllOf(30, 0, 14, 7141) }))).toBe("directml");
      expect(pickBuild("win32", machineOf({}))).toBe("directml");
    });

    it("is cuda with an nvcuda.dll whose version can't be read", () => {
      expect(pickBuild("win32", machineOf({ [NVCUDA]: new Uint8Array(64) }))).toBe("cuda");
    });

    it("looks in the system folder Windows names, else C:\\Windows", () => {
      const elsewhere = "D:\\Win\\System32\\nvcuda.dll";
      expect(pickBuild("win32", machineOf({ [elsewhere]: dllOf(32, 0, 16, 1060) }, "D:\\Win"))).toBe("cuda");
      expect(pickBuild("win32", machineOf({ [NVCUDA]: dllOf(32, 0, 16, 1060) }, undefined))).toBe("cuda");
    });
  });

  describe("on Linux", () => {
    it("is cuda with a driver at least 580.65.06, CUDA 13's", () => {
      expect(pickBuild("linux", machineOf({ [PROC_VERSION]: procOf("580.65.06") }))).toBe("cuda");
      expect(pickBuild("linux", machineOf({ [PROC_VERSION]: procOf("610.43") }))).toBe("cuda");
      expect(pickBuild("linux", machineOf({ [PROC_VERSION]: procOf("580.66") }))).toBe("cuda");
    });

    it("is cpu with an older driver", () => {
      expect(pickBuild("linux", machineOf({ [PROC_VERSION]: procOf("580.65.05") }))).toBe("cpu");
      expect(pickBuild("linux", machineOf({ [PROC_VERSION]: procOf("535.104.05") }))).toBe("cpu");
      expect(pickBuild("linux", machineOf({ [PROC_VERSION]: procOf("470.82.01") }))).toBe("cpu");
    });

    it("is cuda with a version file whose version can't be read", () => {
      expect(pickBuild("linux", machineOf({ [PROC_VERSION]: "NVRM version: something new" }))).toBe("cuda");
    });

    it("is cuda with no version file but libcuda.so.1 in a usual place, as under WSL", () => {
      for (const path of [
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        "/usr/lib/wsl/lib/libcuda.so.1",
      ])
        expect(pickBuild("linux", machineOf({ [path]: "" })), path).toBe("cuda");
    });

    it("is cpu with no driver", () => {
      expect(pickBuild("linux", machineOf({}))).toBe("cpu");
    });
  });

  it("is cpu on any other platform", () => {
    expect(pickBuild("darwin", machineOf({ [PROC_VERSION]: procOf("535.104.05") }))).toBe("cpu");
  });

  it("is cpu, never an error, when the machine can't be read", () => {
    const broken: Machine = {
      systemRoot: SYSTEM_ROOT,
      exists: () => true,
      read: () => {
        throw new Error("EACCES");
      },
    };
    expect(pickBuild("win32", broken)).toBe("cpu");
    expect(pickBuild("linux", broken)).toBe("cpu");
    const throwing: Machine = {
      ...broken,
      exists: () => {
        throw new Error("EIO");
      },
    };
    expect(pickBuild("win32", throwing)).toBe("cpu");
  });
});
