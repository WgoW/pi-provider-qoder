import { existsSync, readFileSync } from "node:fs";
import { dirname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8")) as {
  type?: string;
  main?: string;
  exports?: Record<string, unknown> | string;
  files: string[];
  pi: { extensions: string[] };
  scripts: Record<string, string>;
};

type ExportTarget = string | { [condition: string]: ExportTarget } | Array<ExportTarget>;

/** Every string leaf of an `exports` entry (conditions nest arbitrarily). */
function exportTargets(value: ExportTarget): string[] {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(exportTargets);
  return Object.values(value).flatMap(exportTargets);
}

describe("published Pi extension entry", () => {
  it("points pi.extensions at the bundled dist file, not src", () => {
    // `files` only ships dist + README. Pointing at src/index.ts (as 0.3.0 on
    // npm did) means Pi lists the package as installed but never loads it
    // (issues #18 and #21). The build script emits dist/index.js.
    expect(pkg.files).toContain("dist");
    expect(pkg.files).not.toContain("src");
    expect(pkg.pi.extensions).toEqual(["./dist/index.js"]);
  });

  it("keeps every pi.extensions path inside the published files set", () => {
    for (const entry of pkg.pi.extensions) {
      const normalized = normalize(entry).replaceAll("\\", "/");
      const published = pkg.files.some((file) => {
        const prefix = file.replace(/\/$/, "");
        return normalized === prefix || normalized.startsWith(`${prefix}/`);
      });
      expect(published, entry).toBe(true);
    }
  });

  it("resolves a bare specifier for both import and require consumers", () => {
    // 0.6.1 shipped with only `type: module` and no `main`/`exports`, so a bare
    // specifier resolved to a non-existent <pkg>/index.js:
    //   import -> ERR_MODULE_NOT_FOUND, require -> MODULE_NOT_FOUND
    // and `moduleResolution: node16` consumers failed with TS2307 before that.
    expect(pkg.main).toBe("./dist/index.js");
    expect(typeof pkg.exports).toBe("object");
    const dot = (pkg.exports as Record<string, ExportTarget>)["."];
    expect(dot).toBeDefined();
    expect((dot as Record<string, string>).import).toBe("./dist/index.js");
    expect((dot as Record<string, string>).require).toBe("./dist/index.js");
  });

  it("never points a published export at a file the build does not emit", () => {
    // The reported suggestion was `require: "./dist/index.cjs"`, but the bundle
    // is ESM-only: @earendil-works/pi-ai and pi-coding-agent have no CJS entry,
    // so a .cjs build cannot exist. Claiming one in `exports` would swap a
    // clear MODULE_NOT_FOUND for a confusing one. Assert every target exists.
    const exportsField = pkg.exports as Record<string, ExportTarget>;
    const targets = Object.entries(exportsField).flatMap(([key, value]) =>
      key === "./package.json" ? [] : exportTargets(value),
    );
    expect(targets.length).toBeGreaterThan(0);
    for (const target of targets) {
      expect(existsSync(join(root, target)), target).toBe(true);
    }
  });

  it("declares a type entry that the build emits", () => {
    // Without `types`, `moduleResolution: node16` consumers fail with TS7016
    // (implicit any) or TS2307 even once the JS specifier resolves.
    const dot = (pkg.exports as Record<string, ExportTarget>)["."] as Record<string, string>;
    expect(dot.types).toBe("./dist/index.d.ts");
    expect(existsSync(join(root, dot.types))).toBe(true);
  });

  it("builds the bundle and the declarations together", () => {
    // `prepare` runs on install for git consumers and `prepublishOnly` on
    // publish, so the single `build` entry is what both invoke.
    expect(pkg.scripts.build).toContain("build:bundle");
    expect(pkg.scripts.build).toContain("build:types");
  });
});
