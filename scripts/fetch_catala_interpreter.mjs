#!/usr/bin/env node
/**
 * Fetches and verifies the official Catala web interpreter bundle.
 *
 * The ~10 MB js_of_ocaml bundle is a generated artifact, so it is not
 * committed. Everything about the artifact (upstream URL, mirrors, byte size,
 * SHA-256, Catala version and js_of_ocaml version) lives in
 * tools/catala.lock.json. This script reads that manifest, downloads from the
 * first working URL, verifies the digest, and installs atomically.
 *
 * The `version=6.4.1` in the bundle header is js_of_ocaml's version, not the
 * Catala version. The embedded Catala_utils.Version is recorded separately as
 * `catala_version` in the manifest.
 *
 * Usage:
 *   node scripts/fetch_catala_interpreter.mjs            # install if needed
 *   node scripts/fetch_catala_interpreter.mjs --check    # verify only
 *   node scripts/fetch_catala_interpreter.mjs --check <path>
 *
 * Environment:
 *   CATALA_INTERPRETER_URL  override the primary URL (manifest mirrors are
 *                           still tried as fallbacks)
 *   CATALA_LOCKFILE         override the lockfile path
 *
 * Exit codes: 0 success, 1 download/integrity/manifest failure.
 */

import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_LOCKFILE = path.join(__dirname, '..', 'tools', 'catala.lock.json');
const DEFAULT_DESTINATION = path.join(__dirname, 'catala_interpreter.cjs');

function sha256(buffer) {
    return createHash('sha256').update(buffer).digest('hex');
}

function describe(pathname) {
    return path.relative(process.cwd(), pathname) || pathname;
}

async function readIfPresent(pathname) {
    try {
        return await fs.readFile(pathname);
    } catch (error) {
        if (error.code === 'ENOENT') return null;
        throw error;
    }
}

function manifestError(lockfile, message) {
    return new Error(`${describe(lockfile)}: ${message}`);
}

async function loadLockfile(lockfile) {
    let raw;
    try {
        raw = await fs.readFile(lockfile, 'utf8');
    } catch (error) {
        if (error.code === 'ENOENT') {
            throw manifestError(lockfile, 'lockfile not found');
        }
        throw error;
    }

    let manifest;
    try {
        manifest = JSON.parse(raw);
    } catch (error) {
        throw manifestError(lockfile, `invalid JSON: ${error.message}`);
    }

    if (manifest === null || typeof manifest !== 'object' || Array.isArray(manifest)) {
        throw manifestError(lockfile, 'expected a JSON object');
    }
    if (manifest.schemaVersion !== 1) {
        throw manifestError(lockfile, `unsupported schemaVersion ${JSON.stringify(manifest.schemaVersion)}`);
    }
    if (typeof manifest.name !== 'string' || manifest.name.length === 0) {
        throw manifestError(lockfile, 'name must be a non-empty string');
    }
    if (typeof manifest.url !== 'string' || !/^https?:\/\//.test(manifest.url)) {
        throw manifestError(lockfile, 'url must be an http(s) URL');
    }
    if (manifest.mirrors !== undefined) {
        if (!Array.isArray(manifest.mirrors) || manifest.mirrors.some((url) => typeof url !== 'string' || !/^https?:\/\//.test(url))) {
            throw manifestError(lockfile, 'mirrors must be an array of http(s) URLs');
        }
    }
    if (typeof manifest.sha256 !== 'string' || !/^[0-9a-fA-F]{64}$/.test(manifest.sha256)) {
        throw manifestError(lockfile, 'sha256 must be a 64-character hex digest');
    }
    if (!Number.isSafeInteger(manifest.bytes) || manifest.bytes <= 0) {
        throw manifestError(lockfile, 'bytes must be a positive integer');
    }
    if (typeof manifest.catala_version !== 'string' || manifest.catala_version.length === 0) {
        throw manifestError(lockfile, 'catala_version must be a non-empty string');
    }
    if (typeof manifest.jsoo_version !== 'string' || manifest.jsoo_version.length === 0) {
        throw manifestError(lockfile, 'jsoo_version must be a non-empty string');
    }

    return {
        ...manifest,
        sha256: manifest.sha256.toLowerCase(),
        mirrors: manifest.mirrors ?? [],
    };
}

function candidateUrls(manifest) {
    const urls = process.env.CATALA_INTERPRETER_URL
        ? [process.env.CATALA_INTERPRETER_URL, ...manifest.mirrors]
        : [manifest.url, ...manifest.mirrors];
    return [...new Set(urls)];
}

function verifyBundle(buffer, manifest, lockfile, source) {
    if (buffer.length !== manifest.bytes) {
        throw new Error(
            `refusing ${manifest.name} from ${source}: expected ${manifest.bytes} bytes, got ${buffer.length}`,
        );
    }
    const actual = sha256(buffer);
    if (actual !== manifest.sha256) {
        throw new Error(
            `refusing ${manifest.name} from ${source}: SHA-256 mismatch\n` +
                `  expected ${manifest.sha256} (${describe(lockfile)})\n` +
                `  actual   ${actual}`,
        );
    }
}

function label(manifest) {
    return `Catala ${manifest.catala_version} (js_of_ocaml ${manifest.jsoo_version})`;
}

async function installAtomically(destination, bundle) {
    const temporary = `${destination}.${process.pid}.tmp`;
    await fs.mkdir(path.dirname(destination), { recursive: true });
    try {
        await fs.writeFile(temporary, bundle);
        try {
            await fs.rename(temporary, destination);
        } catch (error) {
            // Windows does not always replace an existing destination.
            if (error.code !== 'EEXIST' && error.code !== 'EPERM') throw error;
            await fs.rm(destination, { force: true });
            await fs.rename(temporary, destination);
        }
    } catch (error) {
        await fs.rm(temporary, { force: true });
        throw error;
    }
}

async function main(argv) {
    const checkOnly = argv.includes('--check');
    const targetArg = argv.find((arg) => !arg.startsWith('--'));
    const destination = targetArg
        ? path.resolve(targetArg)
        : DEFAULT_DESTINATION;
    const lockfile = path.resolve(process.env.CATALA_LOCKFILE ?? DEFAULT_LOCKFILE);

    const manifest = await loadLockfile(lockfile);

    if (checkOnly) {
        const existing = await readIfPresent(destination);
        if (existing === null) {
            throw new Error(
                `missing ${describe(destination)}; run: node scripts/fetch_catala_interpreter.mjs`,
            );
        }
        verifyBundle(existing, manifest, lockfile, describe(destination));
        process.stdout.write(
            `${label(manifest)} interpreter OK (sha256 ${manifest.sha256})\n`,
        );
        return;
    }

    const existing = await readIfPresent(destination);
    if (existing !== null && sha256(existing) === manifest.sha256 && existing.length === manifest.bytes) {
        process.stdout.write(
            `${label(manifest)} interpreter already present at ${describe(destination)}\n`,
        );
        return;
    }

    const failures = [];
    for (const url of candidateUrls(manifest)) {
        process.stdout.write(`Downloading ${manifest.name} from ${url}\n`);
        let bundle;
        try {
            const response = await fetch(url, { redirect: 'follow' });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            bundle = Buffer.from(await response.arrayBuffer());
            verifyBundle(bundle, manifest, lockfile, url);
        } catch (error) {
            failures.push(`${url}: ${error.message}`);
            continue;
        }

        await installAtomically(destination, bundle);
        process.stdout.write(
            `Installed ${describe(destination)} (${bundle.length} bytes, ${label(manifest)}, sha256 verified)\n`,
        );
        return;
    }

    throw new Error(
        `could not install ${manifest.name} from any configured URL:\n  ` +
            failures.join('\n  '),
    );
}

main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
});
