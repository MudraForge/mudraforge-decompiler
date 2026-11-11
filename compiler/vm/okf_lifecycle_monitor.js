/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MUDRAFORGE USVM - OKF LIFECYCLE MONITOR & TEMPORAL EPOCH SPLITTER
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Detects whether a statutory AST (compiled from circulars, Finance Bills, or
 * user rule definitions) represents:
 *   1. OKF_VERIFIED_ACTIVE            - Matches active verified OKF baseline.
 *   2. OKF_AMENDMENT_STALE_DETECTED   - Contradicts active OKF baseline metrics
 *                                       (rate, threshold, sunset, penalty).
 *                                       Proposes non-destructive Temporal Epoch Split.
 *   3. OKF_UNREGISTERED_NEW           - Novel provision absent from OKF corpus.
 *
 * Guarantees:
 *   - Zero external dependencies.
 *   - Non-destructive temporal epoch splitting (closes prior validTo instead of
 *     overwriting history so retrospective audits prior to cutover remain intact).
 *   - Strict regex escaping on all dynamic pattern lookups (Rule 4).
 *   - Immutable fiduciary invariant: isAuthorizedToDisburse = false.
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import {
    OKF_LIFECYCLE_STATUSES,
    validateDeepOfficialCitations
} from './schemas/statutory_ast_schema.js';

const __filename = typeof import.meta?.url === 'string' && import.meta.url.startsWith('file:')
    ? fileURLToPath(import.meta.url)
    : '';
const __dirname = __filename ? path.dirname(__filename) : '';

let CACHED_DEFAULT_REGISTRY = null;

/**
 * Rule 4 compliant regex character escaping.
 */
export function escapeRegexLiteral(raw) {
    return String(raw ?? '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Computes YYYY-MM-DD immediately preceding the given ISO date in UTC.
 */
export function subtractOneDayIso(isoDate) {
    if (!isoDate || typeof isoDate !== 'string') return null;
    const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(isoDate.trim());
    if (!match) return null;
    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    const utcMs = Date.UTC(year, month - 1, day) - 86400000;
    const d = new Date(utcMs);
    return d.toISOString().slice(0, 10);
}

/**
 * Normalizes a section code or identifier into a comparable uppercase token.
 */
export function canonicalizeSectionToken(raw) {
    if (!raw || typeof raw !== 'string') return '';
    return raw
        .trim()
        .toUpperCase()
        .replace(/^(SECTION|SEC|REGULATION|REG|RULE|ART|ARTICLE)\s+/i, '')
        .replace(/[^A-Z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '');
}

/**
 * Extracts normalized comparable statutory metrics from either an AST or an OKF section.json.
 */
export function extractRuleMetrics(pkg) {
    if (!pkg || typeof pkg !== 'object') return {};

    const ast = pkg.synthesizedAst || pkg;
    const comp = ast.computation || pkg.computation || {};
    const blocks = pkg.compositionalBuildingBlocks || ast.compositionalBuildingBlocks || {};
    const kt = pkg.keyStatutoryThresholds;
    const eff = pkg.effectivePeriod || ast.effectivePeriod || {};
    const symbolLegend = pkg.symbolLegend || ast.symbolLegend || blocks.symbolLegend || null;

    let thresholdPaisa = comp.thresholdPaisa !== undefined && comp.thresholdPaisa !== null
        ? String(comp.thresholdPaisa)
        : null;

    if (thresholdPaisa === null && kt && !Array.isArray(kt) && typeof kt === 'object') {
        if (kt.annualPurchaseThresholdInr !== undefined && kt.annualPurchaseThresholdInr !== null) {
            thresholdPaisa = String(Math.round(Number(kt.annualPurchaseThresholdInr) * 100));
        } else if (kt.thresholdPaisa !== undefined && kt.thresholdPaisa !== null) {
            thresholdPaisa = String(kt.thresholdPaisa);
        }
    }

    if (thresholdPaisa === null && symbolLegend && typeof symbolLegend === 'object') {
        for (const sym of Object.values(symbolLegend)) {
            if (!sym || typeof sym !== 'object') continue;
            const role = sym.role || '';
            if (
                role === 'ROLE_TRANSACTION_EXCESS_THRESHOLD' ||
                role === 'ROLE_TRANSACTION_CLIFF_THRESHOLD' ||
                role === 'ROLE_EXCESS_THRESHOLD'
            ) {
                if (sym.targetThresholdPaisa !== undefined && sym.targetThresholdPaisa !== null) {
                    thresholdPaisa = String(sym.targetThresholdPaisa);
                    break;
                }
                if (sym.targetThreshold !== undefined && sym.targetThreshold !== null) {
                    thresholdPaisa = String(Math.round(Number(sym.targetThreshold) * 100));
                    break;
                }
            }
        }
    }

    let standardRateBps = comp.standardRateBps !== undefined && comp.standardRateBps !== null
        ? Number(comp.standardRateBps)
        : null;

    if (standardRateBps === null && kt && !Array.isArray(kt) && typeof kt === 'object') {
        if (kt.standardDeductionRatePercent !== undefined && kt.standardDeductionRatePercent !== null) {
            standardRateBps = Math.round(Number(kt.standardDeductionRatePercent) * 100);
        }
    }

    if (standardRateBps === null && symbolLegend && typeof symbolLegend === 'object') {
        for (const sym of Object.values(symbolLegend)) {
            if (!sym || typeof sym !== 'object') continue;
            if (sym.role === 'ROLE_STANDARD_STATUTORY_RATE' && sym.targetThreshold !== undefined && sym.targetThreshold !== null) {
                standardRateBps = Math.round(Number(sym.targetThreshold) * 100);
                break;
            }
        }
    }

    let penaltyOverrideRateBps = comp.penaltyOverrideRateBps !== undefined && comp.penaltyOverrideRateBps !== null
        ? Number(comp.penaltyOverrideRateBps)
        : (comp.nonPanOverrideRateBps !== undefined && comp.nonPanOverrideRateBps !== null
            ? Number(comp.nonPanOverrideRateBps)
            : null);

    if (penaltyOverrideRateBps === null && kt && !Array.isArray(kt) && typeof kt === 'object') {
        if (kt.inoperativePanPenaltyRatePercent !== undefined && kt.inoperativePanPenaltyRatePercent !== null) {
            penaltyOverrideRateBps = Math.round(Number(kt.inoperativePanPenaltyRatePercent) * 100);
        }
    }

    if (penaltyOverrideRateBps === null && Array.isArray(ast.guards)) {
        const overrideGuard = ast.guards.find((g) => g?.onMatch?.action === 'OVERRIDE_RATE' && g?.onMatch?.overrideRateBps !== undefined);
        if (overrideGuard) {
            penaltyOverrideRateBps = Number(overrideGuard.onMatch.overrideRateBps);
        }
    }

    const sunsetDate = pkg.sunsetDate
        || ast.sunsetDate
        || blocks?.temporalCrystallization?.sunsetDate
        || eff.sunsetDate
        || null;

    const validFrom = eff.validFrom || eff.effectiveFrom || pkg.validFrom || null;
    const validTo = eff.validTo ?? eff.effectiveTo ?? pkg.validTo ?? null;

    return {
        thresholdPaisa,
        standardRateBps,
        penaltyOverrideRateBps,
        sunsetDate,
        validFrom,
        validTo,
        archetype: comp.archetype || ast.rootCondition?.nodeType || null,
        triggerPolicyType: ast.triggerPolicy?.type || null
    };
}

function walkSectionJsonFiles(dirPath, acc = []) {
    if (!dirPath || !fs.existsSync(dirPath)) return acc;
    let entries = [];
    try {
        entries = fs.readdirSync(dirPath, { withFileTypes: true });
    } catch {
        return acc;
    }
    for (const entry of entries) {
        const full = path.join(dirPath, entry.name);
        if (entry.isDirectory()) {
            if (entry.name === 'node_modules' || entry.name === '.git' || entry.name === 'circulars' || entry.name === 'forms') {
                continue;
            }
            walkSectionJsonFiles(full, acc);
        } else if (entry.isFile() && entry.name === 'section.json') {
            acc.push(full);
        }
    }
    return acc;
}

function mergeSnapshots(existing, incoming) {
    const existingCitCount = Array.isArray(existing.officialCitations) ? existing.officialCitations.length : 0;
    const incomingCitCount = Array.isArray(incoming.officialCitations) ? incoming.officialCitations.length : 0;

    return {
        ...existing,
        ...incoming,
        sectionId: existing.sectionId || incoming.sectionId,
        aliases: Array.from(new Set([...(existing.aliases || []), ...(incoming.aliases || [])])),
        officialCitations: incomingCitCount >= existingCitCount
            ? (incoming.officialCitations || existing.officialCitations || [])
            : (existing.officialCitations || []),
        computation: {
            ...(existing.computation || {}),
            ...(incoming.computation || {})
        },
        keyStatutoryThresholds: existing.keyStatutoryThresholds && !Array.isArray(existing.keyStatutoryThresholds)
            ? existing.keyStatutoryThresholds
            : (incoming.keyStatutoryThresholds || existing.keyStatutoryThresholds),
        guards: (Array.isArray(incoming.guards) && incoming.guards.length > 0)
            ? incoming.guards
            : (existing.guards || []),
        effectivePeriod: {
            ...(existing.effectivePeriod || {}),
            ...(incoming.effectivePeriod || {})
        }
    };
}

/**
 * Loads OKF section packages and registered USVM baseline rules into an indexed registry.
 */
export function loadOkfRegistrySnapshots(options = {}) {
    const hasCustomPaths = Boolean(options.okfRootDir || options.rulesDir || options.customSnapshots);
    const useCache = options.useCache !== false && !hasCustomPaths;

    if (useCache && CACHED_DEFAULT_REGISTRY) {
        return CACHED_DEFAULT_REGISTRY;
    }

    const bySectionId = new Map();
    const bySectionCode = new Map();
    const byAlias = new Map();
    const allSnapshots = [];

    const indexSnapshot = (rawSnap, sourcePath = null) => {
        if (!rawSnap || typeof rawSnap !== 'object' || !rawSnap.sectionId) return;
        const idUpper = String(rawSnap.sectionId).trim().toUpperCase();
        const enriched = {
            ...rawSnap,
            _sourcePath: sourcePath || rawSnap._sourcePath || null,
            _metrics: extractRuleMetrics(rawSnap)
        };

        const existing = bySectionId.get(idUpper);
        const merged = existing ? mergeSnapshots(existing, enriched) : enriched;
        merged._metrics = extractRuleMetrics(merged);
        bySectionId.set(idUpper, merged);

        if (!existing) {
            allSnapshots.push(merged);
        } else {
            const idx = allSnapshots.indexOf(existing);
            if (idx >= 0) allSnapshots[idx] = merged;
        }

        const codeCandidates = [
            merged.sectionCode,
            canonicalizeSectionToken(merged.statutorySection),
            canonicalizeSectionToken(merged.sectionId)
        ].filter(Boolean);

        for (const c of codeCandidates) {
            const cUpper = String(c).trim().toUpperCase();
            if (!bySectionCode.has(cUpper)) {
                bySectionCode.set(cUpper, merged);
            }
        }

        if (Array.isArray(merged.aliases)) {
            for (const al of merged.aliases) {
                const tok = canonicalizeSectionToken(String(al));
                if (tok && tok.length >= 2 && !byAlias.has(tok)) {
                    byAlias.set(tok, merged);
                }
            }
        }
    };

    // 1. Auto-discover USVM rules JSON first (canonical numeric thresholds & rates)
    const defaultRulesDir = __dirname ? path.resolve(__dirname, 'rules') : null;
    const rulesDir = options.rulesDir || defaultRulesDir;
    if (rulesDir && fs.existsSync(rulesDir)) {
        let ruleFiles = [];
        try {
            ruleFiles = fs.readdirSync(rulesDir).filter((f) => f.endsWith('.json'));
        } catch {
            ruleFiles = [];
        }
        for (const rf of ruleFiles) {
            const fullPath = path.join(rulesDir, rf);
            try {
                const parsed = JSON.parse(fs.readFileSync(fullPath, 'utf8'));
                indexSnapshot(parsed, fullPath);
            } catch {
                // Skip malformed rule file silently during indexing
            }
        }
    }

    // 2. Discover OKF corpus section.json files (enterprise_okf or packages/statutory/okf)
    const defaultOkfCandidates = [];
    if (options.okfRootDir) {
        defaultOkfCandidates.push(options.okfRootDir);
    } else if (__dirname) {
        defaultOkfCandidates.push(path.resolve(__dirname, '../../../enterprise_okf'));
        defaultOkfCandidates.push(path.resolve(__dirname, '../../../okf'));
        defaultOkfCandidates.push(path.resolve(__dirname, '../../okf'));
    }

    for (const okfDir of defaultOkfCandidates) {
        const files = walkSectionJsonFiles(okfDir, []);
        for (const filePath of files) {
            try {
                const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                indexSnapshot(parsed, filePath);
            } catch {
                // Skip malformed section.json silently during indexing
            }
        }
    }

    // 3. Index caller-supplied snapshots (e.g., in unit tests or Workers KV bundles)
    if (options.customSnapshots) {
        const list = Array.isArray(options.customSnapshots)
            ? options.customSnapshots
            : Object.values(options.customSnapshots);
        for (const snap of list) {
            indexSnapshot(snap, 'customSnapshot');
        }
    }

    const registry = {
        bySectionId,
        bySectionCode,
        byAlias,
        allSnapshots
    };

    if (useCache) {
        CACHED_DEFAULT_REGISTRY = registry;
    }

    return registry;
}

/**
 * Matches a candidate AST rule against indexed OKF snapshots.
 */
export function matchOkfSnapshot(candidateAst, registry) {
    if (!candidateAst || !registry) return null;

    if (candidateAst.sectionId && typeof candidateAst.sectionId === 'string') {
        const upperId = candidateAst.sectionId.trim().toUpperCase();
        if (registry.bySectionId.has(upperId)) {
            return registry.bySectionId.get(upperId);
        }
        const strippedId = canonicalizeSectionToken(upperId);
        if (strippedId && registry.bySectionCode.has(strippedId)) {
            return registry.bySectionCode.get(strippedId);
        }
    }

    const candidateTokens = [];
    if (candidateAst.sectionCode) candidateTokens.push(canonicalizeSectionToken(candidateAst.sectionCode));
    if (candidateAst.statutorySection) candidateTokens.push(canonicalizeSectionToken(candidateAst.statutorySection));
    if (Array.isArray(candidateAst.sourceLineage?.referencedSections)) {
        for (const refSec of candidateAst.sourceLineage.referencedSections) {
            candidateTokens.push(canonicalizeSectionToken(refSec));
        }
    }

    for (const tok of candidateTokens) {
        if (!tok) continue;
        if (registry.bySectionCode.has(tok)) {
            return registry.bySectionCode.get(tok);
        }
        if (registry.byAlias.has(tok)) {
            return registry.byAlias.get(tok);
        }
        // Safe exact escaped token lookup across aliases
        const escaped = escapeRegexLiteral(tok);
        const tokRe = new RegExp(`^${escaped}$`, 'i');
        for (const snap of registry.allSnapshots) {
            if (snap.sectionCode && tokRe.test(canonicalizeSectionToken(snap.sectionCode))) {
                return snap;
            }
        }
    }

    return null;
}

function safeCloneJson(obj) {
    return JSON.parse(JSON.stringify(obj || {}, (_, val) => (typeof val === 'bigint' ? val.toString() : val)));
}

/**
 * Proposes a non-destructive Temporal Epoch Split when an amendment or circular
 * alters an existing OKF provision as of `cutoverDate`. Historical transactions
 * prior to `cutoverDate` preserve the closed baseline epoch untouched.
 */
export function applyTemporalEpochSplit(existingOkfRule, amendmentSpec = {}, cutoverDate = null) {
    const resolvedCutover = cutoverDate
        || amendmentSpec.cutoverDate
        || amendmentSpec.effectiveFrom
        || amendmentSpec.validFrom
        || new Date().toISOString().slice(0, 10);

    const priorCloseDate = subtractOneDayIso(resolvedCutover) || resolvedCutover;
    const baseClone = safeCloneJson(existingOkfRule || {});
    const existingEff = baseClone.effectivePeriod || {};
    const existingValidTo = existingEff.validTo || existingEff.effectiveTo || null;

    const historicalValidTo = (!existingValidTo || existingValidTo >= resolvedCutover)
        ? priorCloseDate
        : existingValidTo;

    const historicalEpoch = {
        ...baseClone,
        effectivePeriod: {
            ...existingEff,
            validFrom: existingEff.validFrom || existingEff.effectiveFrom || '1970-01-01',
            effectiveFrom: existingEff.effectiveFrom || existingEff.validFrom || '1970-01-01',
            validTo: historicalValidTo,
            effectiveTo: historicalValidTo,
            temporalStatus: 'SUPERSEDED_HISTORICAL'
        },
        supersededByAmendmentAt: resolvedCutover
    };

    const newEpoch = safeCloneJson(existingOkfRule || {});
    newEpoch.computation = { ...(newEpoch.computation || {}) };
    if (amendmentSpec.thresholdPaisa !== undefined && amendmentSpec.thresholdPaisa !== null) {
        newEpoch.computation.thresholdPaisa = String(amendmentSpec.thresholdPaisa);
    }
    if (amendmentSpec.standardRateBps !== undefined && amendmentSpec.standardRateBps !== null) {
        newEpoch.computation.standardRateBps = Number(amendmentSpec.standardRateBps);
    }
    if (amendmentSpec.penaltyOverrideRateBps !== undefined && amendmentSpec.penaltyOverrideRateBps !== null) {
        newEpoch.computation.penaltyOverrideRateBps = Number(amendmentSpec.penaltyOverrideRateBps);
    }
    if (amendmentSpec.sunsetDate !== undefined) {
        newEpoch.sunsetDate = amendmentSpec.sunsetDate;
    }
    if (Array.isArray(amendmentSpec.officialCitations) && amendmentSpec.officialCitations.length > 0) {
        newEpoch.officialCitations = amendmentSpec.officialCitations;
    }

    newEpoch.effectivePeriod = {
        validFrom: resolvedCutover,
        effectiveFrom: resolvedCutover,
        validTo: amendmentSpec.validTo ?? null,
        effectiveTo: amendmentSpec.effectiveTo ?? null,
        temporalStatus: 'PENDING_CONTROLLERSHIP_RATIFICATION'
    };
    newEpoch.previousEpochId = `${existingOkfRule?.sectionId || 'OKF_RULE'}@${historicalValidTo}`;
    newEpoch.statutoryInvariants = {
        ...(newEpoch.statutoryInvariants || {}),
        isAuthorizedToDisburse: false
    };

    const fieldDiffs = Array.isArray(amendmentSpec.fieldDiffs) ? amendmentSpec.fieldDiffs : [];
    const digestPayload = JSON.stringify({
        sectionId: existingOkfRule?.sectionId || null,
        cutoverDate: resolvedCutover,
        historicalValidTo,
        fieldDiffs
    });
    const amendmentDigest = crypto.createHash('sha256').update(digestPayload).digest('hex');

    return {
        status: 'TEMPORAL_EPOCH_SPLIT_PROPOSED',
        cutoverDate: resolvedCutover,
        historicalValidTo,
        historicalEpoch,
        pendingEpoch: newEpoch,
        fieldDiffs,
        amendmentDigest,
        isAuthorizedToDisburse: false
    };
}

/**
 * Classifies a candidate AST's relationship against the verified OKF registry.
 */
export function classifyOkfRuleLifecycle(candidateAst, options = {}) {
    if (!candidateAst || typeof candidateAst !== 'object') {
        throw new Error('[OKF_LIFECYCLE_MONITOR] candidateAst must be a non-null object');
    }

    const registry = options.okfRegistry || loadOkfRegistrySnapshots({
        okfRootDir: options.okfRootDir,
        rulesDir: options.rulesDir,
        customSnapshots: options.customSnapshots
    });

    const citValidation = validateDeepOfficialCitations(candidateAst.officialCitations);
    const matched = matchOkfSnapshot(candidateAst, registry);

    if (!matched) {
        return {
            lifecycleStatus: OKF_LIFECYCLE_STATUSES[2], // OKF_UNREGISTERED_NEW
            matchedSectionId: null,
            matchedSnapshot: null,
            fieldDiffs: [],
            epochSplitProposed: null,
            hasValidCitations: citValidation.valid,
            citationIssue: citValidation.valid ? null : citValidation.reason,
            code: 'ERR_UNVERIFIED_PRODUCTION_RULE',
            canRunDevelopment: true,
            canRunProduction: Boolean(candidateAst.isVerifiedInternalPolicy),
            isAuthorizedToDisburse: false
        };
    }

    const baseMetrics = matched._metrics || extractRuleMetrics(matched);
    const candMetrics = extractRuleMetrics(candidateAst);
    const fieldDiffs = [];

    if (
        baseMetrics.thresholdPaisa !== null &&
        candMetrics.thresholdPaisa !== null &&
        String(baseMetrics.thresholdPaisa) !== String(candMetrics.thresholdPaisa)
    ) {
        fieldDiffs.push({
            field: 'computation.thresholdPaisa',
            registeredValue: String(baseMetrics.thresholdPaisa),
            candidateValue: String(candMetrics.thresholdPaisa)
        });
    }

    if (
        baseMetrics.standardRateBps !== null &&
        candMetrics.standardRateBps !== null &&
        Number(baseMetrics.standardRateBps) !== Number(candMetrics.standardRateBps)
    ) {
        fieldDiffs.push({
            field: 'computation.standardRateBps',
            registeredValue: Number(baseMetrics.standardRateBps),
            candidateValue: Number(candMetrics.standardRateBps)
        });
    }

    if (
        baseMetrics.penaltyOverrideRateBps !== null &&
        candMetrics.penaltyOverrideRateBps !== null &&
        Number(baseMetrics.penaltyOverrideRateBps) !== Number(candMetrics.penaltyOverrideRateBps)
    ) {
        fieldDiffs.push({
            field: 'computation.penaltyOverrideRateBps',
            registeredValue: Number(baseMetrics.penaltyOverrideRateBps),
            candidateValue: Number(candMetrics.penaltyOverrideRateBps)
        });
    }

    if (
        candMetrics.sunsetDate !== null &&
        String(baseMetrics.sunsetDate || '') !== String(candMetrics.sunsetDate || '')
    ) {
        fieldDiffs.push({
            field: 'sunsetDate',
            registeredValue: baseMetrics.sunsetDate || null,
            candidateValue: candMetrics.sunsetDate
        });
    }

    const effectiveCitationsValid = citValidation.valid || validateDeepOfficialCitations(matched.officialCitations).valid;

    if (fieldDiffs.length > 0 || options.forceStale) {
        const cutoverDate = options.cutoverDate
            || candidateAst.transitionalCutoverDate
            || candidateAst.effectivePeriod?.validFrom
            || candidateAst.effectivePeriod?.effectiveFrom
            || new Date().toISOString().slice(0, 10);

        const epochSplitProposed = applyTemporalEpochSplit(matched, {
            thresholdPaisa: candMetrics.thresholdPaisa,
            standardRateBps: candMetrics.standardRateBps,
            penaltyOverrideRateBps: candMetrics.penaltyOverrideRateBps,
            sunsetDate: candMetrics.sunsetDate,
            officialCitations: candidateAst.officialCitations,
            fieldDiffs
        }, cutoverDate);

        return {
            lifecycleStatus: OKF_LIFECYCLE_STATUSES[1], // OKF_AMENDMENT_STALE_DETECTED
            matchedSectionId: matched.sectionId,
            matchedSnapshot: matched,
            fieldDiffs,
            cutoverDate,
            epochSplitProposed,
            hasValidCitations: effectiveCitationsValid,
            code: 'HOLD_STALE_OKF_PENDING_AMENDMENT',
            canRunDevelopment: true,
            canRunProduction: false,
            isAuthorizedToDisburse: false
        };
    }

    return {
        lifecycleStatus: OKF_LIFECYCLE_STATUSES[0], // OKF_VERIFIED_ACTIVE
        matchedSectionId: matched.sectionId,
        matchedSnapshot: matched,
        fieldDiffs: [],
        cutoverDate: null,
        epochSplitProposed: null,
        hasValidCitations: effectiveCitationsValid,
        code: 'OKF_SYNCED',
        canRunDevelopment: true,
        canRunProduction: effectiveCitationsValid || Boolean(candidateAst.isVerifiedInternalPolicy),
        isAuthorizedToDisburse: false
    };
}
