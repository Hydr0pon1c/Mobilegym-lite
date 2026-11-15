#!/usr/bin/env node
/*
 * Copyright 2026 OPPO
 *
 * Modified by OPPO for this release.
 * OPPO copyright applies only to the modifications.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Directly backfill Google vector tile responses into mobilegym-data/map/vector.
 *
 * This supplements, rather than replaces, the browser-driven snapshot:
 * - reads the existing vector manifest for the current Roadmap pb template
 * - calculates z/x/y tile ranges for named centers + radius
 * - fetches missing vector tiles
 * - merges them into manifest.json and updates map-sw.js VECTOR_CACHE_NAME
 * - regenerates the small bootstrap manifest used by service worker install
 *
 * With one or more run/log paths as arguments, only the real vector-tile 504s
 * observed in those browser logs are fetched. Without paths, defaults are
 * intentionally scoped to the bench hot spots requested during development:
 * Palace Museum and Wudaokou Shopping Center, 2km radius, z10-z18.
 *
 * Examples:
 *   node apps/Map/scripts/backfill_vector_tiles.mjs /path/to/eval/run
 *   MAP_VECTOR_BACKFILL_DRY_RUN=1 node apps/Map/scripts/backfill_vector_tiles.mjs /path/to/browser_logs
 *
 * Env:
 *   MAP_VECTOR_BACKFILL_CENTERS='[{"name":"...","lat":39.9,"lng":116.3}]'
 *   MAP_VECTOR_BACKFILL_RADIUS_M=2000
 *   MAP_VECTOR_BACKFILL_ZOOMS=10-18       # also supports "10,12,17-18"
 *   MAP_VECTOR_BACKFILL_CONCURRENCY=8
 *   MAP_VECTOR_BACKFILL_LIMIT=0           # non-zero for a capped probe
 *   MAP_VECTOR_BACKFILL_DRY_RUN=1
 *   MAP_VECTOR_BACKFILL_REMOVE_OBSERVED=1 # unlink exact observed URLs; keeps files recoverable
 *   MAP_VECTOR_BACKFILL_ALLOW_EPOCH_MISMATCH=1 # expert-only, normally unsafe
 */

import { createHash } from 'crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'fs';
import https from 'https';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import { HttpsProxyAgent } from 'https-proxy-agent';
import {
  updateServiceWorkerCacheName,
  writeVectorBootstrapManifest,
} from './bootstrap_manifest_utils.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const APP_DIR = resolve(__dirname, '..');
const ROOT = resolve(APP_DIR, '../..');
const VECTOR_DIR = resolve(ROOT, 'mobilegym-data/map/vector');
const FILES_DIR = resolve(VECTOR_DIR, 'files');
const MANIFEST_PATH = resolve(VECTOR_DIR, 'manifest.json');
const SW_PATH = resolve(APP_DIR, 'sw/map-sw.js');
const BOOTSTRAP_MANIFEST_PATH = resolve(APP_DIR, 'sw/vector/bootstrap-manifest.json');
const PLACES_PATH = resolve(APP_DIR, 'data/places.json');
const VECTOR_MANIFEST_SOURCE = 'mobilegym-data/map/vector/manifest.json';
const INPUT_PATHS = process.argv.slice(2).map((item) => resolve(item));
const PROXY_URL = process.env.HTTPS_PROXY || process.env.https_proxy || '';
const PROXY_AGENT = PROXY_URL ? new HttpsProxyAgent(PROXY_URL) : undefined;
const TILE_EPOCH_OVERRIDE = process.env.MAP_VECTOR_BACKFILL_TILE_EPOCH || '';
const TILE_EPOCH_BUILD_OVERRIDE = process.env.MAP_VECTOR_BACKFILL_TILE_EPOCH_BUILD || '';
let upstreamTileEpoch = null;
const MAP_TILE_504_RE = /http_error:\s*504\s+[—-]\s+(https:\/\/maps\.googleapis\.com\/maps\/vt\S+)/g;

const DEFAULT_CENTER_NAMES = ['故宫博物院', '五道口购物中心'];
const RADIUS_M = Number.parseFloat(process.env.MAP_VECTOR_BACKFILL_RADIUS_M || '2000');
const CONCURRENCY = Math.max(1, Number.parseInt(process.env.MAP_VECTOR_BACKFILL_CONCURRENCY || '8', 10));
const LIMIT = Math.max(0, Number.parseInt(process.env.MAP_VECTOR_BACKFILL_LIMIT || '0', 10));
const DRY_RUN = process.env.MAP_VECTOR_BACKFILL_DRY_RUN === '1';
const REMOVE_OBSERVED = process.env.MAP_VECTOR_BACKFILL_REMOVE_OBSERVED === '1';
const ALLOW_EPOCH_MISMATCH = process.env.MAP_VECTOR_BACKFILL_ALLOW_EPOCH_MISMATCH === '1';
const ZOOMS = parseZooms(process.env.MAP_VECTOR_BACKFILL_ZOOMS || '10-20');

function parseZooms(raw) {
  const out = new Set();
  for (const part of raw.split(',').map((p) => p.trim()).filter(Boolean)) {
    const range = part.match(/^(\d+)-(\d+)$/);
    if (range) {
      const a = Number.parseInt(range[1], 10);
      const b = Number.parseInt(range[2], 10);
      for (let z = Math.min(a, b); z <= Math.max(a, b); z++) out.add(z);
      continue;
    }
    const z = Number.parseInt(part, 10);
    if (Number.isFinite(z)) out.add(z);
  }
  return [...out].filter((z) => z >= 0 && z <= 22).sort((a, b) => a - b);
}

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf-8'));
}

function loadManifest() {
  if (!existsSync(MANIFEST_PATH)) {
    throw new Error(`Missing vector manifest: ${MANIFEST_PATH}`);
  }
  return readJson(MANIFEST_PATH);
}

function discoverBrowserLogs(paths) {
  const logs = [];
  const visit = (path) => {
    if (!existsSync(path)) throw new Error(`Input path does not exist: ${path}`);
    const stat = statSync(path);
    if (stat.isFile()) {
      if (path.endsWith('.log')) logs.push(path);
      return;
    }
    if (!stat.isDirectory()) return;
    const directBrowserLogs = resolve(path, 'browser_logs');
    if (!path.endsWith('/browser_logs') && existsSync(directBrowserLogs)) {
      visit(directBrowserLogs);
      return;
    }
    for (const name of readdirSync(path)) {
      const child = resolve(path, name);
      const childStat = statSync(child);
      if (childStat.isDirectory() && path.endsWith('/browser_logs')) visit(child);
      else if (name.endsWith('.log') && child.includes('/browser_logs/')) logs.push(child);
    }
  };
  for (const path of paths) visit(path);
  return [...new Set(logs)].sort();
}

function normalizeTilePb(value) {
  return value
    .replace(/(!2sm!3i)\d+/g, '$1E')
    .replace(/(!28i)\d+/g, '$1E')
    .replace(/!5m2!1e3!5f\d+/g, '!5m1!1e3');
}

function canonicalUrl(rawUrl) {
  const url = new URL(rawUrl);
  url.searchParams.delete('key');
  url.searchParams.delete('token');
  if (url.pathname === '/maps/vt' && url.searchParams.has('pb')) {
    url.pathname = `/maps/vt/pb=${normalizeTilePb(url.searchParams.get('pb') || '')}`;
    url.searchParams.delete('pb');
  } else if (url.pathname.startsWith('/maps/vt/pb=')) {
    url.pathname = normalizeTilePb(url.pathname);
  }
  const sorted = [...url.searchParams.entries()].sort(([a], [b]) => a.localeCompare(b));
  url.search = new URLSearchParams(sorted).toString();
  return url.toString();
}

function liveTileUrl(canonical) {
  if (!upstreamTileEpoch) throw new Error('upstream tile epoch has not been resolved');
  return canonical
    .replace(/(!2sm!3i)E/g, `$1${upstreamTileEpoch.build}`)
    .replace(/(!28i)E/g, `$1${upstreamTileEpoch.epoch}`);
}

function loadTrajectoryWorkItems(paths, existingEntries) {
  const logs = discoverBrowserLogs(paths);
  if (!logs.length) throw new Error('No browser_logs/*.log files found under input paths');
  const observed = [];
  for (const path of logs) {
    const text = readFileSync(path, 'utf-8');
    for (const match of text.matchAll(MAP_TILE_504_RE)) observed.push(match[1]);
  }

  const availableFiles = new Set(readdirSync(FILES_DIR));
  const existing = new Set(
    existingEntries
      .filter((entry) => entry?.url && entry?.file && availableFiles.has(entry.file))
      .map((entry) => canonicalUrl(entry.url)),
  );
  const observedUrls = new Set(observed.map(canonicalUrl));
  const unique = new Map();
  for (const rawUrl of observed) {
    const url = canonicalUrl(rawUrl);
    if (existing.has(url) || unique.has(url)) continue;
    const match = url.match(/!1i(\d+)!2i(\d+)!3i(\d+)/);
    if (!match) continue;
    unique.set(url, {
      center: 'trajectory-cache-miss',
      z: Number.parseInt(match[1], 10),
      x: Number.parseInt(match[2], 10),
      y: Number.parseInt(match[3], 10),
      url,
      // Fetch the exact URL proven by the trajectory. Only the manifest key is
      // canonicalized; reconstructing an upstream URL can change its semantics.
      fetchUrl: rawUrl,
    });
  }
  return {
    items: [...unique.values()],
    logCount: logs.length,
    observedCount: observed.length,
    uniqueObservedCount: observedUrls.size,
    observedUrls,
  };
}

function findBundledLoaderEpoch(entries) {
  const entry = entries.find((item) => (
    item?.file
    && typeof item.url === 'string'
    && item.url.startsWith('https://maps.googleapis.com/maps/api/js?')
  ));
  if (!entry) return null;
  const path = resolve(FILES_DIR, entry.file);
  if (!existsSync(path)) return null;
  const text = readFileSync(path, 'utf-8');
  return parseLoaderEpoch(text);
}

function parseLoaderEpoch(text) {
  const match = text.match(/\/maps\/vt",(\d+)000000,(\d+),(\d+)/);
  if (!match || match[1] !== match[2]) return null;
  return {
    epoch: match[2],
    build: match[3],
  };
}

async function resolveUpstreamLoaderEpoch(entries) {
  if (TILE_EPOCH_OVERRIDE && TILE_EPOCH_BUILD_OVERRIDE) {
    return {
      epoch: TILE_EPOCH_OVERRIDE,
      build: TILE_EPOCH_BUILD_OVERRIDE,
    };
  }
  if (TILE_EPOCH_OVERRIDE || TILE_EPOCH_BUILD_OVERRIDE) {
    throw new Error('Set both MAP_VECTOR_BACKFILL_TILE_EPOCH and MAP_VECTOR_BACKFILL_TILE_EPOCH_BUILD');
  }
  const entry = entries.find((item) => (
    typeof item?.url === 'string'
    && item.url.startsWith('https://maps.googleapis.com/maps/api/js?')
  ));
  if (!entry) throw new Error('Cannot find the Google Maps loader URL in the vector manifest');
  const response = await requestBuffer(entry.url);
  if (response.status < 200 || response.status >= 300) {
    throw new Error(`Cannot resolve upstream tile epoch: loader returned HTTP ${response.status}`);
  }
  const parsed = parseLoaderEpoch(response.body.toString('utf-8'));
  if (!parsed) throw new Error('Cannot parse the upstream Google Maps tile epoch');
  return parsed;
}

function loadCenters() {
  const raw = process.env.MAP_VECTOR_BACKFILL_CENTERS;
  if (raw?.trim()) {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) throw new Error('MAP_VECTOR_BACKFILL_CENTERS must be a JSON array');
    return parsed.map((item, index) => normalizeCenter(item, `center-${index + 1}`));
  }

  const places = readJson(PLACES_PATH).places || {};
  const rows = Object.values(places);
  return DEFAULT_CENTER_NAMES.map((name) => {
    const row = rows.find((r) => localizedName(r) === name)
      || rows.find((r) => JSON.stringify(r).includes(name));
    if (!row) throw new Error(`Cannot find default center in places.json: ${name}`);
    return normalizeCenter({
      name,
      lat: row.lat,
      lng: row.lng,
    }, name);
  });
}

function localizedName(row) {
  if (!row?.name) return '';
  return typeof row.name === 'string' ? row.name : (row.name.zh || row.name.en || '');
}

function normalizeCenter(item, fallbackName) {
  const lat = Number(item?.lat);
  const lng = Number(item?.lng);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    throw new Error(`Invalid center ${fallbackName}: lat/lng required`);
  }
  return {
    name: String(item?.name || fallbackName),
    lat,
    lng,
  };
}

function findVectorTileTemplate(entries) {
  const entry = entries.find((e) => typeof e.url === 'string' && e.url.includes('/maps/vt/pb='));
  if (!entry) throw new Error('No /maps/vt/pb= vector tile URL found in vector manifest');
  const match = entry.url.match(/^(.*\/maps\/vt\/pb=!1m4!1m3!1i)\d+(!2i)\d+(!3i)\d+(.*)$/);
  if (!match) throw new Error(`Cannot parse vector tile pb template: ${entry.url}`);
  return {
    prefix: match[1],
    xToken: match[2],
    yToken: match[3],
    suffix: match[4],
    contentType: entry.contentType || 'application/vnd.google.octet-stream-compressible; charset=x-user-defined',
    cacheControl: entry.cacheControl,
  };
}

function tileUrl(template, z, x, y) {
  return `${template.prefix}${z}${template.xToken}${x}${template.yToken}${y}${template.suffix}`;
}

function lonToTileX(lng, z) {
  const n = 2 ** z;
  return Math.floor(((lng + 180) / 360) * n);
}

function latToTileY(lat, z) {
  const n = 2 ** z;
  const clamped = Math.max(-85.05112878, Math.min(85.05112878, lat));
  const rad = (clamped * Math.PI) / 180;
  return Math.floor((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2 * n);
}

function tileBoundsForRadius(center, radiusM, z) {
  const dLat = radiusM / 111_320;
  const dLng = radiusM / (111_320 * Math.cos((center.lat * Math.PI) / 180));
  const n = 2 ** z;
  const xMin = clampInt(lonToTileX(center.lng - dLng, z), 0, n - 1);
  const xMax = clampInt(lonToTileX(center.lng + dLng, z), 0, n - 1);
  const yMin = clampInt(latToTileY(center.lat + dLat, z), 0, n - 1);
  const yMax = clampInt(latToTileY(center.lat - dLat, z), 0, n - 1);
  return {
    xMin: Math.min(xMin, xMax),
    xMax: Math.max(xMin, xMax),
    yMin: Math.min(yMin, yMax),
    yMax: Math.max(yMin, yMax),
  };
}

function clampInt(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function buildWorkItems(centers, template, existingEntries) {
  const existing = new Map();
  for (const entry of existingEntries) {
    if (entry?.url && entry?.file && existsSync(resolve(FILES_DIR, entry.file))) {
      existing.set(entry.url, entry);
    }
  }

  const items = [];
  const seen = new Set();
  const perZoom = new Map();
  for (const center of centers) {
    for (const z of ZOOMS) {
      const bounds = tileBoundsForRadius(center, RADIUS_M, z);
      let total = 0;
      let missing = 0;
      for (let x = bounds.xMin; x <= bounds.xMax; x++) {
        for (let y = bounds.yMin; y <= bounds.yMax; y++) {
          total++;
          const url = tileUrl(template, z, x, y);
          if (existing.has(url) || seen.has(url)) continue;
          seen.add(url);
          missing++;
          items.push({ center: center.name, z, x, y, url });
        }
      }
      const prev = perZoom.get(z) || { total: 0, missing: 0 };
      prev.total += total;
      prev.missing += missing;
      perZoom.set(z, prev);
    }
  }
  return { items, perZoom };
}

function guessExt(contentType) {
  const ct = (contentType || '').toLowerCase();
  if (ct.includes('javascript')) return '.js';
  if (ct.startsWith('application/json')) return '.json';
  if (ct.startsWith('text/css')) return '.css';
  if (ct.startsWith('font/woff2') || ct.includes('woff2')) return '.woff2';
  if (ct.startsWith('font/woff') || ct.includes('woff')) return '.woff';
  if (ct.startsWith('image/png')) return '.png';
  if (ct.startsWith('image/jpeg')) return '.jpg';
  if (ct.startsWith('image/webp')) return '.webp';
  return '.bin';
}

async function requestBuffer(url, redirects = 0) {
  if (redirects > 5) throw new Error('too many redirects');
  return new Promise((resolveRequest, rejectRequest) => {
    const req = https.get(url, {
      agent: PROXY_AGENT,
      headers: {
        Accept: '*/*',
        // Google rejects otherwise identical proxy requests that omit a user
        // agent. This identifies the maintenance tool without impersonating a
        // browser or affecting the stored response bytes.
        'User-Agent': 'mobilegym-map-cache-backfill/1.0',
      },
    }, (res) => {
      const status = res.statusCode || 0;
      if (status >= 300 && status < 400 && res.headers.location) {
        res.resume();
        requestBuffer(new URL(res.headers.location, url).toString(), redirects + 1)
          .then(resolveRequest, rejectRequest);
        return;
      }
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => resolveRequest({
        status,
        headers: res.headers,
        body: Buffer.concat(chunks),
      }));
    });
    req.setTimeout(30_000, () => req.destroy(new Error('request timeout')));
    req.on('error', rejectRequest);
  });
}

async function fetchOne(item, template) {
  const res = await requestBuffer(item.fetchUrl || liveTileUrl(item.url));
  if (res.status < 200 || res.status >= 300) {
    throw new Error(`HTTP ${res.status}`);
  }
  const contentType = res.headers['content-type'] || template.contentType;
  const body = res.body;
  if (body.length === 0) throw new Error('empty body');
  const file = `${createHash('sha1').update(item.url).digest('hex').slice(0, 16)}${guessExt(contentType)}`;
  writeFileSync(resolve(FILES_DIR, file), body);
  return {
    url: item.url,
    file,
    contentType,
    cacheControl: res.headers['cache-control'] || template.cacheControl || undefined,
    status: res.status,
    size: body.length,
    bodyHash: createHash('sha1').update(body).digest('hex'),
  };
}

async function runQueue(items, template) {
  const results = [];
  const failures = [];
  let cursor = 0;
  let done = 0;
  async function worker() {
    while (cursor < items.length) {
      const item = items[cursor++];
      try {
        const entry = await fetchOne(item, template);
        results.push(entry);
      } catch (error) {
        failures.push({ item, error: error instanceof Error ? error.message : String(error) });
      }
      done++;
      if (done % 50 === 0 || done === items.length) {
        process.stdout.write(`\r[vector-backfill] fetched ${done}/${items.length}, ok=${results.length}, fail=${failures.length}     `);
      }
    }
  }
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  process.stdout.write('\n');
  return { results, failures };
}

function makeSnapshotCacheName(entries) {
  const seed = entries
    .map((entry) => [
      entry.url,
      entry.file,
      entry.status,
      entry.size,
      entry.bodyHash,
    ].join('\t'))
    .join('\n');
  const hash = createHash('sha1').update(seed).digest('hex').slice(0, 12);
  return `map-vector-cache-${hash}`;
}

function updateMapSwCacheName(cacheName) {
  updateServiceWorkerCacheName({
    swPath: SW_PATH,
    constName: 'VECTOR_CACHE_NAME',
    cacheName,
    prefix: 'map-vector-cache-',
    logPrefix: '[vector-backfill]',
  });
}

function writeBootstrapManifest(entries, sourceCacheName) {
  // Backfill only adds tile payloads, but still rewrite the bootstrap manifest
  // from the merged vector manifest so every vector-writing script has the same
  // derivation path and future non-tile changes cannot leave it stale.
  const result = writeVectorBootstrapManifest({
    entries,
    outputPath: BOOTSTRAP_MANIFEST_PATH,
    source: VECTOR_MANIFEST_SOURCE,
    sourceCacheName,
  });
  updateServiceWorkerCacheName({
    swPath: SW_PATH,
    constName: 'BOOTSTRAP_CACHE_NAME',
    cacheName: result.cacheName,
    prefix: 'map-bootstrap-cache-',
    logPrefix: '[vector-backfill]',
  });
  return result;
}

function mergeAndWriteManifest(manifest, newEntries) {
  const byUrl = new Map();
  for (const entry of manifest.entries || []) {
    if (entry?.url && entry?.file) byUrl.set(entry.url, entry);
  }
  for (const entry of newEntries) byUrl.set(entry.url, entry);
  const entries = [...byUrl.values()].sort((a, b) => a.url.localeCompare(b.url));
  const cacheName = makeSnapshotCacheName(entries);
  const nextManifest = {
    ...manifest,
    version: 1,
    generatedAt: new Date().toISOString(),
    cacheName,
    entries,
  };
  writeFileSync(MANIFEST_PATH, JSON.stringify(nextManifest, null, 2));
  updateMapSwCacheName(cacheName);
  const bootstrapManifest = writeBootstrapManifest(entries, cacheName);
  console.log(`[vector-backfill] bootstrap manifest: ${bootstrapManifest.entries.length} startup entries`);
  return nextManifest;
}

async function main() {
  if (!ZOOMS.length) throw new Error('No zoom levels configured');
  if (!Number.isFinite(RADIUS_M) || RADIUS_M <= 0) throw new Error('MAP_VECTOR_BACKFILL_RADIUS_M must be > 0');
  if (!existsSync(FILES_DIR)) mkdirSync(FILES_DIR, { recursive: true });

  const manifest = loadManifest();
  const template = findVectorTileTemplate(manifest.entries || []);
  const trajectoryMode = INPUT_PATHS.length > 0;
  const centers = trajectoryMode ? [] : loadCenters();
  const trajectory = trajectoryMode
    ? loadTrajectoryWorkItems(INPUT_PATHS, manifest.entries || [])
    : null;
  const areaWork = trajectoryMode
    ? null
    : buildWorkItems(centers, template, manifest.entries || []);
  const allItems = trajectory ? trajectory.items : areaWork.items;
  const items = LIMIT > 0 ? allItems.slice(0, LIMIT) : allItems;

  if (trajectory && REMOVE_OBSERVED) {
    const keptEntries = (manifest.entries || []).filter((entry) => (
      !entry?.url || !trajectory.observedUrls.has(canonicalUrl(entry.url))
    ));
    const removed = (manifest.entries?.length || 0) - keptEntries.length;
    console.log(`[vector-backfill] remove-observed: matched ${removed} manifest entries; payload files are retained`);
    if (DRY_RUN || removed === 0) return;
    mergeAndWriteManifest({ ...manifest, entries: keptEntries }, []);
    return;
  }

  if (trajectory) {
    console.log(`[vector-backfill] trajectory logs=${trajectory.logCount}, observed 504s=${trajectory.observedCount}, unique=${trajectory.uniqueObservedCount}`);
  } else {
    console.log('[vector-backfill] centers:', centers.map((c) => `${c.name}(${c.lat},${c.lng})`).join(', '));
    console.log(`[vector-backfill] radius=${RADIUS_M}m zooms=${ZOOMS.join(',')} concurrency=${CONCURRENCY}`);
    for (const z of ZOOMS) {
      const row = areaWork.perZoom.get(z) || { total: 0, missing: 0 };
      console.log(`[vector-backfill] z${z}: total tiles in requested areas=${row.total}, missing=${row.missing}`);
    }
  }
  console.log(`[vector-backfill] will fetch ${items.length}${LIMIT > 0 ? ` (limited from ${allItems.length})` : ''} tiles`);

  const loaderEpoch = findBundledLoaderEpoch(manifest.entries || []);
  if (loaderEpoch) {
    console.log(`[vector-backfill] bundled loader epoch=${loaderEpoch.build}/${loaderEpoch.epoch}`);
  }
  if (DRY_RUN && items.length === 0) {
    console.log('[vector-backfill] dry run: no files written');
    return;
  }
  if (items.length === 0) return;

  upstreamTileEpoch = await resolveUpstreamLoaderEpoch(manifest.entries || []);
  console.log(`[vector-backfill] proxy=${PROXY_URL ? 'configured' : 'not configured'} upstream tile epoch=${upstreamTileEpoch.build}/${upstreamTileEpoch.epoch}`);
  if (loaderEpoch && loaderEpoch.epoch !== upstreamTileEpoch.epoch && !ALLOW_EPOCH_MISMATCH) {
    throw new Error(
      `Refusing to mix upstream tile epoch ${upstreamTileEpoch.epoch} into bundled loader epoch ${loaderEpoch.epoch}. `
      + 'Refresh SDK/config and the task viewport tiles as one snapshot; do not treat a 200 download as render-compatible.',
    );
  }

  if (DRY_RUN) {
    console.log('[vector-backfill] dry run: no files written');
    return;
  }

  const beforeCount = manifest.entries?.length || 0;
  const { results, failures } = await runQueue(items, template);
  if (results.length === 0 && failures.length > 0) {
    console.warn('[vector-backfill] no successful responses; manifest and cache names were not changed');
    for (const failure of failures.slice(0, 12)) {
      console.warn(`  z${failure.item.z}/${failure.item.x}/${failure.item.y}: ${failure.error}`);
    }
    process.exitCode = 1;
    return;
  }
  const nextManifest = mergeAndWriteManifest(manifest, results);
  const addedBytes = results.reduce((sum, entry) => sum + (entry.size || 0), 0);
  console.log(`[vector-backfill] manifest: ${beforeCount} -> ${nextManifest.entries.length} entries`);
  console.log(`[vector-backfill] added ${results.length} tiles, ${(addedBytes / 1024 / 1024).toFixed(1)} MB`);
  if (failures.length) {
    console.warn(`[vector-backfill] failures=${failures.length}`);
    for (const failure of failures.slice(0, 12)) {
      console.warn(`  z${failure.item.z}/${failure.item.x}/${failure.item.y}: ${failure.error}`);
    }
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error('[vector-backfill] failed:', error);
  process.exit(1);
});
