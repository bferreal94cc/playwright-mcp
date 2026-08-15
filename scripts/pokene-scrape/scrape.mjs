#!/usr/bin/env node
/**
 * Scrapes the public catalogue of pokene.com.
 *
 * Usage:
 *   node scripts/pokene-scrape/scrape.mjs [options]
 *
 *   --backend=direct|firecrawl   fetch strategy (default: direct)
 *   --only=all|products|pages|categories   what to scrape (default: all)
 *   --concurrency=N              parallel requests (default: 4)
 *   --delay=MS                   pause between requests per worker (default: 250)
 *   --limit=N                    cap URLs per group, for smoke runs
 *   --out=DIR                    output directory (default: <script dir>/data)
 *
 * The firecrawl backend needs FIRECRAWL_API_KEY, and honours FIRECRAWL_API_URL
 * when pointing at a self-hosted instance. Both backends share one extractor,
 * so output is identical either way.
 *
 * Scope note: pokene.com's sitemap index also lists member-profile sitemaps.
 * Those are personal data belonging to individual site members and are
 * deliberately excluded here - see EXCLUDED_SITEMAPS.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { extractPage, extractProduct, sitemapLocs } from './extract.mjs';

const ORIGIN = 'https://www.pokene.com';
const SITEMAP_INDEX = `${ORIGIN}/sitemap.xml`;

// robots.txt allows `User-agent: *`, but identify the client honestly anyway.
const USER_AGENT =
  'pokene-catalogue-scraper/1.0 (+https://github.com/bferreal94cc/playwright-mcp; contact via repo issues)';

/** Personal data - out of scope for a catalogue scrape. */
const EXCLUDED_SITEMAPS = [/member-profiles/i];

const GROUPS = {
  products: { sitemap: 'store-products-sitemap.xml', kind: 'product' },
  categories: { sitemap: 'store-categories-sitemap.xml', kind: 'page' },
  pages: { sitemap: 'pages-sitemap.xml', kind: 'page' },
};

function parseArgs(argv) {
  const opts = {
    backend: 'direct',
    only: 'all',
    concurrency: 4,
    delay: 250,
    limit: 0,
    out: resolve(dirname(fileURLToPath(import.meta.url)), 'data'),
  };
  for (const arg of argv) {
    const [key, value = ''] = arg.replace(/^--/, '').split('=');
    if (!(key in opts)) throw new Error(`unknown option: ${arg}`);
    opts[key] = typeof opts[key] === 'number' ? Number(value) : value;
  }
  if (!['direct', 'firecrawl'].includes(opts.backend)) throw new Error(`bad --backend: ${opts.backend}`);
  if (!['all', ...Object.keys(GROUPS)].includes(opts.only)) throw new Error(`bad --only: ${opts.only}`);
  return opts;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** GET with retry on transient failures and 429/5xx. */
async function fetchWithRetry(url, init = {}, attempts = 4) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      const res = await fetch(url, {
        ...init,
        headers: { 'user-agent': USER_AGENT, ...(init.headers ?? {}) },
      });
      if (res.status === 429 || res.status >= 500) {
        throw new Error(`HTTP ${res.status}`);
      }
      if (!res.ok) {
        // 404/403 are terminal for this URL - do not burn retries on them.
        const err = new Error(`HTTP ${res.status}`);
        err.terminal = true;
        throw err;
      }
      return res;
    } catch (err) {
      lastError = err;
      if (err.terminal || attempt === attempts) break;
      await sleep(2 ** attempt * 500); // 1s, 2s, 4s
    }
  }
  throw lastError;
}

/** Backend: plain HTTP fetch of the server-rendered HTML. */
async function fetchDirect(url) {
  const res = await fetchWithRetry(url);
  return res.text();
}

/** Backend: Firecrawl scrape API, returning the same raw HTML. */
async function fetchFirecrawl(url) {
  const key = process.env.FIRECRAWL_API_KEY;
  if (!key) throw new Error('FIRECRAWL_API_KEY is not set (required for --backend=firecrawl)');
  const base = (process.env.FIRECRAWL_API_URL ?? 'https://api.firecrawl.dev').replace(/\/$/, '');

  const res = await fetchWithRetry(`${base}/v2/scrape`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({ url, formats: ['rawHtml'], onlyMainContent: false }),
  });
  const body = await res.json();
  const html = body?.data?.rawHtml ?? body?.data?.html;
  if (!html) throw new Error(`firecrawl returned no HTML for ${url}`);
  return html;
}

async function discover(limit) {
  const index = await (await fetchWithRetry(SITEMAP_INDEX)).text();
  const available = sitemapLocs(index).filter((loc) => !EXCLUDED_SITEMAPS.some((re) => re.test(loc)));
  const skipped = sitemapLocs(index).length - available.length;

  const groups = {};
  for (const [name, { sitemap }] of Object.entries(GROUPS)) {
    const loc = available.find((u) => u.endsWith(sitemap));
    if (!loc) {
      groups[name] = [];
      continue;
    }
    const xml = await (await fetchWithRetry(loc)).text();
    const urls = sitemapLocs(xml);
    groups[name] = limit > 0 ? urls.slice(0, limit) : urls;
  }
  return { groups, skipped };
}

/** Runs `worker` over `items` with bounded concurrency and a per-request delay. */
async function pool(items, concurrency, delay, worker) {
  const results = [];
  let cursor = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      results[index] = await worker(items[index], index);
      if (delay > 0) await sleep(delay);
    }
  });
  await Promise.all(runners);
  return results;
}

function toCsv(rows, columns) {
  const cell = (v) => {
    if (v === null || v === undefined) return '';
    const s = Array.isArray(v) ? v.join('|') : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [columns.join(','), ...rows.map((r) => columns.map((c) => cell(r[c])).join(','))].join('\n');
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const fetchHtml = opts.backend === 'firecrawl' ? fetchFirecrawl : fetchDirect;

  console.log(`[pokene] backend=${opts.backend} concurrency=${opts.concurrency} delay=${opts.delay}ms`);
  const { groups, skipped } = await discover(opts.limit);
  if (skipped) console.log(`[pokene] skipped ${skipped} member-profile sitemap(s) (personal data, out of scope)`);

  const selected = opts.only === 'all' ? Object.keys(GROUPS) : [opts.only];
  await mkdir(opts.out, { recursive: true });

  const summary = { startedAt: new Date().toISOString(), backend: opts.backend, groups: {} };

  for (const name of selected) {
    const urls = groups[name] ?? [];
    const { kind } = GROUPS[name];
    console.log(`[pokene] ${name}: ${urls.length} url(s)`);

    let done = 0;
    const failures = [];
    const settled = await pool(urls, opts.concurrency, opts.delay, async (url) => {
      try {
        // Wix intermittently serves a render with no warm-up state, so an
        // empty extraction is retried rather than treated as a dead URL.
        let lastError;
        for (let attempt = 1; attempt <= 3; attempt++) {
          try {
            const html = await fetchHtml(url);
            const record = kind === 'product' ? extractProduct(html, url) : extractPage(html, url);
            if (record) return record;
            lastError = new Error('no extractable content');
          } catch (err) {
            if (err.terminal) throw err;
            lastError = err;
          }
          if (attempt < 3) await sleep(attempt * 1000);
        }
        throw lastError;
      } catch (err) {
        failures.push({ url, error: String(err.message ?? err) });
        return null;
      } finally {
        done++;
        if (done % 25 === 0 || done === urls.length) {
          process.stdout.write(`\r[pokene] ${name}: ${done}/${urls.length}`);
        }
      }
    });
    if (urls.length) process.stdout.write('\n');

    const records = settled.filter(Boolean);
    await writeFile(`${opts.out}/${name}.json`, JSON.stringify(records, null, 2));

    if (kind === 'product') {
      const columns = [
        'name', 'slug', 'brand', 'sku', 'price', 'comparePrice', 'discountedPrice',
        'currency', 'inStock', 'inventoryStatus', 'quantity', 'availableForPreOrder',
        'ribbon', 'productType', 'weight', 'url',
      ];
      await writeFile(`${opts.out}/${name}.csv`, toCsv(records, columns));
    }

    summary.groups[name] = { urls: urls.length, scraped: records.length, failed: failures.length, failures };
    console.log(`[pokene] ${name}: ${records.length} ok, ${failures.length} failed`);
  }

  summary.finishedAt = new Date().toISOString();
  await writeFile(`${opts.out}/summary.json`, JSON.stringify(summary, null, 2));
  console.log(`[pokene] wrote output to ${opts.out}`);
}

main().catch((err) => {
  console.error(`[pokene] fatal: ${err.message ?? err}`);
  process.exit(1);
});
