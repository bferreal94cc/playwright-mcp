/**
 * Extractors for pokene.com (a Wix Stores site).
 *
 * Two useful payloads are embedded in the server-rendered HTML, so no JS
 * execution is required:
 *   1. a JSON-LD <script> carrying schema.org Product name/description/images
 *   2. Wix's warm-up state, which carries the commercial fields that JSON-LD
 *      omits on this site: price, compare price, currency and live inventory.
 */

/**
 * Reads a balanced JSON object out of `text` that encloses `index`.
 *
 * Wix inlines its state as JSON embedded in a script body, so the object is
 * not addressable by DOM parsing. We scan left for candidate opening braces
 * and return the first one that parses and actually contains `index`.
 */
export function objectAround(text, index, maxSpan = 400_000) {
  let start = index;
  while (start > 0) {
    start = text.lastIndexOf('{', start - 1);
    if (start < 0) return null;

    let depth = 0;
    let inString = false;
    let escaped = false;
    const limit = Math.min(text.length, start + maxSpan);

    for (let i = start; i < limit; i++) {
      const c = text[i];
      if (escaped) { escaped = false; continue; }
      if (c === '\\') { escaped = true; continue; }
      if (c === '"') { inString = !inString; continue; }
      if (inString) continue;

      if (c === '{') depth++;
      else if (c === '}') {
        depth--;
        if (depth === 0) {
          if (i < index) break; // closed before our anchor - widen the search
          try {
            return JSON.parse(text.slice(start, i + 1));
          } catch {
            break; // not valid on its own, try an outer brace
          }
        }
      }
    }
  }
  return null;
}

/** Returns every parsed application/ld+json block, flattened. */
export function jsonLdBlocks(html) {
  const out = [];
  const re = /<script[^>]*type="application\/ld\+json"[^>]*>([\s\S]*?)<\/script>/gi;
  let m;
  while ((m = re.exec(html))) {
    try {
      const parsed = JSON.parse(m[1]);
      out.push(...(Array.isArray(parsed) ? parsed : [parsed]));
    } catch {
      // A malformed block should not abort the rest of the page.
    }
  }
  return out;
}

const decodeEntities = (s) =>
  typeof s === 'string'
    ? s
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(+d))
    : s;

const stripHtml = (s) =>
  typeof s === 'string' ? decodeEntities(s.replace(/<[^>]+>/g, ' ')).replace(/\s+/g, ' ').trim() : s;

/** Pulls the Wix Stores product record out of the warm-up state. */
export function wixProduct(html) {
  const anchor = html.indexOf('"formattedPrice"');
  if (anchor < 0) return null;
  const obj = objectAround(html, anchor);
  return obj && typeof obj.name === 'string' && 'price' in obj ? obj : null;
}

/**
 * Merges both payloads into one flat product row.
 * Returns null when the page carries no product (e.g. a soft 404).
 */
export function extractProduct(html, url) {
  const wix = wixProduct(html);
  const ld = jsonLdBlocks(html).find((b) => b['@type'] === 'Product') || null;
  if (!wix && !ld) return null;

  const inventory = wix?.inventory ?? {};
  const images = [];
  if (ld?.image) {
    for (const img of Array.isArray(ld.image) ? ld.image : [ld.image]) {
      const u = typeof img === 'string' ? img : img?.contentUrl;
      if (u) images.push(u);
    }
  }

  return {
    url,
    id: wix?.id ?? null,
    name: decodeEntities(wix?.name ?? ld?.name ?? null),
    slug: wix?.urlPart ?? url.split('/').pop(),
    brand: wix?.brand ?? ld?.brand?.name ?? null,
    sku: wix?.sku || null,
    price: wix?.price ?? null,
    comparePrice: wix?.comparePrice ?? null,
    discountedPrice: wix?.discountedPrice ?? null,
    currency: wix?.currency ?? null,
    formattedPrice: wix?.formattedPrice ?? null,
    // `isInStock` is the flag the storefront renders from; the nested status
    // string is kept because it distinguishes pre-order from plain in-stock.
    inStock: wix?.isInStock ?? null,
    inventoryStatus: inventory.status ?? null,
    quantity: inventory.quantity ?? null,
    availableForPreOrder: inventory.availableForPreOrder ?? null,
    trackingInventory: wix?.isTrackingInventory ?? null,
    visible: wix?.isVisible ?? null,
    ribbon: wix?.ribbon || null,
    productType: wix?.productType ?? null,
    weight: wix?.weight ?? null,
    categoryIds: wix?.categoryIds ?? [],
    description: stripHtml(wix?.description ?? ld?.description ?? null),
    images,
    scrapedAt: new Date().toISOString(),
  };
}

/** Minimal record for non-product pages (content pages, categories). */
export function extractPage(html, url) {
  const title = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1];
  const desc = html.match(/<meta[^>]+name="description"[^>]+content="([^"]*)"/i)?.[1];
  const h1s = [...html.matchAll(/<h1[^>]*>([\s\S]*?)<\/h1>/gi)].map((m) => stripHtml(m[1])).filter(Boolean);
  return {
    url,
    title: decodeEntities(title?.trim() ?? null),
    description: decodeEntities(desc ?? null),
    headings: h1s.slice(0, 5),
    scrapedAt: new Date().toISOString(),
  };
}

/** Extracts <loc> values from a sitemap or sitemap index. */
export function sitemapLocs(xml) {
  return [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1].trim());
}
