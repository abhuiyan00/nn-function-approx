/** Repository-relative links in the Markdown docs, mapped to routes of this app. */
export function rewriteHref(href: string | undefined): string | undefined {
  if (!href) return undefined;
  if (/^https?:\/\//i.test(href) || href.startsWith("#") || href.startsWith("mailto:")) return href;
  if (/^[a-z][a-z0-9+.-]*:/i.test(href)) return undefined; // javascript:, data:, ...
  const image = href.match(/(?:^|\/)(figures|screenshots)\/([^/]+)$/);
  if (image) return `/${image[1]}/${image[2]}`;
  const doc = href.match(/([^/]+)\.md(#.*)?$/i);
  if (doc) return `/documents/${doc[1].toLowerCase()}${doc[2] ?? ""}`;
  return undefined;
}

export function rewriteImage(src: string | undefined): string | undefined {
  return rewriteHref(src);
}
