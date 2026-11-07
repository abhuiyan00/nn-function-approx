import { type ComponentProps, Suspense, lazy } from "react";
import ReactMarkdown from "react-markdown";
import { Link } from "react-router";
import remarkGfm from "remark-gfm";
import { rewriteHref, rewriteImage } from "@/lib/markdown";
import { cn } from "@/lib/utils";

const Mermaid = lazy(() => import("./Mermaid")); // large library, only loaded for documents with a diagram

const components: ComponentProps<typeof ReactMarkdown>["components"] = {
  h1: ({ node: _node, ...p }) => <h1 className="mt-6 mb-3 text-2xl font-semibold tracking-tight text-balance first:mt-0" {...p} />,
  h2: ({ node: _node, ...p }) => <h2 className="mt-7 mb-2 border-b pb-1.5 text-lg font-semibold tracking-tight first:mt-0" {...p} />,
  h3: ({ node: _node, ...p }) => <h3 className="mt-5 mb-1.5 text-[15px] font-semibold first:mt-0" {...p} />,
  p: ({ node: _node, ...p }) => <p className="my-2 leading-relaxed" {...p} />,
  ul: ({ node: _node, ...p }) => <ul className="my-2 list-disc space-y-1 pl-5 marker:text-subtle" {...p} />,
  ol: ({ node: _node, ...p }) => <ol className="my-2 list-decimal space-y-1 pl-5 marker:text-subtle" {...p} />,
  li: ({ node: _node, ...p }) => <li className="leading-relaxed" {...p} />,
  strong: ({ node: _node, ...p }) => <strong className="font-semibold" {...p} />,
  hr: () => <hr className="my-6" />,
  blockquote: ({ node: _node, ...p }) => <blockquote className="my-3 border-l-2 pl-3 text-muted-foreground" {...p} />,
  a: ({ href, children }) => {
    const to = rewriteHref(href);
    if (!to) return <span>{children}</span>;
    const style = "text-primary underline underline-offset-2 hover:no-underline";
    if (to.startsWith("/documents/")) return <Link to={to} className={style}>{children}</Link>;
    return <a href={to} className={style} target={to.startsWith("#") ? undefined : "_blank"} rel="noreferrer">{children}</a>;
  },
  img: ({ src, alt }) => {
    const url = rewriteImage(typeof src === "string" ? src : undefined);
    return url ? <img src={url} alt={alt ?? ""} loading="lazy" className="my-3 max-w-full rounded-md border bg-white" /> : null;
  },
  table: ({ node: _node, ...p }) => (
    <div className="my-3 overflow-x-auto rounded-md border bg-card">
      <table className="w-full text-[13px]" {...p} />
    </div>
  ),
  thead: ({ node: _node, ...p }) => <thead className="bg-wash text-left text-muted-foreground" {...p} />,
  th: ({ node: _node, ...p }) => <th scope="col" className="px-3 py-2 font-medium whitespace-nowrap" {...p} />,
  td: ({ node: _node, ...p }) => <td className="border-t px-3 py-1.5 align-top" {...p} />,
  code: ({ className, children }) => {
    const language = /language-(\w+)/.exec(className ?? "")?.[1];
    if (language === "mermaid") {
      return (
        <Suspense fallback={<pre className="text-xs text-subtle">Loading diagram…</pre>}>
          <Mermaid source={String(children)} />
        </Suspense>
      );
    }
    if (className) return <code className="font-mono text-[12.5px]">{children}</code>;
    return <code className="rounded bg-wash px-1 py-0.5 font-mono text-[12.5px]">{children}</code>;
  },
  pre: ({ children }) => {
    const child = children as { props?: { className?: string } } | undefined;
    if (child?.props?.className?.includes("language-mermaid")) return <>{children}</>;
    return <pre className="my-3 overflow-x-auto rounded-md border bg-wash p-3 leading-relaxed">{children}</pre>;
  },
};

export function Markdown({ children, className }: { children: string; className?: string }) {
  return (
    <div className={cn("min-w-0 text-sm", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>{children}</ReactMarkdown>
    </div>
  );
}
