import mermaid from "mermaid";
import { useEffect, useId, useState } from "react";
import { useTheme } from "@/hooks/useTheme";

/** Renders a diagram from a project document; falls back to the source text if it cannot. */
export default function Mermaid({ source }: { source: string }) {
  const { theme } = useTheme();
  const id = `mermaid-${useId().replace(/\W/g, "")}`;
  const [svg, setSvg] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let current = true;
    mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: theme === "dark" ? "dark" : "neutral", fontFamily: "Inter Variable, system-ui, sans-serif" });
    mermaid
      .render(id, source)
      .then((result) => {
        if (current) setSvg(result.svg);
      })
      .catch((error) => {
        console.warn("Mermaid could not draw this diagram:", error);
        if (current) setFailed(true);
      });
    return () => {
      current = false;
    };
  }, [source, theme, id]);

  if (failed || !svg) return <pre className="my-3 overflow-x-auto rounded-md border bg-wash p-3 text-xs">{source}</pre>;
  return <div className="my-3 overflow-x-auto rounded-md border bg-card p-3" dangerouslySetInnerHTML={{ __html: svg }} />;
}
