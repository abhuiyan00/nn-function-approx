import { type ReactNode, createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useState } from "react";
import { flushSync } from "react-dom";
import type { Theme } from "@/components/charts/core";

const KEY = "funcprox-theme";

function stored(): Theme | null {
  try {
    const t = localStorage.getItem(KEY);
    return t === "light" || t === "dark" ? t : null;
  } catch {
    return null; // storage can be blocked; the OS preference still applies
  }
}

const system = (): Theme => (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({ theme: "light", toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoice] = useState<Theme | null>(stored);
  const [os, setOs] = useState<Theme>(system);
  const theme = choice ?? os;

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setOs(system());
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  useLayoutEffect(() => { // before paint, and inside flushSync: the cross-fade below needs the new colours at once
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const toggle = useCallback(() => {
    const flip = () => setChoice((current) => {
      const next: Theme = (current ?? system()) === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(KEY, next);
      } catch {
        /* the toggle still works for this visit */
      }
      return next;
    });
    // browsers with view transitions fade the whole page from one theme to the other
    if (document.startViewTransition) document.startViewTransition(() => flushSync(flip));
    else flip();
  }, []);

  const value = useMemo(() => ({ theme, toggle }), [theme, toggle]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);
