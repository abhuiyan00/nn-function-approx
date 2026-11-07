import { type Marker, markerPath } from "./core";

export interface LegendItem {
  label: string;
  color: string;
  marker?: Marker | null;
  band?: boolean;
}

/** Text wears ink tokens; the small key beside it carries the series colour. */
export function Legend({ items }: { items: LegendItem[] }) {
  if (items.length < 2) return null;
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5">
          <svg width="20" height="10" aria-hidden="true" className="shrink-0">
            <line x1="1" y1="5" x2="19" y2="5" stroke={item.color} strokeWidth="2" strokeLinecap="round" />
            {item.marker && <path d={markerPath(item.marker, 3.4)} transform="translate(10,5)" fill={item.color} stroke="var(--surface)" strokeWidth="1.2" />}
          </svg>
          {item.label}
        </li>
      ))}
    </ul>
  );
}
