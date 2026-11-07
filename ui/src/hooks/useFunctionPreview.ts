import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { ApiError, type RunSpec, api } from "@/lib/api";
import { functionFields } from "@/lib/functions";

const DEBOUNCE_MS = 250;

/** The server is the only judge of a formula, so ask it while the user types. */
export function useFunctionPreview(spec: RunSpec) {
  const key = JSON.stringify(functionFields(spec));
  const [asked, setAsked] = useState(key);
  useEffect(() => {
    const timer = setTimeout(() => setAsked(key), DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [key]);
  const query = useQuery({
    queryKey: ["function-preview", asked],
    queryFn: () => api.previewFunction(JSON.parse(asked)),
    retry: false,
    staleTime: Infinity,
    placeholderData: keepPreviousData,
  });
  const failed = query.isError && key === asked;
  return {
    preview: query.data,
    error: failed ? (query.error instanceof ApiError ? query.error.message : "Could not check this function.") : null,
    /** Not yet checked: starting now could still be refused. */
    checking: key !== asked || query.isFetching,
  };
}
