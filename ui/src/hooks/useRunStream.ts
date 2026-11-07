import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useReducer } from "react";
import type { RunDetail, RunFinal, RunFrame, RunSummary } from "@/lib/api";

type Action =
  | { type: "clear" }
  | { type: "init"; run: RunDetail }
  | { type: "frame"; frame: RunFrame }
  | { type: "done"; summary: RunSummary & { final: RunFinal | null } };

function reducer(run: RunDetail | null, action: Action): RunDetail | null {
  if (action.type === "clear") return null;
  if (action.type === "init") return action.run;
  if (!run) return run;
  if (action.type === "done") return { ...run, ...action.summary };

  const { history, test, components, live, status } = action.frame;
  const next: RunDetail = { ...run };
  if (status) next.status = status;
  if (live) next.live = live;
  if (history || test) {
    const h = run.history;
    next.history = {
      epoch: history ? h.epoch.concat(history.epoch) : h.epoch,
      train: history ? h.train.concat(history.train) : h.train,
      val: history ? h.val.concat(history.val) : h.val,
      wall: history ? h.wall.concat(history.wall) : h.wall,
      updates: history ? h.updates.concat(history.updates) : h.updates,
      mu: history ? h.mu.concat(history.mu) : h.mu,
      test_epoch: test ? h.test_epoch.concat(test.epoch) : h.test_epoch,
      test_mse: test ? h.test_mse.concat(test.mse) : h.test_mse,
    };
  }
  if (components && run.components) {
    next.components = {
      ...run.components,
      epoch: run.components.epoch.concat(components.epoch),
      rel: run.components.rel.concat(components.rel),
    };
  }
  return next;
}

/** Follows one lab run over Server-Sent Events; a reconnect starts again from the init event. */
export function useRunStream(id: string | null) {
  const [run, dispatch] = useReducer(reducer, null);
  const queryClient = useQueryClient();

  useEffect(() => {
    dispatch({ type: "clear" });
    if (!id) return;
    const source = new EventSource(`/api/lab/runs/${id}/stream`);
    source.addEventListener("init", (e) => {
      dispatch({ type: "init", run: JSON.parse(e.data) });
      queryClient.invalidateQueries({ queryKey: ["lab-runs"] });
    });
    source.addEventListener("frame", (e) => {
      const frame: RunFrame = JSON.parse(e.data);
      dispatch({ type: "frame", frame });
      if (frame.status) queryClient.invalidateQueries({ queryKey: ["lab-runs"] });
    });
    source.addEventListener("done", (e) => {
      dispatch({ type: "done", summary: JSON.parse(e.data) });
      source.close();
      queryClient.invalidateQueries({ queryKey: ["lab-runs"] });
    });
    return () => source.close();
  }, [id, queryClient]);

  return run;
}
