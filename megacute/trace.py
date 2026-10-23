"""In-kernel tracing: decode the per-SM, per-role globaltimer records of one launch.

    prog = mc.compile(g, trace=True)
    prog(**inputs)
    tl = prog.timeline()
    print(tl.summary())                 # per-op time split: dependency wait, load, compute
    tl.save_chrome("trace.json")        # open in Perfetto / chrome://tracing: one process per SM

Records (globaltimer ns, see kernel.py): loader (fetched, deps satisfied, loads issued),
mma (fetched, -, issued), compute (fetched, dep_ok, published). All times are rebased to the
earliest record of the launch.
"""

from __future__ import annotations

import json
from collections import defaultdict

ROLES = ("loader", "mma", "compute")


class Timeline:
    def __init__(self, prog, buf):
        self.prog = prog
        self.records = []  # (sm, role, seq, t0, t1, t2)
        sms, roles, cap, _ = buf.shape
        nz = buf[..., 1].nonzero(as_tuple=False)
        for sm, role, ev in nz.tolist():
            seq, t0, t1, t2 = buf[sm, role, ev].tolist()
            self.records.append((sm, role, seq, t0, t1, t2))
        self.t_base = min((r[3] for r in self.records), default=0)
        self.node_of = []
        for k, (a, b) in enumerate(prog.sched.node_ranges):
            self.node_of += [k] * (b - a)

    def span_us(self):
        if not self.records:
            return 0.0
        return (max(r[5] for r in self.records) - self.t_base) / 1e3

    def _label(self, seq):
        k = self.node_of[seq]
        return f"{k}:{self.prog.g.ops[k].kind}"

    def summary(self) -> str:
        """Per node: instructions, total compute-side time, mean dependency wait, mean busy time."""
        agg = defaultdict(lambda: {"n": 0, "dep": 0.0, "busy": 0.0, "first": 1e30, "last": 0.0})
        for sm, role, seq, t0, t1, t2 in self.records:
            if role != 2:
                continue
            a = agg[self.node_of[seq]]
            a["n"] += 1
            a["dep"] += (t1 - t0) / 1e3
            a["busy"] += (t2 - t1) / 1e3
            a["first"] = min(a["first"], (t0 - self.t_base) / 1e3)
            a["last"] = max(a["last"], (t2 - self.t_base) / 1e3)
        lines = [f"launch span {self.span_us():.1f} us over {len({r[0] for r in self.records})} SMs; "
                 f"{len([r for r in self.records if r[1] == 2])} instructions traced",
                 f"{'node':<22}{'insts':>6}{'start':>9}{'end':>9}{'dep-wait/inst':>15}{'busy/inst':>12}"]
        for k in sorted(agg):
            a = agg[k]
            lines.append(f"{k:>3} {self.prog.g.ops[k].kind:<18}{a['n']:>6}{a['first']:>9.1f}{a['last']:>9.1f}"
                         f"{a['dep'] / a['n']:>15.2f}{a['busy'] / a['n']:>12.2f}")
        return "\n".join(lines)

    def chrome_events(self):
        ev = []
        for sm, role, seq, t0, t1, t2 in self.records:
            name = self._label(seq)
            base = lambda t: (t - self.t_base) / 1e3
            if role == 0:
                if t1 > t0:
                    ev.append({"name": "dep-wait " + name, "ph": "X", "pid": sm, "tid": "loader",
                               "ts": base(t0), "dur": (t1 - t0) / 1e3, "args": {"inst": seq}})
                ev.append({"name": "load " + name, "ph": "X", "pid": sm, "tid": "loader",
                           "ts": base(t1), "dur": max(0.0, (t2 - t1) / 1e3), "args": {"inst": seq}})
            elif role == 1:
                ev.append({"name": "mma " + name, "ph": "X", "pid": sm, "tid": "mma",
                           "ts": base(t0), "dur": max(0.0, (t2 - t0) / 1e3), "args": {"inst": seq}})
            else:
                if t1 > t0:
                    ev.append({"name": "wait " + name, "ph": "X", "pid": sm, "tid": "compute",
                               "ts": base(t0), "dur": (t1 - t0) / 1e3, "args": {"inst": seq}})
                ev.append({"name": name, "ph": "X", "pid": sm, "tid": "compute",
                           "ts": base(t1), "dur": max(0.0, (t2 - t1) / 1e3), "args": {"inst": seq}})
        return ev

    def save_chrome(self, path):
        with open(path, "w") as f:
            json.dump({"traceEvents": self.chrome_events(), "displayTimeUnit": "ns"}, f)
        return path
