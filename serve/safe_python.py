"""Restricted evaluator for the model's `python` tool calls (the default; no subprocess, no imports beyond a whitelist).

Vajram was trained to call `python` for calculations (arithmetic, percentages, GST, interest, EMI, dates, unit
conversion, statistics). This module runs that kind of code by walking the AST itself: only whitelisted syntax,
functions, modules and attributes; no names or attributes starting with "_"; budgets on steps, wall time, number
size, sequence length and output. Anything else is refused with an error the model can read.

It is NOT a general Python sandbox and does not try to be one. Arbitrary Python (a real subprocess) is available only
with VAJRAM_UNSAFE_PYTHON=1 (see tools.py), and that is unsafe on any machine you care about.
"""
from __future__ import annotations

import ast
import datetime as _dt
import math
import operator
import statistics
import time

MAX_STEPS = 200_000
MAX_SECONDS = 3.0
MAX_INT_BITS = 4096
MAX_SEQ = 100_000
MAX_OUTPUT = 2000
MAX_DEPTH = 40


class Refused(Exception):
    pass


def _public(mod, names=None):
    names = names or [n for n in dir(mod) if not n.startswith("_")]
    return {n: getattr(mod, n) for n in names if hasattr(mod, n)}


def _capped(f, limit):
    def g(*a):
        if any(isinstance(x, int) and x > limit for x in a):
            raise Refused(f"{f.__name__}() argument too large (max {limit})")
        return f(*a)
    g.__name__ = f.__name__
    return g


MODULES = {
    "math": {**_public(math), "factorial": _capped(math.factorial, 1000), "comb": _capped(math.comb, 10000),
             "perm": _capped(math.perm, 10000)},
    "statistics": _public(statistics, ["mean", "median", "mode", "stdev", "pstdev", "variance", "pvariance",
                                       "fmean", "geometric_mean", "harmonic_mean", "median_low", "median_high"]),
    "datetime": {"date": _dt.date, "datetime": _dt.datetime, "timedelta": _dt.timedelta, "time": _dt.time},
}
# attributes allowed on values (not modules), by type
SAFE_ATTRS = {
    _dt.date: {"year", "month", "day", "weekday", "isoweekday", "isoformat", "strftime", "replace", "today",
               "toordinal", "fromisoformat"},
    _dt.datetime: {"year", "month", "day", "hour", "minute", "second", "weekday", "isoweekday", "isoformat",
                   "strftime", "replace", "date", "now", "today", "fromisoformat", "toordinal"},
    _dt.timedelta: {"days", "seconds", "total_seconds"},
    # no str.format (its field syntax can walk attributes), no padding / replace (memory bombs)
    str: {"upper", "lower", "strip", "lstrip", "rstrip", "title", "capitalize", "join", "split",
          "startswith", "endswith", "count", "find"},
    list: {"append", "extend", "insert", "pop", "index", "count", "sort", "reverse"},
    dict: {"get", "keys", "values", "items"},
    float: {"is_integer"},
    int: set(),
    tuple: {"index", "count"},
}


def _range(*a):
    r = range(*a)
    if len(r) > MAX_SEQ:
        raise Refused(f"range too long (max {MAX_SEQ})")
    return r


def _pow(a, b, *m):
    _check_pow(a, b)
    return pow(a, b, *m)


def _check_pow(a, b):
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        if isinstance(a, int) and isinstance(b, int) and abs(a) > 1 and b > 0:
            if b * max(1, abs(a).bit_length()) > MAX_INT_BITS:
                raise Refused("number too large")


def _check_spec(spec):
    spec = str(spec)
    if any(len(d) > 3 for d in "".join(c if c.isdigit() else " " for c in spec).split()):
        raise Refused("format width too large")


def _format(v, spec=""):
    _check_spec(spec)
    return format(v, spec)


BUILTINS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len, "range": _range, "int": int,
    "float": float, "str": str, "bool": bool, "list": list, "tuple": tuple, "dict": dict, "sorted": sorted,
    "enumerate": enumerate, "zip": zip, "pow": _pow, "divmod": divmod, "any": any, "all": all, "reversed": reversed,
    "format": _format, "True": True, "False": False, "None": None,
}

BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
UNOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
CMPOPS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
          ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b,
          ast.NotIn: lambda a, b: a not in b, ast.Is: operator.is_, ast.IsNot: operator.is_not}


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Func:
    def __init__(self, node, interp):
        self.node, self.interp = node, interp

    def __call__(self, *args):
        a = self.node.args
        params = [p.arg for p in a.args]
        defaults = [self.interp.eval(d, self.interp.globals) for d in a.defaults]
        if len(args) > len(params):
            raise Refused(f"{self.node.name}() takes {len(params)} arguments")
        vals = list(args) + defaults[len(defaults) - (len(params) - len(args)):] if len(args) < len(params) else list(args)
        if len(vals) != len(params):
            raise Refused(f"{self.node.name}() missing arguments")
        self.interp.depth += 1
        if self.interp.depth > MAX_DEPTH:
            raise Refused("recursion too deep")
        local = dict(zip(params, vals))
        try:
            self.interp.exec_block(self.node.body, local)
        except _Return as r:
            return r.value
        finally:
            self.interp.depth -= 1
        return None


class Interp:
    def __init__(self):
        self.out: list[str] = []
        self.out_len = 0
        self.steps = 0
        self.t0 = time.monotonic()
        self.depth = 0
        self.globals: dict = {}

    # ------------------------------------------------------------ budget
    def tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise Refused("step limit reached")
        if self.steps % 1000 == 0 and time.monotonic() - self.t0 > MAX_SECONDS:
            raise Refused(f"time limit ({MAX_SECONDS:.0f} s) reached")

    @staticmethod
    def check_value(v):
        if isinstance(v, int) and not isinstance(v, bool) and v.bit_length() > MAX_INT_BITS:
            raise Refused("number too large")
        if isinstance(v, (str, list, tuple, dict)) and len(v) > MAX_SEQ:
            raise Refused("value too long")
        return v

    def print_(self, *args, sep=" ", end="\n"):
        s = str(sep).join(str(a) for a in args) + str(end)
        self.out_len += len(s)
        if self.out_len > MAX_OUTPUT * 4:
            raise Refused("too much output")
        self.out.append(s)

    # ------------------------------------------------------------ names
    @staticmethod
    def check_name(n):
        if n.startswith("_"):
            raise Refused(f"name not allowed: {n}")

    def lookup(self, name, local):
        self.check_name(name)
        if local is not None and name in local:
            return local[name]
        if name in self.globals:
            return self.globals[name]
        if name == "print":
            return self.print_
        if name in BUILTINS:
            return BUILTINS[name]
        raise Refused(f"name '{name}' is not defined or not allowed")

    def assign(self, target, value, local):
        scope = local if local is not None else self.globals
        if isinstance(target, ast.Name):
            self.check_name(target.id)
            scope[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            vals = list(value)
            if len(vals) != len(target.elts):
                raise Refused("unpacking mismatch")
            for t, v in zip(target.elts, vals):
                self.assign(t, v, local)
        elif isinstance(target, ast.Subscript):
            obj = self.eval(target.value, local)
            if not isinstance(obj, (list, dict)):
                raise Refused("item assignment only on lists and dicts")
            obj[self.eval(target.slice, local)] = value
            self.check_value(obj)
        else:
            raise Refused(f"assignment target not allowed: {type(target).__name__}")

    # ------------------------------------------------------------ statements
    def exec_block(self, body, local=None):
        for st in body:
            self.exec(st, local)

    def exec(self, node, local=None):
        self.tick()
        t = type(node)
        if t is ast.Expr:
            self.eval(node.value, local)
        elif t is ast.Assign:
            v = self.eval(node.value, local)
            for tg in node.targets:
                self.assign(tg, v, local)
        elif t is ast.AnnAssign:
            if node.value is not None:
                self.assign(node.target, self.eval(node.value, local), local)
        elif t is ast.AugAssign:
            cur = self.eval(ast.Name(node.target.id, ast.Load()) if isinstance(node.target, ast.Name)
                            else node.target, local)
            v = self.binop(node.op, cur, self.eval(node.value, local))
            self.assign(node.target, v, local)
        elif t is ast.If:
            self.exec_block(node.body if self.eval(node.test, local) else node.orelse, local)
        elif t is ast.For:
            it = self.eval(node.iter, local)
            broke = False
            for v in it:
                self.tick()
                self.assign(node.target, v, local)
                try:
                    self.exec_block(node.body, local)
                except _Break:
                    broke = True
                    break
                except _Continue:
                    continue
            if not broke:
                self.exec_block(node.orelse, local)
        elif t is ast.While:
            broke = False
            while self.eval(node.test, local):
                self.tick()
                try:
                    self.exec_block(node.body, local)
                except _Break:
                    broke = True
                    break
                except _Continue:
                    continue
            if not broke:
                self.exec_block(node.orelse, local)
        elif t is ast.Break:
            raise _Break()
        elif t is ast.Continue:
            raise _Continue()
        elif t is ast.Pass:
            pass
        elif t is ast.Return:
            raise _Return(self.eval(node.value, local) if node.value is not None else None)
        elif t is ast.FunctionDef:
            self.check_name(node.name)
            if node.decorator_list or node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
                raise Refused("only plain functions with positional arguments are allowed")
            (local if local is not None else self.globals)[node.name] = _Func(node, self)
        elif t is ast.Import:
            for a in node.names:
                if a.name not in MODULES:
                    raise Refused(f"import of '{a.name}' is not allowed (allowed: {', '.join(MODULES)})")
                self.assign(ast.Name(a.asname or a.name, ast.Store()), _Module(a.name), local)
        elif t is ast.ImportFrom:
            if node.module not in MODULES:
                raise Refused(f"import from '{node.module}' is not allowed (allowed: {', '.join(MODULES)})")
            for a in node.names:
                if a.name == "*" or a.name not in MODULES[node.module]:
                    raise Refused(f"'{a.name}' from {node.module} is not allowed")
                self.assign(ast.Name(a.asname or a.name, ast.Store()), MODULES[node.module][a.name], local)
        else:
            raise Refused(f"statement not allowed: {t.__name__}")

    # ------------------------------------------------------------ expressions
    def binop(self, op, a, b):
        f = BINOPS.get(type(op))
        if f is None:
            raise Refused(f"operator not allowed: {type(op).__name__}")
        if isinstance(op, ast.Pow):
            _check_pow(a, b)
        if isinstance(op, ast.Mult):
            for x, y in ((a, b), (b, a)):
                if isinstance(x, (str, list, tuple)) and isinstance(y, int) and len(x) * y > MAX_SEQ:
                    raise Refused("value too long")
            if isinstance(a, int) and isinstance(b, int) and a.bit_length() + b.bit_length() > MAX_INT_BITS:
                raise Refused("number too large")
        if isinstance(op, ast.Mod) and isinstance(a, str):
            raise Refused("%-formatting of strings is not allowed; use an f-string")
        return self.check_value(f(a, b))

    def eval(self, node, local=None):
        self.tick()
        t = type(node)
        if t is ast.Constant:
            if not isinstance(node.value, (int, float, str, bool, type(None), complex)):
                raise Refused("constant not allowed")
            return node.value
        if t is ast.Name:
            return self.lookup(node.id, local)
        if t is ast.BinOp:
            return self.binop(node.op, self.eval(node.left, local), self.eval(node.right, local))
        if t is ast.UnaryOp:
            f = UNOPS.get(type(node.op))
            if f is None:
                raise Refused("operator not allowed")
            return f(self.eval(node.operand, local))
        if t is ast.BoolOp:
            if isinstance(node.op, ast.And):
                v = True
                for x in node.values:
                    v = self.eval(x, local)
                    if not v:
                        return v
                return v
            v = False
            for x in node.values:
                v = self.eval(x, local)
                if v:
                    return v
            return v
        if t is ast.Compare:
            left = self.eval(node.left, local)
            for op, c in zip(node.ops, node.comparators):
                right = self.eval(c, local)
                if not CMPOPS[type(op)](left, right):
                    return False
                left = right
            return True
        if t is ast.IfExp:
            return self.eval(node.body if self.eval(node.test, local) else node.orelse, local)
        if t is ast.Call:
            f = self.eval(node.func, local)
            if not (f in BUILTINS.values() or f == self.print_ or isinstance(f, (_Func, _Bound))
                    or _is_module_value(f)):
                raise Refused("call not allowed")
            args = []
            for a in node.args:
                if isinstance(a, ast.Starred):
                    args.extend(self.eval(a.value, local))
                else:
                    args.append(self.eval(a, local))
            kw = {}
            for k in node.keywords:
                if k.arg is None or k.arg.startswith("_"):
                    raise Refused("keyword not allowed")
                kw[k.arg] = self.eval(k.value, local)
            if isinstance(f, _Func) and kw:
                raise Refused("keyword arguments to your own functions are not supported")
            return self.check_value(f(*args, **kw))
        if t is ast.Attribute:
            self.check_name(node.attr)
            obj = self.eval(node.value, local)
            if isinstance(obj, _Module):
                if node.attr not in MODULES[obj.name]:
                    raise Refused(f"{obj.name}.{node.attr} is not allowed")
                return MODULES[obj.name][node.attr]
            for typ, attrs in SAFE_ATTRS.items():
                if (obj is typ or type(obj) is typ) and node.attr in attrs:
                    v = getattr(obj, node.attr)
                    return _Bound(v, obj, node.attr) if callable(v) else v
            raise Refused(f"attribute '{node.attr}' is not allowed here")
        if t is ast.Subscript:
            obj = self.eval(node.value, local)
            return self.check_value(obj[self.eval(node.slice, local)])
        if t is ast.Slice:
            return slice(*(self.eval(x, local) if x is not None else None for x in (node.lower, node.upper, node.step)))
        if t in (ast.List, ast.Tuple, ast.Set):
            vals = []
            for e in node.elts:
                if isinstance(e, ast.Starred):
                    vals.extend(self.eval(e.value, local))
                else:
                    vals.append(self.eval(e, local))
            return self.check_value({ast.List: list, ast.Tuple: tuple, ast.Set: set}[t](vals))
        if t is ast.Dict:
            if any(k is None for k in node.keys):
                raise Refused("dict unpacking not allowed")
            return {self.eval(k, local): self.eval(v, local) for k, v in zip(node.keys, node.values)}
        if t in (ast.ListComp, ast.GeneratorExp, ast.SetComp):
            vals = self.comp(node.generators, lambda sc: self.eval(node.elt, sc), local)
            return set(vals) if t is ast.SetComp else vals
        if t is ast.DictComp:
            pairs = self.comp(node.generators, lambda sc: (self.eval(node.key, sc), self.eval(node.value, sc)), local)
            return dict(pairs)
        if t is ast.JoinedStr:
            return "".join(str(self.eval(v, local)) for v in node.values)
        if t is ast.FormattedValue:
            v = self.eval(node.value, local)
            if node.conversion == ord("r"):
                v = repr(v)
            elif node.conversion == ord("s"):
                v = str(v)
            spec = self.eval(node.format_spec, local) if node.format_spec is not None else ""
            return _format(v, spec)
        raise Refused(f"expression not allowed: {t.__name__}")

    def comp(self, gens, elt, local):
        scope = dict(local or {})
        out = []

        def rec(i):
            if i == len(gens):
                out.append(elt(scope))
                if len(out) > MAX_SEQ:
                    raise Refused("value too long")
                return
            g = gens[i]
            if g.is_async:
                raise Refused("async not allowed")
            for v in self.eval(g.iter, scope):
                self.tick()
                self.assign(g.target, v, scope)
                if all(self.eval(c, scope) for c in g.ifs):
                    rec(i + 1)
        rec(0)
        return out


class _Module:
    def __init__(self, name):
        self.name = name


class _Bound:
    """A whitelisted method of a safe value."""

    def __init__(self, f, owner=None, name=""):
        self.f, self.owner, self.name = f, owner, name

    def __call__(self, *a, **k):
        if isinstance(self.owner, list) and self.name == "extend" and a:
            a = (list(a[0]),)
            if len(self.owner) + len(a[0]) > MAX_SEQ:
                raise Refused("value too long")
        if isinstance(self.owner, list) and self.name in ("append", "insert") and len(self.owner) >= MAX_SEQ:
            raise Refused("value too long")
        if self.name == "strftime" and a and len(str(a[0])) > 200:
            raise Refused("format too long")
        return self.f(*a, **k)


_MODULE_VALUES = {id(v) for m in MODULES.values() for v in m.values()}


def _is_module_value(f):
    return id(f) in _MODULE_VALUES


def run(code: str) -> dict:
    """Run restricted Python. Returns {"output": text} or {"error": message}, like the subprocess tool did."""
    code = str(code or "")
    if len(code) > 20_000:
        return {"error": "code too long"}
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e.msg} (line {e.lineno})"}
    it = Interp()
    try:
        body = tree.body
        # like a REPL cell: a bare final expression is printed when nothing else was
        last_expr = body[-1] if body and isinstance(body[-1], ast.Expr) else None
        it.exec_block(body[:-1] if last_expr else body)
        if last_expr is not None:
            v = it.eval(last_expr.value)
            if v is not None and not it.out and not (isinstance(last_expr.value, ast.Call)
                                                     and getattr(last_expr.value.func, "id", "") == "print"):
                it.print_(v)
    except Refused as e:
        return {"error": f"not allowed in the safe calculator: {e}"}
    except (_Break, _Continue, _Return):
        return {"error": "break / continue / return outside a loop or function"}
    except RecursionError:
        return {"error": "not allowed in the safe calculator: nesting too deep"}
    except Exception as e:  # the model's own bug (ZeroDivisionError, TypeError, ...): show it like Python would
        out = "".join(it.out)
        return {"output": (out + f"{type(e).__name__}: {e}").strip()[-MAX_OUTPUT:]}
    out = "".join(it.out).strip()
    return {"output": out[-MAX_OUTPUT:] or "(no output)"}
