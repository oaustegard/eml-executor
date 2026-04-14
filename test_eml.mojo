"""Comprehensive EML test suite.

Tests the full bootstrap chain from Odrzywolek (2026):
  eml(x, y) = exp(x) - ln(y)  with constant 1

Tiers:
  1 — Raw eml_op correctness
  2 — Bootstrap chain: constants + unary + binary
  3 — Domain sweeps matching the paper's numpy suite
  4 — RPN executor integration
"""

from std.math import exp, log, cos, sin, sqrt, abs, cosh, sinh, tanh, acos, asin, atan
from std.time import perf_counter_ns


# ─── Complex64 ─────────────────────────────────────────────────

def _atan2(y: Float64, x: Float64) -> Float64:
    var pi: Float64 = 3.141592653589793
    if x > 0: return _atan(y / x)
    elif x < 0:
        if y >= 0: return _atan(y / x) + pi
        else: return _atan(y / x) - pi
    else:
        if y > 0: return pi / 2.0
        elif y < 0: return -(pi / 2.0)
        else: return 0.0

def _atan(x: Float64) -> Float64:
    var pi_2: Float64 = 1.5707963267948966
    if x > 1.0: return pi_2 - _atan_core(1.0 / x)
    elif x < -1.0: return -pi_2 - _atan_core(1.0 / x)
    else: return _atan_core(x)

def _atan_core(x: Float64) -> Float64:
    var x2 = x * x
    return x * (0.9998660 + x2 * (-0.3302995 + x2 * (0.1801410 + x2 * (-0.0851330 + x2 * 0.0208351))))


struct Z(Writable, Copyable, Movable, ImplicitlyCopyable):
    """Complex number."""
    var re: Float64
    var im: Float64
    def __init__(out self, re: Float64, im: Float64 = 0.0):
        self.re = re; self.im = im
    def __add__(self, o: Self) -> Self: return Self(self.re + o.re, self.im + o.im)
    def __sub__(self, o: Self) -> Self: return Self(self.re - o.re, self.im - o.im)
    def __mul__(self, o: Self) -> Self:
        return Self(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    def __neg__(self) -> Self: return Self(-self.re, -self.im)
    def mag(self) -> Float64: return sqrt(self.re * self.re + self.im * self.im)
    def write_to(self, mut writer: Some[Writer]):
        if abs(self.im) < 1e-12: writer.write(self.re)
        elif abs(self.re) < 1e-12: writer.write(self.im, "i")
        elif self.im >= 0: writer.write(self.re, "+", self.im, "i")
        else: writer.write(self.re, self.im, "i")


def zr(x: Float64) -> Z:
    return Z(x, 0.0)

def z_one() -> Z:
    return Z(1.0, 0.0)


# ─── Complex elementary ops ────────────────────────────────────

def c_exp(z: Z) -> Z:
    var ea = exp(z.re)
    return Z(ea * cos(z.im), ea * sin(z.im))

def c_ln(z: Z) -> Z:
    """Lower-edge principal branch (paper convention)."""
    var PI: Float64 = 3.141592653589793
    if abs(z.im) < 1e-15 and z.re < 0:
        return Z(log(-z.re), -PI)
    var m = z.mag()
    if m < 1e-300: return Z(-1e308, 0.0)
    return Z(log(m), _atan2(z.im, z.re))


# ─── EML operator ──────────────────────────────────────────────

def eml(x: Z, y: Z) -> Z:
    """EML(x,y) = exp(x) - ln(y)."""
    return c_exp(x) - c_ln(y)


# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP CHAIN (from verify_eml_symbolic_chain.wl)
# ═══════════════════════════════════════════════════════════════

def eml_e() -> Z:         return eml(z_one(), z_one())
def eml_exp(x: Z) -> Z:   return eml(x, z_one())
def eml_ln(x: Z) -> Z:    return eml(z_one(), eml_exp(eml(z_one(), x)))
def eml_sub(a: Z, b: Z) -> Z: return eml(eml_ln(a), eml_exp(b))
def eml_zero() -> Z:      return eml_ln(z_one())
def eml_neg1() -> Z:      return eml_sub(eml_ln(z_one()), z_one())
def eml_two() -> Z:       return eml_sub(z_one(), eml_neg1())
def eml_minus(x: Z) -> Z: return eml_sub(eml_zero(), x)
def eml_plus(a: Z, b: Z) -> Z: return eml_sub(a, eml_minus(b))
def eml_inv(x: Z) -> Z:   return eml_exp(eml_minus(eml_ln(x)))
def eml_times(a: Z, b: Z) -> Z: return eml_exp(eml_plus(eml_ln(a), eml_ln(b)))
def eml_sqr(x: Z) -> Z:   return eml_times(x, x)
def eml_div(a: Z, b: Z) -> Z: return eml_times(a, eml_inv(b))
def eml_half(x: Z) -> Z:  return eml_div(x, eml_two())
def eml_avg(a: Z, b: Z) -> Z: return eml_half(eml_plus(a, b))
def eml_sqrt(x: Z) -> Z:  return eml_exp(eml_half(eml_ln(x)))
def eml_pow(a: Z, b: Z) -> Z: return eml_exp(eml_times(b, eml_ln(a)))
def eml_logb(base: Z, x: Z) -> Z: return eml_div(eml_ln(x), eml_ln(base))
def eml_hypot(a: Z, b: Z) -> Z: return eml_sqrt(eml_plus(eml_sqr(a), eml_sqr(b)))

def eml_i() -> Z:         return eml_minus(eml_exp(eml_half(eml_ln(eml_neg1()))))
def eml_pi() -> Z:        return eml_times(eml_i(), eml_ln(eml_neg1()))

def eml_cosh(x: Z) -> Z:  return eml_avg(eml_exp(x), eml_exp(eml_minus(x)))
def eml_sinh(x: Z) -> Z:  return eml(x, eml_exp(eml_cosh(x)))
def eml_tanh(x: Z) -> Z:  return eml_div(eml_sinh(x), eml_cosh(x))
def eml_cos(x: Z) -> Z:   return eml_cosh(eml_div(x, eml_i()))
def eml_sin(x: Z) -> Z:   return eml_cos(eml_sub(x, eml_half(eml_pi())))
def eml_tan(x: Z) -> Z:   return eml_div(eml_sin(x), eml_cos(x))
def eml_sigma(x: Z) -> Z: return eml_inv(eml(eml_minus(x), eml_exp(eml_neg1())))

def eml_arcsinh(x: Z) -> Z: return eml_ln(eml_plus(x, eml_hypot(eml_neg1(), x)))
def eml_arccosh(x: Z) -> Z: return eml_arcsinh(eml_hypot(x, eml_sqrt(eml_neg1())))
def eml_arccos(x: Z) -> Z:  return eml_arccosh(eml_cos(eml_arccosh(x)))
def eml_arcsin(x: Z) -> Z:  return eml_sub(eml_half(eml_pi()), eml_arccos(x))
def eml_arctanh(x: Z) -> Z: return eml_arcsinh(eml_inv(eml_tan(eml_arccos(x))))
def eml_arctan(x: Z) -> Z:  return eml_arcsin(eml_tanh(eml_arcsinh(x)))


# ═══════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════

struct TestRunner:
    var total: Int
    var passed: Int
    var failed: Int
    var worst_err: Float64
    var worst_name: String

    def __init__(out self):
        self.total = 0; self.passed = 0; self.failed = 0
        self.worst_err = 0.0; self.worst_name = String("")

    def check(mut self, name: String, got: Z, expected: Z, tol: Float64, real_only: Bool):
        self.total += 1
        var err_re = abs(got.re - expected.re)
        var err_im = abs(got.im - expected.im)
        var err = err_re
        if not real_only:
            err = sqrt(err_re * err_re + err_im * err_im)
        elif err_im > tol * 100:
            print("  FAIL  ", name, " imag residual:", err_im)
            self.failed += 1
            return
        if err > tol:
            print("  FAIL  ", name, "  got:", got, " exp:", expected, " err:", err)
            self.failed += 1
        else:
            self.passed += 1
        if err > self.worst_err:
            self.worst_err = err
            self.worst_name = name

    def check_r(mut self, name: String, got: Z, expected: Float64, tol: Float64 = 1e-9):
        self.check(name, got, zr(expected), tol, True)

    def check_c(mut self, name: String, got: Z, expected: Z, tol: Float64 = 1e-9):
        self.check(name, got, expected, tol, False)

    def sweep_pass(mut self, name: String, worst: Float64, tol: Float64, n_valid: Int, n: Int, worst_x: Float64, rms: Float64):
        self.total += 1
        var status = String("PASS")
        if worst > tol:
            status = "FAIL"
            self.failed += 1
        else:
            self.passed += 1
        if worst > self.worst_err:
            self.worst_err = worst
            self.worst_name = "sweep:" + name
        print("  ", status, " ", name, "  valid=", n_valid, "/", n, "  worst=", worst, " @", worst_x, "  rms=", rms)

    def report(self):
        print()
        print("═══════════════════════════════════════════")
        print("  Total:", self.total, " Pass:", self.passed, " Fail:", self.failed)
        if self.worst_err > 0:
            print("  Worst:", self.worst_err, " (", self.worst_name, ")")
        print("═══════════════════════════════════════════")


# ═══════════════════════════════════════════════════════════════
# TIER 1: Raw eml_op
# ═══════════════════════════════════════════════════════════════

def test_tier1(mut t: TestRunner):
    print("\n━━━ Tier 1: Raw eml(x, y) ━━━")
    var PI: Float64 = 3.141592653589793
    var EV: Float64 = 2.718281828459045
    t.check_r("eml(0,1)=1", eml(zr(0), z_one()), 1.0)
    t.check_r("eml(1,1)=e", eml(z_one(), z_one()), EV)
    t.check_r("eml(0,e)=0", eml(zr(0), zr(EV)), 0.0)
    t.check_r("eml(2,1)=e²", eml(zr(2), z_one()), exp(Float64(2)))
    t.check_r("eml(0,0.5)=1+ln2", eml(zr(0), zr(0.5)), 1.0 + log(Float64(2)))
    var x: Float64 = 1.5
    t.check_r("eml(x,exp(x))=exp(x)-x", eml(zr(x), zr(exp(x))), exp(x) - x)
    t.check_c("eml(iπ,1)=-1", eml(Z(0, PI), z_one()), zr(-1.0))
    t.check_c("eml(0,-1)=1+iπ", eml(zr(0), zr(-1)), Z(1.0, PI))


# ═══════════════════════════════════════════════════════════════
# TIER 2: Bootstrap chain
# ═══════════════════════════════════════════════════════════════

def test_tier2_const(mut t: TestRunner):
    print("\n━━━ Tier 2: Constants ━━━")
    var EV: Float64 = 2.718281828459045
    var PI: Float64 = 3.141592653589793
    t.check_r("e", eml_e(), EV)
    t.check_r("0", eml_zero(), 0.0)
    t.check_r("-1", eml_neg1(), -1.0)
    t.check_r("2", eml_two(), 2.0)
    t.check_c("i", eml_i(), Z(0, 1), tol=1e-8)
    t.check_r("π", eml_pi(), PI, tol=1e-6)

def test_tier2_unary(mut t: TestRunner):
    print("\n━━━ Tier 2: Unary functions ━━━")
    # Transcendental test points (algebraically independent)
    var g: Float64 = 0.5772156649015329   # Euler-Mascheroni γ
    var a: Float64 = 1.2824271291006226   # Glaisher A
    var zg = zr(g); var za = zr(a)
    t.check_r("exp(γ)", eml_exp(zg), exp(g))
    t.check_r("exp(A)", eml_exp(za), exp(a))
    t.check_r("ln(γ)", eml_ln(zg), log(g))
    t.check_r("ln(A)", eml_ln(za), log(a))
    t.check_r("-γ", eml_minus(zg), -g)
    t.check_r("-A", eml_minus(za), -a)
    t.check_r("1/γ", eml_inv(zg), 1.0/g)
    t.check_r("1/A", eml_inv(za), 1.0/a)
    t.check_r("γ/2", eml_half(zg), g/2.0)
    t.check_r("γ²", eml_sqr(zg), g*g)
    t.check_r("A²", eml_sqr(za), a*a)
    t.check_r("√γ", eml_sqrt(zg), sqrt(g), tol=1e-8)
    t.check_r("√A", eml_sqrt(za), sqrt(a), tol=1e-8)
    t.check_r("cosh(γ)", eml_cosh(zg), cosh(g))
    t.check_r("cosh(A)", eml_cosh(za), cosh(a))
    t.check_r("sinh(γ)", eml_sinh(zg), sinh(g))
    t.check_r("sinh(A)", eml_sinh(za), sinh(a))
    t.check_r("tanh(γ)", eml_tanh(zg), tanh(g))
    t.check_r("tanh(A)", eml_tanh(za), tanh(a))
    t.check_r("cos(γ)", eml_cos(zg), cos(g), tol=1e-6)
    t.check_r("cos(A)", eml_cos(za), cos(a), tol=1e-6)
    t.check_r("sin(γ)", eml_sin(zg), sin(g), tol=1e-5)
    t.check_r("sin(A)", eml_sin(za), sin(a), tol=1e-5)
    t.check_r("tan(γ)", eml_tan(zg), sin(g)/cos(g), tol=1e-5)
    t.check_r("σ(γ)", eml_sigma(zg), 1.0/(1.0+exp(-g)), tol=1e-8)
    t.check_r("σ(A)", eml_sigma(za), 1.0/(1.0+exp(-a)), tol=1e-8)
    t.check_r("arcsinh(γ)", eml_arcsinh(zg), log(g + sqrt(1.0+g*g)), tol=1e-7)
    t.check_r("arcsinh(A)", eml_arcsinh(za), log(a + sqrt(1.0+a*a)), tol=1e-7)
    t.check_r("arccosh(A)", eml_arccosh(za), log(a + sqrt(a*a - 1.0)), tol=1e-5)
    t.check_r("arccos(γ)", eml_arccos(zg), acos(g), tol=1e-4)
    t.check_r("arcsin(γ)", eml_arcsin(zg), asin(g), tol=1e-4)
    var ref_atanh = 0.5 * log((1.0+g)/(1.0-g))
    t.check_r("arctanh(γ)", eml_arctanh(zg), ref_atanh, tol=1e-4)
    t.check_r("arctan(γ)", eml_arctan(zg), atan(g), tol=1e-4)
    t.check_r("arctan(A)", eml_arctan(za), atan(a), tol=1e-4)

def test_tier2_binary(mut t: TestRunner):
    print("\n━━━ Tier 2: Binary operations ━━━")
    var g: Float64 = 0.5772156649015329
    var a: Float64 = 1.2824271291006226
    var zg = zr(g); var za = zr(a)
    t.check_r("γ-A", eml_sub(zg, za), g - a)
    t.check_r("A-γ", eml_sub(za, zg), a - g)
    t.check_r("γ+A", eml_plus(zg, za), g + a)
    t.check_r("γ×A", eml_times(zg, za), g * a)
    t.check_r("γ/A", eml_div(zg, za), g / a)
    t.check_r("A/γ", eml_div(za, zg), a / g)
    t.check_r("γ^A", eml_pow(zg, za), exp(a * log(g)), tol=1e-7)
    t.check_r("A^γ", eml_pow(za, zg), exp(g * log(a)), tol=1e-7)
    t.check_r("log_A(γ)", eml_logb(za, zg), log(g)/log(a))
    t.check_r("avg(γ,A)", eml_avg(zg, za), (g+a)/2.0)
    t.check_r("hypot(γ,A)", eml_hypot(zg, za), sqrt(g*g+a*a), tol=1e-7)

def test_identities(mut t: TestRunner):
    print("\n━━━ Identities ━━━")
    var g: Float64 = 0.5772156649015329
    var zg = zr(g)
    t.check_r("exp(ln(γ))=γ", eml_exp(eml_ln(zg)), g)
    t.check_r("ln(exp(γ))=γ", eml_ln(eml_exp(zg)), g)
    t.check_r("--γ=γ", eml_minus(eml_minus(zg)), g)
    t.check_r("1/(1/γ)=γ", eml_inv(eml_inv(zg)), g)
    t.check_r("γ+0=γ", eml_plus(zg, eml_zero()), g)
    t.check_r("γ-γ=0", eml_sub(zg, zg), 0.0)
    t.check_r("γ×1=γ", eml_times(zg, z_one()), g)
    t.check_r("γ/γ=1", eml_div(zg, zg), 1.0)
    t.check_r("γ^1=γ", eml_pow(zg, z_one()), g, tol=1e-8)
    t.check_r("γ^0=1", eml_pow(zg, eml_zero()), 1.0, tol=1e-8)
    t.check_r("(√γ)²=γ", eml_sqr(eml_sqrt(zg)), g, tol=1e-7)
    t.check_r("cosh²-sinh²=1", eml_sub(eml_sqr(eml_cosh(zg)), eml_sqr(eml_sinh(zg))), 1.0, tol=1e-7)
    t.check_r("sin²+cos²=1", eml_plus(eml_sqr(eml_sin(zg)), eml_sqr(eml_cos(zg))), 1.0, tol=1e-3)
    # Euler: Re(e^iπ)+1=0
    var eipi = eml_exp(Z(0, 3.141592653589793))
    t.check_r("Re(e^iπ)+1=0", eml_plus(eipi, z_one()), 0.0, tol=1e-9)


# ═══════════════════════════════════════════════════════════════
# TIER 3: Domain sweeps
# ═══════════════════════════════════════════════════════════════

def sweep(mut t: TestRunner, name: String, x_min: Float64, x_max: Float64, step: Float64, tol: Float64):
    var n = 0
    var n_valid = 0
    var worst: Float64 = 0.0
    var worst_x: Float64 = 0.0
    var sum_sq: Float64 = 0.0
    var x = x_min
    while x <= x_max + step * 0.1:
        n += 1
        var xc = zr(x)
        var got = zr(0.0)
        var expected = 0.0
        var skip = False

        if name == "exp":    got = eml_exp(xc);   expected = exp(x)
        elif name == "ln":   got = eml_ln(xc);    expected = log(x)
        elif name == "minus": got = eml_minus(xc); expected = -x
        elif name == "inv":  got = eml_inv(xc);   expected = 1.0/x
        elif name == "half": got = eml_half(xc);   expected = x/2.0
        elif name == "sqr":  got = eml_sqr(xc);   expected = x*x
        elif name == "sqrt": got = eml_sqrt(xc);   expected = sqrt(x)
        elif name == "cosh": got = eml_cosh(xc);   expected = cosh(x)
        elif name == "sinh": got = eml_sinh(xc);   expected = sinh(x)
        elif name == "tanh": got = eml_tanh(xc);   expected = tanh(x)
        elif name == "cos":  got = eml_cos(xc);    expected = cos(x)
        elif name == "sin":  got = eml_sin(xc);    expected = sin(x)
        elif name == "tan":  got = eml_tan(xc);    expected = sin(x)/cos(x)
        elif name == "sigma": got = eml_sigma(xc); expected = 1.0/(1.0+exp(-x))
        elif name == "arcsinh": got = eml_arcsinh(xc); expected = log(x + sqrt(1.0+x*x))
        elif name == "arccosh": got = eml_arccosh(xc); expected = log(x + sqrt(x*x - 1.0))
        elif name == "arcsin":  got = eml_arcsin(xc);  expected = asin(x)
        elif name == "arccos":  got = eml_arccos(xc);  expected = acos(x)
        elif name == "arctan":  got = eml_arctan(xc);  expected = atan(x)
        elif name == "arctanh": got = eml_arctanh(xc); expected = 0.5*log((1.0+x)/(1.0-x))
        else: skip = True

        if not skip:
            # NaN/inf guard
            if expected != expected or expected > 1e300 or expected < -1e300 or got.re != got.re or got.re > 1e300 or got.re < -1e300:
                skip = True

        if not skip:
            n_valid += 1
            var err = abs(got.re - expected)
            sum_sq += err * err
            if err > worst:
                worst = err
                worst_x = x
        x += step

    var rms: Float64 = 0.0
    if n_valid > 0:
        rms = sqrt(sum_sq / Float64(n_valid))
    t.sweep_pass(name, worst, tol, n_valid, n, worst_x, rms)

def test_tier3(mut t: TestRunner):
    print("\n━━━ Tier 3: Domain sweeps ━━━")
    # Ranges from run_unary_suite_numpy.py
    sweep(t, "exp",   -4.0,  4.0,  0.125, 1e-9)
    sweep(t, "ln",     0.125, 8.0,  0.125, 1e-9)
    sweep(t, "minus", -4.0,  4.0,  0.125, 1e-9)
    sweep(t, "inv",    0.125, 8.0,  0.125, 1e-8)
    sweep(t, "half",  -4.0,  4.0,  0.125, 1e-9)
    sweep(t, "sqr",   -4.0,  4.0,  0.125, 1e-7)
    sweep(t, "sqrt",   0.125, 8.0,  0.125, 1e-8)
    sweep(t, "cosh",  -4.0,  4.0,  0.125, 1e-9)
    sweep(t, "sinh",  -4.0,  4.0,  0.125, 1e-9)
    sweep(t, "tanh",  -4.0,  4.0,  0.125, 1e-8)
    sweep(t, "sigma", -8.0,  8.0,  0.125, 1e-8)
    sweep(t, "cos",   -6.0,  6.0,  0.0625, 1e-4)
    sweep(t, "sin",   -6.0,  6.0,  0.0625, 1e-4)
    sweep(t, "tan",   -1.25, 1.25, 0.03125, 1e-3)
    sweep(t, "arcsinh", -8.0, 8.0, 0.125, 1e-6)
    sweep(t, "arccosh",  1.125, 9.0, 0.125, 1e-4)
    sweep(t, "arcsin", -0.95, 0.95, 0.01, 1e-3)
    sweep(t, "arccos", -0.95, 0.95, 0.01, 1e-3)
    sweep(t, "arctan", -4.0, 4.0, 0.125, 1e-3)
    sweep(t, "arctanh", -0.9, 0.9, 0.01, 1e-3)


# ═══════════════════════════════════════════════════════════════
# TIER 4: RPN executor
# ═══════════════════════════════════════════════════════════════

comptime OP_HALT = 0
comptime OP_PUSH = 1
comptime OP_EML  = 2

struct Inst(Copyable, Movable):
    var op: Int; var re: Float64; var im: Float64
    def __init__(out self, op: Int, re: Float64 = 0.0, im: Float64 = 0.0):
        self.op = op; self.re = re; self.im = im

struct PStack:
    var k0: List[Float64]; var k1: List[Float64]
    var vr: List[Float64]; var vi: List[Float64]
    var sp: Int; var cnt: Int
    def __init__(out self):
        self.k0 = List[Float64](); self.k1 = List[Float64]()
        self.vr = List[Float64](); self.vi = List[Float64]()
        self.sp = 0; self.cnt = 0
    def push(mut self, v: Z):
        self.sp += 1
        var a = Float64(self.sp)
        self.k0.append(2.0*a); self.k1.append(-(a*a))
        self.vr.append(v.re); self.vi.append(v.im)
        self.cnt += 1
    def attn(self, addr: Int) -> Z:
        var q = Float64(addr)
        var best = -1e308; var bi = 0
        for i in range(self.cnt):
            var s = self.k0[i]*q + self.k1[i]
            if s >= best: best = s; bi = i
        return Z(self.vr[bi], self.vi[bi])
    def pop2(mut self) -> Tuple[Z, Z]:
        var top = self.attn(self.sp)
        var sec = self.attn(self.sp - 1)
        self.sp -= 2
        return (sec^, top^)

def run_rpn(prog: List[Inst]) -> Z:
    var st = PStack()
    var ip = 0; var steps = 0
    # Build program memory
    var pk0 = List[Float64](); var pk1 = List[Float64]()
    var pop = List[Int](); var par = List[Float64](); var pai = List[Float64]()
    for i in range(len(prog)):
        var a = Float64(i)
        pk0.append(2.0*a); pk1.append(-(a*a))
        pop.append(prog[i].op); par.append(prog[i].re); pai.append(prog[i].im)
    while steps < 10000:
        # Fetch via attention
        var q = Float64(ip); var best = -1e308; var bi = 0
        for i in range(len(pop)):
            var s = pk0[i]*q + pk1[i]
            if s >= best: best = s; bi = i
        var opc = pop[bi]
        if opc == OP_HALT: break
        if opc == OP_PUSH:
            st.push(Z(par[bi], pai[bi]))
        elif opc == OP_EML:
            var p = st.pop2()
            st.push(eml(p[0], p[1]))
        ip += 1; steps += 1
    return st.attn(st.sp)

def P(v: Float64) -> Inst: return Inst(OP_PUSH, v)
def Pc(r: Float64, i: Float64) -> Inst: return Inst(OP_PUSH, r, i)
def Em() -> Inst: return Inst(OP_EML)
def Ht() -> Inst: return Inst(OP_HALT)

def test_tier4(mut t: TestRunner):
    print("\n━━━ Tier 4: RPN executor ━━━")
    var EV: Float64 = 2.718281828459045
    var PI: Float64 = 3.141592653589793

    # e = eml(1,1)   K=3
    t.check_r("RPN:e", run_rpn([P(1), P(1), Em(), Ht()]), EV)

    # exp(x) = eml(x,1)   K=3
    t.check_r("RPN:exp(2)", run_rpn([P(2), P(1), Em(), Ht()]), exp(Float64(2)))
    t.check_r("RPN:exp(-1)", run_rpn([P(-1), P(1), Em(), Ht()]), exp(Float64(-1)))
    t.check_r("RPN:exp(0)=1", run_rpn([P(0), P(1), Em(), Ht()]), 1.0)

    # ln(x) = 1 1 x E 1 E E   K=7
    for v in [0.5, 1.0, 2.0, 5.0, 10.0, 0.1]:
        t.check_r("RPN:ln("+String(v)+")", run_rpn([P(1),P(1),P(v),Em(),P(1),Em(),Em(),Ht()]), log(v))

    # 0 = ln(1) = 1 1 1 E 1 E E   K=7
    t.check_r("RPN:zero", run_rpn([P(1),P(1),P(1),Em(),P(1),Em(),Em(),Ht()]), 0.0)

    # Euler: eml(iπ,1) = exp(iπ) = -1
    t.check_c("RPN:Euler", run_rpn([Pc(0,PI), P(1), Em(), Ht()]), zr(-1))

    # Round-trip: exp(ln(x)) = x   RPN: 1 1 x E 1 E E 1 E   (ln then exp)
    for v in [0.5, 2.0, 3.7]:
        t.check_r("RPN:exp(ln("+String(v)+"))", run_rpn([P(1),P(1),P(v),Em(),P(1),Em(),Em(),P(1),Em(),Ht()]), v)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  EML Test Suite — Full Bootstrap Chain Verification  ║")
    print("║  7 constants · 20 unary · 8 binary · domain sweeps  ║")
    print("╚═══════════════════════════════════════════════════════╝")
    var start = perf_counter_ns()
    var t = TestRunner()
    test_tier1(t)
    test_tier2_const(t)
    test_tier2_unary(t)
    test_tier2_binary(t)
    test_identities(t)
    test_tier4(t)
    test_tier3(t)
    var ms = (perf_counter_ns() - start) // 1_000_000
    t.report()
    print("  Elapsed:", ms, "ms")
