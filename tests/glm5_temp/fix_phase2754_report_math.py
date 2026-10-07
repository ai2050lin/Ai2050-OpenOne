"""Repair only draft report formula serialization before append/sealing."""
import re
from pathlib import Path

p = Path("tests/glm5/phase2754_delivery.py")
s = p.read_text(encoding="utf-8")
formulas = [
    r"""$$
C_S=\frac{1}{8}\sum_{\theta,\rho,\eta}\chi_S(\theta,\rho,\eta)F(\theta,\rho,\eta),
\qquad
\chi_S\in\{1,\theta,\rho,\eta,\theta\rho,\theta\eta,\rho\eta,\theta\rho\eta\}.
$$""",
    r"""$$
z_\ell=\left[
\frac{h_{\ell,\mathrm{end}}-\mu_0}{s_0},
\frac{h_{\ell,\mathrm{src}}-h_{\ell,\mathrm{dst}}-\mu_1}{s_1},
\frac{h_{\ell,\mathrm{src}}+h_{\ell,\mathrm{dst}}-\mu_2}{s_2}
\right],
\qquad
(\widehat r,\widehat y,\widehat m)=z_\ell B+b.
$$""",
    r"""$$
m=\frac{v^\top h_0+\sum_\ell v^\top A_\ell+\sum_\ell v^\top M_\ell+
\sum_\ell v^\top\epsilon_\ell}{s_L}
+\epsilon_{\mathrm{norm}}+\epsilon_{\mathrm{readout}}.
$$""",
    r"""$$
v^\top M_\ell\approx\sum_{j=0}^{9727}c_{\ell j}u_{\ell j},
\qquad
c_{\ell j}=v^\top W_{\mathrm{down},\ell}[:,j].
$$""",
    r"""$$
\mathcal T^D_{\ell,\tau}:
(\mathcal L(p),\mathbf W_\ell(p),\mathcal X_\ell(p))
\rightharpoonup\mathbf W_{\ell+1}(p),
\qquad
\operatorname{Atlas}_D=(G_{\mathrm{external}},G_{\mathrm{internal}},
E^D_{\mathrm{association}}).
$$""",
    r"""$$
r_\ell=H_\ell+A_\ell(N_\ell(H_\ell);KV_\ell,\mathrm{position},\mathrm{mask}),
\qquad x_\ell=N'_\ell(r_\ell),
$$""",
    r"""$$
H_{\ell+1}=r_\ell+W_{d,\ell}
[\operatorname{SiLU}(W_{g,\ell}x_\ell)\odot W_{u,\ell}x_\ell],
\qquad
p_{t+1}=\operatorname{softmax}(W_U N_f(H_L)_t).
$$""",
]
assert len(re.findall(r"\$\$.*?\$\$", s, re.S)) == len(formulas)
it = iter(formulas)
s = re.sub(r"\$\$.*?\$\$", lambda _: next(it), s, flags=re.S)
s = "\n".join(line for line in s.split("\n") if not line.strip().startswith("text=text.replace("))
assert not [(i, ord(c)) for i, c in enumerate(s) if ord(c) < 32 and c not in "\n\t"]
p.write_text(s, encoding="utf-8")
print("Replaced seven draft formulas; no measurement or prior memo changed.")
