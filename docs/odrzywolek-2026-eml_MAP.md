# All elementary functions from a single operator
*23 pages*

## Contents

- **§title** All elementary functions from a single operator (p.1–2) — The paper introduces a single binary operator eml(x,y) = exp(x) - ln(y) that, together with the constant 1, can generate all standard elementary functions, constants, and arithmetic operations, and demonstrates its use in gradient-based symbolic regression via binary trees.
    - **§summary-paragraph** Summary paragraph (p.2–3) — The section introduces the EML operator eml(x,y) = exp(x)ln(y) and the constant 1 as a universal primitive for all elementary functions, analogous to NAND gates in Boolean logic, and outlines its implications for symbolic regression and combinatorial structure.
    - **§significance** Significance statement (p.3–4) — A single binary operation eml(x,y) can replace all standard mathematical operations (arithmetic, trigonometric, logarithmic, etc.), meaning a calculator with just two buttons—EML and the digit 1—can compute everything a scientific calculator does, with implications for uniform circuit-like encoding of mathematical expressions.
  - **§1** 1 Introduction (p.4–6) — This section introduces the EML (Exp-Minus-Log) operator as a single binary primitive sufficient to express all standard real elementary functions, analogous to NAND for Boolean logic, requiring only the constant 1 as a terminal symbol.
  - **§2** 2 Methods (p.6–8) — The author describes an ablation-based search methodology to identify minimal operator sets for elementary functions, using hybrid numeric bootstrapping verification with transcendental constants and iterative list-transfer between known and target primitives.
  - **§3** 3 Results (p.8–10) — This section presents the progressive reduction of a 36-primitive scientific calculator to a single binary operator EML, tracing historical configurations (Calc 3 through Calc 0) and introducing EML and its cousins EDL and -EML, with verification methods described.
  - **§4** 4 Usage and applications (p.10) — This section introduces usage and applications of EML, beginning with the EML compiler subsection, noting that the uniform tree structure of EML expressions suggests several directions for implementation and application.
    - **§4.1** 4.1 EML compiler (p.10–11) — Describes a prototype EML compiler that converts mathematical formulas into pure EML form using only the Sheffer-stroke-like EML operation, discusses implementation details including complex branch issues, and notes compatibility with various numerical and symbolic computation environments.
    - **§4.2** 4.2 Elementary functions as binary trees and analog circuits (p.11–12) — EML notation naturally produces binary expression trees via a trivial context-free grammar, enabling uniform representation of elementary functions as both symbolic trees and analog circuits using the EML Sheffer operator as a basic building block.
    - **§4.3** 4.3 Symbolic Regression by continuous optimization (p.12–15) — This section presents a 'master formula' approach to symbolic regression using EML, constructing a parameterized binary expression tree that can represent any elementary function by setting discrete parameter values, and demonstrates continuous optimization to recover exact symbolic expressions from data.
  - **§5** 5 Conclusions and open questions (p.15–16) — The conclusions summarize EML as a sufficient primitive for elementary functions, discuss open questions about Sheffer-type operators without distinguished constants, the necessity of complex intermediates, and EML's advantages over conventional neural networks for interpretable symbolic computation.
    - **§ai-use** AI use disclosure (p.16) — The author discloses that AI tools were used only for language editing and coding assistance, while the core ideas, EML Sheffer operator discovery, verification methodology, and results are the author's original work.
    - **§data-availability** Data availability (p.16) — Code, scripts, and reproducibility materials for the paper are available in the SymbolicRegressionPackage repository and archived on Zenodo.
    - **§acknowledgments** Acknowledgments (p.16) — Acknowledgment of computational resources provided by Google Cloud Research Credits, followed by a note about supplementary information.
    - **§supplementary** Supplementary Information (p.16–17) — The supplementary information section indicates that extensive three-part supporting information is provided as a separate SI Appendix PDF, with no additional content presented in this section.

---

## Sections

### §title All elementary functions from a single operator (p.1–2)

The paper introduces a single binary operator eml(x,y) = exp(x) - ln(y) that, together with the constant 1, can generate all standard elementary functions, constants, and arithmetic operations, and demonstrates its use in gradient-based symbolic regression via binary trees.

**Key points:**
- [result] A single binary operator eml(x,y) = exp(x) - ln(y) together with the constant 1 is sufficient to generate all standard elementary functions, arithmetic operations, and mathematical constants. (p.1)
- [claim] No prior primitive analogous to a universal gate for continuous mathematics had been known before this work. (p.1)
- [method] The operator eml was discovered through systematic exhaustive search and verified constructively against a scientific-calculator function basis. (p.1)
- [result] Every elementary function expression in EML form is a binary tree of identical nodes, yielding the simple grammar S → 1 | eml(S,S). (p.1)
- [result] Gradient-based symbolic regression using EML trees as trainable circuits with Adam optimizer can exactly recover closed-form elementary functions from numerical data at tree depths up to 4. (p.1)
- [caveat] EML trees can fit arbitrary data, but exact formula recovery is demonstrated only when the generating law is an elementary function. (p.1)
- [result] Specific derivations show ex = eml(x,1) and ln(x) = eml(1, eml(eml(1,x), 1)). (p.1)

**Defines:**
- `eml(x,y)` — The binary operator defined as exp(x) - ln(y), called the Exp-Minus-Log operator (p.1)
- `S → 1 | eml(S,S)` — The grammar rule generating all EML expression trees using only the constant 1 and the eml operator (p.1)
- `1` — The sole constant used alongside eml to generate all elementary functions (p.1)


##### §summary-paragraph Summary paragraph (p.2–3)

The section introduces the EML operator eml(x,y) = exp(x)ln(y) and the constant 1 as a universal primitive for all elementary functions, analogous to NAND gates in Boolean logic, and outlines its implications for symbolic regression and combinatorial structure.

**Key points:**
- [result] The operator eml(x,y) = exp(x)ln(y) together with the constant 1 can reconstruct arithmetic, all standard elementary transcendental functions, and constants including integers, fractions, radicals, e, π, and i. (p.2)
- [result] A two-button calculator using only 1 and eml suffices to perform everything a full scientific calculator can do. (p.2)
- [claim] Elementary functions belong to a much simpler generative class than previously recognized. (p.2)
- [result] Every EML expression is a binary tree of identical nodes described by the context-free grammar S → 1 | eml(S,S), which is isomorphic to full binary trees and Catalan structures. (p.2)
- [result] Parameterized EML trees can be optimized by standard gradient methods, and trained weights can snap to exact closed-form expressions when the underlying law is elementary. (p.2)
- [claim] A single trainable EML architecture has the potential to discover any elementary formula from data. (p.2)
- [claim] Preliminary searches suggest related operators with even stronger properties exist, including a ternary variant requiring no distinguished constant. (p.2)
- [claim] Classical reductions reduced elementary functions to a few primitives but could not reduce them to a single universal primitive before the EML operator. (p.2)
- [open-question] It was previously unclear whether the apparent diversity of elementary functions is intrinsic or whether a smaller generative basis exists. (p.2)

**Defines:**
- `eml(x,y)` — The EML operator defined as exp(x)·ln(y) (p.2)
- `S → 1 | eml(S,S)` — Context-free grammar generating all EML expressions as binary trees (p.2)
- `1` — The distinguished constant that together with eml forms a universal basis for elementary functions (p.2)

*Depends on: Boolean universality / NAND gate analogy, exp-log representation of elementary functions, symbolic regression, context-free grammars and Catalan structures, logarithm tables and Euler's formula as classical reductions*


##### §significance Significance statement (p.3–4)

A single binary operation eml(x,y) can replace all standard mathematical operations (arithmetic, trigonometric, logarithmic, etc.), meaning a calculator with just two buttons—EML and the digit 1—can compute everything a scientific calculator does, with implications for uniform circuit-like encoding of mathematical expressions.

**Key points:**
- [result] All standard mathematical operations taught in school (fractions, roots, logarithms, trigonometric functions) can be replaced by a single operation eml(x,y). (p.3)
- [result] A calculator with only two buttons—EML and the digit 1—is computationally equivalent to a full scientific calculator. (p.3)
- [claim] Trigonometric functions reduce to the complex exponential, illustrating that many standard operations are redundant. (p.3)
- [claim] Because a single repeatable element suffices to express all operations, mathematical expressions become uniform circuits analogous to electronics built from identical transistors. (p.3)
- [claim] The universality of eml(x,y) opens new approaches to encoding, evaluating, and discovering formulas in scientific computing. (p.3)
- [claim] The result is not merely a mathematical curiosity but has practical significance for how mathematical expressions are represented and processed. (p.3)

**Defines:**
- `eml(x,y)` — A single binary operation claimed to be capable of replacing all standard mathematical operations (p.3)
- `EML` — The calculator button corresponding to the eml operation (p.3)


#### §1 1 Introduction (p.4–6)

This section introduces the EML (Exp-Minus-Log) operator as a single binary primitive sufficient to express all standard real elementary functions, analogous to NAND for Boolean logic, requiring only the constant 1 as a terminal symbol.

**Key points:**
- [claim] Single reusable primitives such as NAND, ReLU, and SUBLEQ play disproportionately large roles in mathematics, engineering, and biology. (p.4)
- [claim] The existence of a single sufficient operator is conceptually striking even if classical logic can function with a redundant family of connectives. (p.4)
- [claim] Elementary functions, arithmetic operations, and standard constants are heavily redundant, especially in the complex domain, but have never previously been treated as candidates for reduction to a single primitive operator. (p.4)
- [result] The exp-log pair allows multiplication and addition to be expressed in terms of each other via exp and ln. (p.5)
- [result] Euler's formula shows that trigonometric functions can be viewed as another face of exp and ln once the imaginary unit is introduced. (p.5)
- [definition] The EML operator eml(x,y) = exp(x) - ln(y) is defined as a single binary operator that can express any standard real elementary function through repeated application together with the constant 1. (p.5)
- [result] The constant 1 is required as a terminal symbol to neutralize the logarithmic term via ln(1)=0, and computations must be performed in the complex domain. (p.5)
- [result] For a pocket calculator with no external input variables, two buttons suffice for full functionality: the binary EML operator and the terminal symbol 1. (p.5)
- [result] No further reduction of operator count below one binary operator and one terminal symbol is possible. (p.5)
- [claim] The existence of a binary elementary function that serves as a Sheffer-type operator for all elementary functions is described as somewhat unexpected. (p.5)

**Defines:**
- `eml(x,y)` — Exp-Minus-Log binary operator defined as exp(x) - ln(y) (p.5)
- `EML` — Abbreviation for the Exp-Minus-Log operator eml(x,y) = exp(x) - ln(y) (p.5)

*Depends on: Boolean logic and NAND gate, Euler's formula, Exp-log representation of arithmetic, Elementary functions definition, Sheffer stroke / functional completeness, Broken calculator problem, Complex principal branch of logarithm*

*Equations: (1), (2), (3)*
*Tables: 1*

#### §2 2 Methods (p.6–8)

The author describes an ablation-based search methodology to identify minimal operator sets for elementary functions, using hybrid numeric bootstrapping verification with transcendental constants and iterative list-transfer between known and target primitives.

**Key points:**
- [method] A set of 36 primitives (constants, unary functions, binary operations) serves as the starting list for ablation testing. (p.6)
- [definition] An operator set is deemed complete if it can reconstruct all 36 primitives from Table 1 on the real axis. (p.6)
- [claim] Nested expressions of RPN depth 5–9 are typical witnesses for completeness, and no regular automatic method exists to find them without brute-force or AI assistance. (p.6)
- [caveat] Direct symbolic verification of operator completeness is computationally intractable, necessitating a hybrid numeric approach. (p.7)
- [result] Kolmogorov-style complexity K of typical reconstructions is at most 7, and the search was conducted up to K = 9. (p.7)
- [method] Substituting algebraically independent transcendental constants for variables enables reliable numeric sieving of formula equivalence candidates under the Schanuel conjecture. (p.7)
- [method] The iterative bootstrapping procedure transfers successfully reconstructed primitives from the target list to the known-operators list until the target list is empty. (p.7)
- [result] Ablation testing yielded progressively smaller calculator configurations (Calc 3, 2, 1, 0) with different required primitives. (p.8)
- [result] The function eml(x,y) paired with the constant 1 forms a complete basis for reconstructing all elementary functions. (p.8)
- [result] EML is not unique; close variants EDL (requiring constant e) and a third variant using negative infinity as a terminal symbol also exist. (p.8)
- [result] A Rust implementation of VerifyBaseSet is three orders of magnitude faster than the Mathematica version, allowing re-checking of EML in seconds. (p.8)

**Defines:**
- `K` — Kolmogorov-style complexity, measured as the length of a Reverse Polish Notation (RPN) calculator program (p.7)
- `suc(x)` — Successor function: x + 1 (p.6)
- `pre(x)` — Predecessor function: x - 1 (p.6)
- `inv(x)` — Reciprocal/inverse function: 1/x (p.6)
- `S_0` — Initial set of constants, functions, and operators verified as known/available (p.7)
- `C_0` — Initial set of constants, functions, and operators to be reconstructed (defines 'computing all elementary functions') (p.7)
- `S_i` — Known operator set at iteration i of the bootstrapping procedure (p.7)
- `C_i` — Remaining target operator set at iteration i of the bootstrapping procedure (p.7)
- `VerifyBaseSet` — Procedure that verifies whether a given operator set can reconstruct all target primitives (p.7)

*Depends on: Table 1 (36 primitives list), Table 2 (Mathematica/Wolfram reference instruction set and Calc configurations), equation (3) (definition of eml(x,y)), Schanuel conjecture, Kolmogorov complexity, Reverse Polish Notation (RPN), Mathematica SymbolicRegression package, Supplementary Information Part II*

*Equations: (3)*
*Tables: 1, 2*

#### §3 3 Results (p.8–10)

This section presents the progressive reduction of a 36-primitive scientific calculator to a single binary operator EML, tracing historical configurations (Calc 3 through Calc 0) and introducing EML and its cousins EDL and -EML, with verification methods described.

**Key points:**
- [result] Calc 3 was the first configuration to surpass the Wolfram Language primitive set, using 6 primitives: exp, ln, negation, reciprocal, and addition. (p.8)
- [result] Calc 2 achieves the same expressiveness as Calc 3 using only exp, ln, and subtraction, reducing the count to 4 primitives while retaining the ability to generate constants internally. (p.8)
- [claim] The non-commutative operator subtraction is crucial because it provides both expression-tree growth and inversion capabilities. (p.8)
- [result] Calc 1 uses binary exponentiation and binary logarithm as base operators and works only with e or π as terminal constants; no other constant was found to work. (p.8)
- [result] Calc 0 absorbs the constant e into the exp function, reducing the system to 3 primitives and suggesting a single binary operator might suffice. (p.8)
- [claim] All minimal configurations involve pairs of inverse functions and non-commutative operations, a pattern that guided the discovery of EML. (p.8)
- [result] EML, defined as eml(x,y) = exp(x) - ln(y) with constant 1, is a single binary operator sufficient to generate all 36 elementary operations. (p.9)
- [result] EML has at least two cousins: EDL defined as exp(x)/ln(y) with constant e, and -EML defined as ln(x) - exp(y) with constant negative infinity. (p.9)
- [result] The natural logarithm can be expressed in EML at depth 3 as ln(z) = eml(1, eml(eml(1,z), 1)). (p.9)
- [result] EML expressions for elementary operations range from depth 1 (exponential) to depth 8 (multiplication). (p.9)
- [method] Verification of the EML completeness is supported by symbolic simplification in Mathematica, numerical cross-checks across four implementations (C, NumPy, PyTorch, mpmath), and a constructive completeness proof sketch. (p.9)
- [open-question] A speculated undiscovered binary operator may generate constants from arbitrary input, unlike EML which requires constant 1. (p.9)
- [open-question] A speculated undiscovered unary operator may retain properties of neural network activation functions while enabling exact evaluation of elementary functions with standard arithmetic. (p.9)

**Defines:**
- `eml(x,y)` — The EML Sheffer operator defined as exp(x) - ln(y), with constant 1 (p.9)
- `edl(x,y)` — The EDL cousin operator defined as exp(x) / ln(y), with constant e (p.9)
- `eml(y,x)` — The -EML cousin operator defined as ln(x) - exp(y), with constant negative infinity (p.9)
- `Calc 3` — 6-primitive calculator configuration using exp, ln, negation, reciprocal, and addition, accepting any real constant as input (p.8)
- `Calc 2` — 4-primitive calculator configuration using exp, ln, and subtraction (p.8)
- `Calc 1` — 4-primitive calculator configuration using binary exponentiation and binary logarithm with terminal constant e or π (p.8)
- `Calc 0` — 3-primitive calculator configuration using exp and binary logarithm with e absorbed into exp (p.8)

*Depends on: Table 1 (36 elementary operations and primitives), Section 2 (methods and discovery search procedure), SymbolicRegression package, Supplementary Information Part II (completeness proof and verification)*

*Equations: (4a), (4b), (4c), (5)*
*Figures: 1*
*Tables: 1, 2*

#### §4 4 Usage and applications (p.10)

This section introduces usage and applications of EML, beginning with the EML compiler subsection, noting that the uniform tree structure of EML expressions suggests several directions for implementation and application.

**Key points:**
- [claim] The uniform tree structure of EML expressions suggests several directions for implementation and application. (p.10)

*Depends on: EML expression syntax, tree structure of EML expressions, Section 3 (EML definitions)*


##### §4.1 4.1 EML compiler (p.10–11)

Describes a prototype EML compiler that converts mathematical formulas into pure EML form using only the Sheffer-stroke-like EML operation, discusses implementation details including complex branch issues, and notes compatibility with various numerical and symbolic computation environments.

**Key points:**
- [result] The VerifyBaseSet procedure provides sufficient data to reconstruct any primitive or composite elementary expression in terms of the EML Sheffer stroke operation. (p.10)
- [method] A prototype EML compiler coded in Python converts formulas into pure EML form. (p.10)
- [result] EML expressions can be evaluated symbolically in Mathematica or numerically in any IEEE 754-compliant language. (p.10)
- [claim] Pure EML form can be executed on a single-instruction stack machine resembling a single-button RPN calculator. (p.10)
- [claim] Pure EML form could potentially be implemented efficiently in FPGA or analog circuits. (p.10)
- [result] The RPN encoding of ln x requires K=7 EML instructions: 11xE1EE. (p.10)
- [caveat] EML-compiled expressions work on the real axis except at isolated points such as zero and domain endpoints, and internal computations for trigonometric functions must be performed in the complex domain. (p.11)
- [caveat] The principal branch choice for complex logarithm in EML causes a 2πi jump on the negative real axis due to the 1/z term, preventing standard use of ln(-1)=iπ. (p.11)
- [result] EML-compiled formulas work correctly in symbolic Mathematica and IEEE 754 floating-point (e.g. math.h in C) because these properly handle ln0=-∞ and e^{-∞}=0. (p.11)
- [caveat] EML expressions do not work out of the box in pure Python/Julia or numerical Mathematica because special floats raise errors or cause overflow, but do work in NumPy and PyTorch. (p.11)
- [caveat] The Lean 4 proof assistant assigns a junk value of 0 to the complex logarithm at zero, causing straightforward formalization of the EML chain to fail. (p.11)
- [claim] The edge-case difficulties encountered in EML are not fundamentally different from those in ordinary floating-point or symbolic computation. (p.11)

**Defines:**
- `K` — Number of EML instructions required to encode a function in RPN form; here K=7 for ln x (p.10)
- `E` — Shorthand notation for the eml operation in RPN strings (p.10)
- `11xE1EE` — RPN string encoding of the natural logarithm ln x in pure EML form (p.10)

*Depends on: VerifyBaseSet procedure, EML Sheffer operation (4a), natural logarithm EML form (5), EML definition (3), Fig. 1 (base set data), Fig. 2 (expression tree for ln), IEEE 754 floating-point standard, principal branch of complex logarithm*

*Equations: (3), (4a), (5)*
*Figures: 1, 2*

##### §4.2 4.2 Elementary functions as binary trees and analog circuits (p.11–12)

EML notation naturally produces binary expression trees via a trivial context-free grammar, enabling uniform representation of elementary functions as both symbolic trees and analog circuits using the EML Sheffer operator as a basic building block.

**Key points:**
- [result] Any elementary function expression tree in EML notation is binary, governed by the trivial context-free grammar S → 1 | eml(S,S). (p.11)
- [definition] Input variables are added as additional terminal symbols in the EML grammar (e.g., x in the univariate case). (p.11)
- [result] The identity function can be computed using an EML tree of depth 4, which allows input variables to be moved down the tree. (p.12)
- [caveat] Trigonometric and other elementary functions have EML trees too large to display in print, as shown in Table 4. (p.12)
- [claim] The EML Sheffer operator can serve as a new basic building block for analog computing circuits, analogous to NAND gates, Op-Amps, and transistors. (p.12)
- [method] Using the EML compiler, any elementary function expression can be converted to an analog circuit with binary tree topology. (p.12)
- [claim] EML trees provide a uniform treatment of generic multivariate elementary functions, addressing a known problem in analog computing. (p.12)
- [definition] The EML Sheffer operation is non-commutative, so arrow chirality determines input order: first counterclockwise input after dot is expx, then lny. (p.12)

**Defines:**
- `eml(S,S)` — Context-free grammar production rule representing the binary EML Sheffer operation applied to two sub-expressions (p.11)
- `S` — Non-terminal symbol in the EML context-free grammar representing any elementary function expression (p.11)
- `minus(x)` — Negation function: minus(x) = -x (p.12)
- `inv(x)` — Reciprocal function: inv(x) = 1/x (p.12)
- `expx` — First (counterclockwise) input to the EML Sheffer operator, representing the argument of the exponential component (p.12)
- `lny` — Second input to the EML Sheffer operator, representing the argument of the logarithm component (p.12)

*Depends on: 4.1, eml operator definition, EML Sheffer stroke, Table 3, Table 4, Fig. 2*

*Equations: (3)*
*Figures: 2*
*Tables: 3, 4*

##### §4.3 4.3 Symbolic Regression by continuous optimization (p.12–15)

This section presents a 'master formula' approach to symbolic regression using EML, constructing a parameterized binary expression tree that can represent any elementary function by setting discrete parameter values, and demonstrates continuous optimization to recover exact symbolic expressions from data.

**Key points:**
- [claim] A single EML operator is sufficient to construct a multiparameter master formula that can represent any elementary function by choosing specific parameter values. (p.12)
- [result] A full binary tree of depth n has 2^n - 1 branches and 2^n leaves, and the largest transformers with trillions of parameters correspond to an equivalent tree depth of 40. (p.12)
- [result] The level-2 master formula has 14 free parameters, and in general the level-n master formula has 5 * 2^n - 6 parameters. (p.14)
- [method] Each input to eml(x,y) in the master formula can be represented as a linear combination alpha_i + beta_i * x + gamma_i * f, which unifies the three possible input cases: constant 1, variable x, or previous eml result f. (p.13)
- [claim] The EML master formula is complete by design, capable of expressing all elementary functions, unlike typical SR approaches that use a potentially incomplete subset of operations. (p.14)
- [result] Using simplex reparameterization with a level-3 master formula and Mathematica's NMinimize, exact recovery of ln(x) was achieved with fitting error at numerical precision, with perfect generalization beyond the training range. (p.14)
- [result] Blind recovery from random initialization succeeds in 100% of runs at depth 2, approximately 25% at depths 3-4, below 1% at depth 5, and not at all at depth 6 in 448 attempts. (p.15)
- [result] When the correct EML tree weights are perturbed by Gaussian noise, optimization converges back to exact values in 100% of runs even at depths 5 and 6, confirming valid basins of attraction exist. (p.15)
- [open-question] Finding correct basins of attraction from random initialization becomes harder with increasing tree depth, suggesting a need for better binary operators with non-exponential asymptotics and no domain issues. (p.15)
- [caveat] Training EML networks in practice requires clamping arguments for exp(x) and careful handling of complex arithmetic to avoid NaN errors and range overflow. (p.15)
- [result] Successful symbolic recovery yields mean squared errors at the level of machine epsilon squared (~10^-32), consistent with exact symbolic recovery. (p.15)

**Defines:**
- `alpha_i + beta_i * x + gamma_i * f` — General linear combination representing each input to eml in the master formula, unifying constant 1, variable x, and previous eml output f (p.13)
- `alpha_i` — Constant coefficient in the linear combination input representation (p.13)
- `beta_i` — Coefficient of input variable x in the linear combination input representation (p.13)
- `gamma_i` — Coefficient of previous eml result f in the linear combination input representation (p.13)
- `f` — Result from a previous eml computation used as input to higher-level eml nodes (p.13)
- `F(x)` — The level-2 master formula as a function of input x parameterized by alpha_i, beta_i, gamma_i (p.13)
- `5 * 2^n - 6` — Number of parameters in the level-n master formula (p.14)

*Depends on: EML operator definition (equation 3), EML binary expression tree structure, EML compiler (Subsection 4.1), Softmax function, Adam optimizer, Simplex reparameterization, Supplementary Information Part III*

*Equations: (1), (1012), (11), (17), (19), (27), (3), (35), (37), (39), (4), (47), (6), (7)*
*Tables: 4*

#### §5 5 Conclusions and open questions (p.15–16)

The conclusions summarize EML as a sufficient primitive for elementary functions, discuss open questions about Sheffer-type operators without distinguished constants, the necessity of complex intermediates, and EML's advantages over conventional neural networks for interpretable symbolic computation.

**Key points:**
- [result] The EML operator provides a single sufficient primitive from which all real elementary functions can be constructed and evaluated. (p.15)
- [claim] EML is not unique; close variants such as EDL and the swapped-argument form eml(y,x) likely exhibit similar universality properties. (p.15)
- [open-question] Whether an EML-type binary Sheffer operator can function without pairing with a distinguished constant remains an open question. (p.15)
- [caveat] Proving impossibility of a Sheffer operator without a distinguished constant for any given candidate is non-trivial, as illustrated by the example B(x,y)=x-y/2. (p.15)
- [claim] A ternary operator T(x,y,z) = e^(x/ln x) * ln z / e^y satisfying T(x,x,x)=1 is identified as a candidate for further analysis. (p.16)
- [open-question] Whether a univariate Sheffer operator exists that simultaneously serves as a neural activation function and generates all elementary functions remains open. (p.16)
- [claim] EML requires complex arithmetic internally to compute real elementary functions, analogously to how quantum computing uses complex amplitudes to produce real probabilities. (p.16)
- [claim] A continuous Sheffer operator working purely in the real domain appears impossible, as no real-domain alternatives were found. (p.16)
- [result] Any conventional neural network is a special case of an EML tree architecture because standard activation functions are themselves elementary functions. (p.16)
- [result] EML representations can recover closed-form elementary subexpressions by snapping trained weights to exact binary values, providing interpretability unavailable to conventional architectures. (p.16)

**Defines:**
- `B(x,y)` — Example binary function B(x,y) = x - y/2 used to illustrate traps in Sheffer operator analysis (p.15)
- `T(x,y,z)` — Ternary operator candidate defined as e^(x/ln x) times ln z / e^y, satisfying T(x,x,x)=1 (p.16)

*Depends on: Section 1 (introduction and EML definition, eq. 3), Section 4.3 (EML use with distinguished constants and weight snapping), Equation (2) (Euler's formula), Equation (4b) (EDL definition), Equation (4c) (swapped-argument eml), SI Section 5 (univariate Sheffer open question)*

*Equations: (2), (3), (4b), (4c)*

##### §ai-use AI use disclosure (p.16)

The author discloses that AI tools were used only for language editing and coding assistance, while the core ideas, EML Sheffer operator discovery, verification methodology, and results are the author's original work.

**Key points:**
- [claim] The discovery of the EML Sheffer operator and all core research contributions are entirely the author's own work, not AI-generated. (p.16)
- [method] Large language models (Claude, Grok, Gemini, ChatGPT) were used only for language editing and coding assistance. (p.16)


##### §data-availability Data availability (p.16)

Code, scripts, and reproducibility materials for the paper are available in the SymbolicRegressionPackage repository and archived on Zenodo.

**Key points:**
- [method] All code and scripts used to generate figures, tables, and numerical results are publicly available in the SymbolicRegressionPackage repository and archived on Zenodo. (p.16)


##### §acknowledgments Acknowledgments (p.16)

Acknowledgment of computational resources provided by Google Cloud Research Credits, followed by a note about supplementary information.

**Key points:**
- [claim] Computational resources for this research were partially provided by Google Cloud Research Credits. (p.16)


##### §supplementary Supplementary Information (p.16–17)

The supplementary information section indicates that extensive three-part supporting information is provided as a separate SI Appendix PDF, with no additional content presented in this section.

**Key points:**
- [method] Extensive three-part supporting information is supplied as a separate SI Appendix PDF document. (p.16)


#### §references References (p.17)

*Equations: (08), (1), (10), (16), (1929), (2), (3), (4), (4361), (6), (7825), (90)*
*Figures: 1, 2*
