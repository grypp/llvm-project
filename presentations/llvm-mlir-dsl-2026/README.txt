mlir dsl tutorial — draft v13

How to build a Python DSL for MLIR
2026 US LLVM Developers’ Meeting · 27 October 2026
Guray Ozen — Principal Compiler Engineer, NVIDIA
Amir Tavakkoli — role and affiliation not yet supplied

V13: CORE FEATURES AND PLUGINS AS SIMPLE BOX DIAGRAMS
Slides 10–11 now use box diagrams with short labels. Slide 10 contains one
Core DSL foundation with four feature boxes: diagnostics, AST preprocessing
hooks, JIT cache and type inference. Slide 11 shows the three plugin families
and reusable implementations, then two DSL assemblies built from selected
plugins over that same Core DSL: the working mlir dsl and Your DSL.
There are no code listings, IR fragments or hook-call examples on these two
slides. Detailed behavior remains in later technical slides and notes.
The existing opening order, 47-slide count and 45-minute timing are retained.

CURRENT PLUGIN ARCHITECTURE
Based on the October 6 working tree at:
  /home/gozen/work/llvm-project/mlir/tools/mlir-dsl

The reusable authoring core is mlir.dsl. Programs for the reference language
use import mlir.mlir_dsl as m. Custom sub-DSLs assemble BaseDSL with selected
plugins and expose their own namespace.

New editable diagrams show core/plugin composition, three plugin families,
installation-to-call lifecycle, OpEmitter dispatch, and AST capture ownership.
Examples now use the current imports, plain Meta configuration, field-wise
structs, Pointer-based DLPack adaptation, and plain m.compile.

The five closing Tile IR slides map the existing DkgDSL target onto the new
plugin interfaces. They are labeled port sketches, not a newly implemented
Tile plugin or a GPU run. Direct typed annotations remain throughout.

ARTIFACTS
  llvm-mlir-dsl-tutorial-v13.pptx  Editable slides and speaker notes
  llvm-mlir-dsl-tutorial-v13.pdf   PDF preview
  overview.png / preview/       Rendered visual review
  slide-skeleton.html          Outline and evidence notes
  slide-skeleton.csv           Rehearsal timing
  speaker-notes.txt            Presenter detail, not slide prose
  extraction-feature-map.csv   Updated feature-to-slide mapping
  verification/v8/             Current architecture and runtime audit
  verification/v9/             Early-exit comparison sources
  validation.txt               Artifact and validation summary

TIMING
47 slides: Guray’s core 30 minutes, Amir’s placeholder 10 minutes, Guray’s
five target slides 5 minutes. No visible timestamps or boxed agenda.
Amir supplies his own mutation/PyIR content. Minimal slide wording and green
code headers remain; detailed contracts and limitations live in the notes.

EVIDENCE
Read-only source review and focused execution use the current LLVM working
tree and its built Python bindings. Audit files record loaded module paths,
source hashes, commands and results. The retained v8 checks cover numeric types, struct and
frozen-record behavior, native control flow, cache, DLPack, remarks, TVM-FFI,
plus GPU/alternate-emitter tracing. No GPU execution is claimed.

One existing diagnostics example fails: a type-unstable loop reports
CONTAINER_UNSUPPORTED instead of TYPE_UNSTABLE_JOIN. This does not imply
normal loop carries fail; native-loop and frozen-record examples passed.
No compiler implementation was edited for this presentation revision.

The private publication checkout’s compiler source snapshot predates the
current refactor. This presentation update does not silently replace that
source snapshot. Historical v6/v7 audits and older decks are preserved, but
are not evidence for the current API. Source publication is separate work.

STYLE
Same supplied cutlass-python-pytorch-v22 template: exact theme, masters,
fonts, editable NVIDIA logo; 10 × 5.625 inches. Native editable diagrams.
Previous v1–v12 PowerPoint/PDF artifacts preserved.

REBUILD
From this directory, with python-pptx, Pillow and PyMuPDF installed:
  python build_deck.py
  libreoffice -env:UserInstallation=file:///tmp/llvm-dsl-slides-lo --headless --convert-to pdf --outdir . llvm-mlir-dsl-tutorial-v13.pptx
  python render_preview.py

Use requirements.txt for dependencies. Install assets/fonts/*.ttf before
LibreOffice export. deck.json holds content/timing; build_deck.py and
tutorial_layouts.py render it with the bundled assets/template.pptx.
