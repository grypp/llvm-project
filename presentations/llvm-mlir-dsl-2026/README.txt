mlir dsl tutorial — draft v16

How to build a Python DSL for MLIR
2026 US LLVM Developers’ Meeting · 27 October 2026
Guray Ozen — Principal Compiler Engineer, NVIDIA
Amir Tavakkoli — role and affiliation not yet supplied

V16: AGENDA LABEL
The final agenda item now reads “Amir: OpenAI Triton DSL”.

V15: A SIMPLER OPENING
The title is unnumbered. Content slides 1–9 follow the requested sequence:
1. Existing options: hand-written IR and Python bindings, as plain bullets.
2. Complete scalar a+b bindings program; all setup, lowering, JIT and call code.
3. PyTorch tensors and the short mlir dsl program: this is what we want.
4. Four NVIDIA use cases, with the existing numbered diagrams.
5. Design questions: dialects, control flow, and IR generation versus execution.
6. Four shared core services, without nested architecture diagrams.
7. What a plugin changes: staged a+b emits arith.addi or your dialect's op.
8. Shared core + selected plugins = your DSL; compilation is optional.
9. AST frontend versus tracing, with optional AST capture for control flow.
The four-role/two-family record now appears beside the sub-DSL authoring
example, after the audience has learned what those features do.

The feature walkthrough follows the supplied inventory in order: tracing,
two kinds of values, numeric and aggregate types, native control flow,
explicit builders, host boundary, kernels, compilation, caches, trace-only,
diagnostics, locations/remarks, observability, sub-DSL authoring.
The Tile closing sketch now uses the same role/family contracts.

CURRENT SOURCE
/home/gozen/work/llvm-project/mlir/tools/mlir-dsl — October 7, 2026
Core namespace: mlir.dsl
Reference namespace: mlir.mlir_dsl; concrete class: MlirTestDSL
Plugins: type_ops, func_entry, ast_preprocessor, compiler; decorators, adapters.
The sub-DSL owns pipeline() and register_dialects(). Op modules are not plugins.
The old list-valued plugin record and default dialect/compiler story are gone.

ARTIFACTS
  llvm-mlir-dsl-tutorial-v16.pptx  Editable slides and speaker notes
  llvm-mlir-dsl-tutorial-v16.pdf   PDF preview
  overview.png / preview/        Rendered visual review
  slide-skeleton.html/.csv       Outline and rehearsal timing
  speaker-notes.txt              Detail and source references
  extraction-feature-map.csv     Current feature-to-slide mapping
  verification/v14/              Source audit and feature execution evidence
  verification/v15/              Complete scalar and PyTorch opening programs
  validation.txt                Artifact validation summary

TIMING
53 pages: title + 52 numbered slides; Guray core30min, Amir10min, target5min.
The core has47 pages, mixing short diagram beats with selected code excerpts.
No visible timing labels. Amir supplies his own mutation content.
Slide numbers in the feature map exclude the title. PDF page = slide number +1.
The rehearsal CSV and notes include both slide numbers and PDF page numbers.

CHECKED EXECUTION
Complete scalar Python-bindings addition prints5. Complete PyTorch DSL program
produces [10,11,12,13]. The displayed programs are in verification/v15/.
Raw bindings and DSL PyTorch CPU addition match exactly at lengths0,1,4,17.
Current CPU examples01,02,03,05,07,09,14 passed. Diagnostics and AST extension
tests passed. TypeOps alternate rank-zero representation was traced only.
Metaprogram examples: scale12, dictionary/polymorphic pipeline18, fused array
[6,9,12,15]. Host scalar results are Numeric wrappers, unwrapped for assertions.
The older v8 diagnostic mismatch is historical: current diagnostics passed.
Commands, outputs and source identities are recorded under verification/v14.
Tile remains a port sketch; no new GPU execution or performance claim.
No compiler implementation was changed for this presentation.

PUBLICATION SCOPE
This update changes presentation artifacts and verification evidence only.
Compiler source updates are preserved from the repository's existing history.
Local source identities used for validation are recorded under verification/v14.
Historical deck/evidence versions are preserved but are not the current API reference.

STYLE AND REBUILD
Same supplied cutlass-python-pytorch-v22 template: exact theme, masters,
fonts and editable NVIDIA logo;10×5.625in; native editable diagrams.
Previous v1–v15 PowerPoint/PDF artifacts preserved.
  python build_deck.py
  libreoffice -env:UserInstallation=file:///tmp/llvm-dsl-slides-lo --headless --convert-to pdf --outdir . llvm-mlir-dsl-tutorial-v16.pptx
  python render_preview.py
Dependencies in requirements.txt. Install assets/fonts/*.ttf for PDF export.
