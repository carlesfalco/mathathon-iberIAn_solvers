# Release coordination

This directory supplies code, figures, data, and verification evidence. It is not the editable report package.

The final report and slides are maintained in the **Final Report** chat. Before submission:

1. Link numerical claims in the final report to the relevant implementations and saved parameter ranges here.
2. Distinguish freshly rerun computations from historical receipts.
3. Decide which longer research suites need fresh end-to-end runs; `scripts/check_local.sh` does not claim them.
4. Run the appropriate Python and Rust checks after any code change.
5. Keep session records and build output out of the repository; include the report and slide LaTeX sources intentionally.
6. Publish the exact release commit and supply its full hash to the report workstream.

No Lean project or Lean-verified claim is currently included.
