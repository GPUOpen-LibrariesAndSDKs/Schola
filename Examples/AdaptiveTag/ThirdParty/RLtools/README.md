# RLtools dependency

The `include` directory and `LICENSE` are unmodified upstream files from
https://github.com/rl-tools/rl-tools at commit
`b32d9985c65a5e098a6bbf190fd994962d288b99` (also recorded in `REVISION`).

Headers are committed so a clone can build without fetching another dependency.
The project stages the upstream MIT notice with packaged games. Run
`../../Prepare-LearningDemo.ps1` from this directory only when preparing or
restoring the dependency; the normal build validates the vendored files.
