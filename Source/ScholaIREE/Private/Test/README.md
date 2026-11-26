# ScholaIREE automation tests

These tests compile without checked-in IREE model fixtures. Metadata parsing tests use inline JSON. Policy tests that require a loaded ``.vmfb`` early-out or log when:

- `Source/ThirdParty/IREE/<Platform>/` is not populated, or
- No fixture assets are present under your project's `Content/` tree.

To exercise end-to-end IREE inference locally, export a model from Python (`save_model_as_iree`), import the `.vmfb` via the editor factory, populate IREE third-party libraries, and assign the resulting `UIREEModelData` asset to a `UIREEPolicy`.