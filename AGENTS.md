# Eigenfaces Demo — Working Notes

## Product intent

This project is a premium, single-page educational visualization of eigenfaces and principal component analysis. It must make the linear-algebra idea tangible without presenting itself as face recognition, identification, classification, or biometric software.

Preserve the sparse Apple Keynote character established by the sibling `Dolly Zoom demo`: a continuous near-black presentation stage, near-white type, cool muted labels, cyan interaction states, warm-gold emphasis, hairline separators, generous negative space, and restrained motion. Every visual element must either explain the reconstruction or improve control clarity.

The eventual site is for non-commercial educational use. Follow the FFHQ attribution and share-alike requirements recorded in `context.md`.

## Non-negotiable composition

- Keep the primary working surface visible in the first desktop viewport.
- The left side contains one large reconstructed-face stage.
- The upper-right hero reads `Eigenfaces` with the subtitle `Face It: It’s Just Linear Algebra`.
- Beneath the hero, the right side contains a balanced 3 × 3 tile grid: the average face first, followed by eigenfaces 1–8 in decreasing explained-variance order.
- The average-face tile is informational and never presents a weight control.
- The eight eigenface tiles are the only exposed component controls. The reconstruction itself must use the larger basis selected by the offline pipeline; never reconstruct from only the eight visible components.
- On narrow screens, present the reconstruction first, then the hero and component grid. Reflow the grid without horizontal scrolling.

## Interaction model

- An eigenface tile subtly enlarges on pointer hover or keyboard focus and reveals its signed weight control across the lower half of the tile.
- Touch users reveal a tile control with a tap. A subsequent drag changes its native range input.
- Slider movement updates the reconstruction on the same interaction frame. Do not animate, debounce, or ease direct manipulation.
- Show the current signed weight in a compact tabular-numeral readout.
- Provide one quiet reset action that restores all eight exposed weights to their original projected values.
- Reset must restore the exact baseline reconstruction. It must not re-run PCA or modify hidden weights.
- Keep the initial page completely still. Motion should clarify focus or state and must respect `prefers-reduced-motion`.

## Architecture boundaries

- Keep dataset acquisition, portrait preprocessing, PCA fitting, component selection, projection, and asset export in an offline pipeline.
- Keep the browser experience deterministic and static: load precomputed assets, retain the eight adjustable weights in local React state, and render the reconstruction from the baseline plus component deltas.
- Do not download FFHQ, fit PCA, detect landmarks, or process the source HEIC in the browser.
- Treat `context.md` as the source of truth for preprocessing, PCA math, selected-component rules, and generated asset contracts.
- Centralize reconstruction math in a framework-independent module. UI components must not independently implement PCA calculations.
- Store the source portrait outside the public web bundle. Only the deliberately processed grayscale visualization asset may be included in the future site.
- Never use the supplied portrait, its reconstruction, or a recognizable derivative in Open Graph or other social-preview imagery.

## Visual system

- Use a near-black background (`#000` or the Dolly Zoom demo's equivalent), near-white foreground type, and muted cool grays.
- Reserve cyan for focus, active sliders, and mathematical interaction. Use warm gold sparingly for the reconstruction result or a single high-value emphasis.
- Use the system Apple font stack; do not add web fonts. Set numeric values with tabular figures.
- Favor square image stages, hairline borders at low opacity, precise alignment, and minimal corner rounding.
- Keep component imagery grayscale. Eigenface thumbnails must use a shared, symmetric mid-gray normalization so their signs remain visually comparable.
- Avoid generic dashboard cards, gradients used only as decoration, oversized pills, ornamental illustrations, and unnecessary explanatory copy.
- Keep hover enlargement small enough that neighboring tiles do not jump or reflow.

## Accessibility

- Use semantic headings, buttons, figures, and native range inputs.
- Give every slider an accessible name that includes its component number and describes that it changes the reconstruction.
- Make hover-only information equally available through keyboard focus and touch.
- Preserve visible, high-contrast focus indicators and a logical reading/tab order: reconstruction, hero context, average face, components 1–8, reset.
- Provide text alternatives for the average face, each eigenface, and the reconstructed output without claiming the output identifies the subject.
- Maintain usable targets and legible values at mobile sizes and at 200% zoom.

## Data, licensing, and privacy

- Use only the deterministic FFHQ subset and derived artifacts described in `context.md`.
- Include dataset attribution, the FFHQ paper citation, modification disclosure, and the applicable non-commercial/share-alike terms before publication.
- Describe the feature as an educational PCA reconstruction. Do not add identity labels, similarity scores, recognition claims, or demographic inference.
- Do not commit the raw FFHQ corpus, the FFHQ metadata archive, or the original HEIC portrait.
- Do not log, upload, or transmit face pixels from the eventual client application.

## Verification

When implementation exists, run the repository's test, lint, and production-build scripts. Tests must cover the PCA and interaction invariants in `context.md`. Verify the interface at desktop, tablet, and mobile widths, with keyboard-only input, touch input, reduced motion, and 200% zoom.

Compare the finished page against the Dolly Zoom demo for hierarchy, type scale, spacing, palette, alignment, motion restraint, and first-viewport density. Match its visual quality, not its camera-specific layout or behavior.
