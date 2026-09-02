# Eigenfaces Demo — Implementation Context

## Experience contract

The page is an interaction-led educational working surface. It should let a visitor understand that a face is represented as an average plus weighted directions of variation, then immediately test that idea by changing a small number of weights.

### Desktop layout

- Use one continuous black stage aligned to a shared content grid.
- Allocate the larger left region to a square reconstructed-face figure.
- Place the hero at the top of the left region:
  - Title: `Eigenfaces`
  - Subtitle: `Face It: It’s Just Linear Algebra`
- Place a 3 × 3 grid in the right region. Reading order is `Mean`, `PC 01`, `PC 02`, …, `PC 08`.
- The reconstruction is visually dominant. Component tiles remain large enough for facial structures and positive/negative lobes to be recognizable.
- The mean tile uses the same visual framing as the eigenface tiles but has no slider or hover affordance.

### Responsive layout

- At narrow widths, order the content as hero, reconstruction, then component grid.
- Reflow the component grid from three columns to two and then one when tile size or label legibility would otherwise suffer.
- Never introduce horizontal page scrolling.
- A revealed slider must stay inside its tile and remain usable without obscuring the component label or weight value.

### Component interaction

- PC tiles enlarge with a small transform on hover or focus; surrounding layout dimensions remain fixed.
- Reveal a translucent control surface over the lower half of an active tile. Preserve enough of the eigenface above and around it to connect the control to the image.
- Use a native range input. Pointer and touch changes update the reconstruction immediately.
- Display weights in standard-deviation units for human readability: `z_i = w_i / sqrt(lambda_i)`. The stored calculation continues to use raw PCA weights.
- Each slider spans `z_i,baseline - 3` through `z_i,baseline + 3`, which is equivalent to raw weight bounds `w_i,baseline ± 3 sqrt(lambda_i)`.
- The reset control restores all eight exposed raw weights to their baseline values in one state update.

## Dataset selection and provenance

Use the Flickr-Faces-HQ dataset (FFHQ) published by NVIDIA. FFHQ contains aligned face images and is appropriate here because its contemporary, varied portraits are a closer domain match for the supplied input than older laboratory datasets.

This project uses FFHQ only for a non-commercial educational visualization of PCA. It must not be described or extended as face recognition, identity matching, biometric verification, or demographic analysis.

### Required attribution and restrictions

- Cite Tero Karras, Samuli Laine, and Timo Aila, *A Style-Based Generator Architecture for Generative Adversarial Networks*.
- Link to the official dataset repository: <https://github.com/NVlabs/ffhq-dataset>.
- State that the source images were automatically aligned and cropped and that this project converts a deterministic subset to grayscale, resizes it, and derives PCA artifacts.
- Preserve FFHQ's CC BY-NC-SA 4.0 dataset terms and the original per-image license metadata obligations.
- Treat all derived mean, eigenface, and reconstruction artifacts as non-commercial/share-alike material unless a later legal review establishes otherwise.
- Before publishing, add a user-visible data attribution notice and retain machine-readable provenance for the selected image identifiers and licenses.

Do not commit the downloaded corpus or the 255 MB FFHQ metadata file. The offline pipeline may cache them in an ignored working directory.

## Deterministic training subset

- Use only FFHQ records whose official metadata category is `training`.
- Select exactly 5,000 unique records with a fixed pseudorandom seed of `20260902` from the sorted training record identifiers. Sampling is without replacement.
- Persist the seed, sorted selected identifiers, source thumbnail checksums, original authors, source URLs, and image licenses in a generated provenance manifest.
- Use the official aligned 128 × 128 thumbnail pixels. Do not realign individual FFHQ thumbnails.
- Decode to sRGB, compute grayscale luminance as `Y = 0.2126 R + 0.7152 G + 0.0722 B`, and scale values to floating point `[0, 1]`.
- Flatten every image in row-major order to a 16,384-element vector. Preserve one consistent row-major convention through training, export, tests, and browser rendering.

The deterministic sample rule provides reproducibility; it does not claim demographic balance. Do not describe the sample or resulting eigenspace as unbiased or representative of humanity.

## Supplied portrait preprocessing

The source portrait is `/Users/kosta/Downloads/IMG_6763 2.heic`. It is a 914 × 1003 Display P3 HEIC photograph with a near-frontal, smiling subject and a real-world background.

- Perform preprocessing offline; never bundle or request the raw HEIC from the browser.
- Decode the embedded orientation and convert the image to sRGB before landmark detection or resampling.
- Detect a single primary face and stable eye and mouth landmarks.
- Apply the same eye/mouth-oriented square alignment geometry used by FFHQ's published alignment procedure: center and rotate from the eye midpoint, eye-to-eye vector, and eye-to-mouth vector; include the full forehead, jaw, and lateral face boundary while excluding as much room and clothing as the FFHQ crop permits.
- If automated landmarks are ambiguous, inspect the aligned result and record any manual landmark correction in the provenance manifest. Do not silently substitute a different photograph.
- Resample the aligned square to 128 × 128 with a high-quality Lanczos filter, convert it with the same sRGB luminance equation used for training, and scale to `[0, 1]`.
- Validate that the aligned face is upright, both eyes are visible, the face scale matches representative FFHQ thumbnails, and the output contains exactly 16,384 finite samples.
- Store the processed grayscale input only if the interface needs a comparison or development fixture. The required first-version interface displays the reconstruction rather than the source image.

Because the portrait includes a broad smile and stronger expression than many training examples, some residual blur or expression loss is an honest property of a linear eigenspace. Do not conceal that limitation with generative enhancement, identity-aware retouching, or post-reconstruction sharpening.

## PCA pipeline

Let `X` be the `5000 × 16384` training matrix, with one grayscale image per row.

1. Compute the per-pixel mean `mu` and center the data: `Xc = X - mu`.
2. Fit up to 512 principal components with a deterministic randomized or incremental SVD implementation configured with the same seed, `20260902`.
3. Sort components by descending eigenvalue. Preserve the fitted ordering in every exported artifact.
4. Canonicalize each eigenvector's otherwise arbitrary sign: locate the element with greatest absolute magnitude and multiply the vector by `-1` when that element is negative. Apply the same sign change to all corresponding projected weights.
5. Calculate cumulative explained-variance ratios at checkpoints 128, 256, and 512.
6. Choose `K_full` as the smallest checkpoint whose cumulative explained variance is at least `0.95`. If none qualifies, set `K_full = 512` and record the achieved ratio rather than claiming 95% retention.
7. Project the processed portrait into the selected basis:
   `w = U_full @ (x - mu)`.
8. Compute the complete baseline reconstruction:
   `r_baseline = mu + U_full.T @ w`.

Here, `U_full` has shape `K_full × 16384`, `w` has length `K_full`, and `r_baseline` has length 16,384. Keep unclamped floating-point values for the exported baseline calculation. Clamp only while converting a displayed image to luminance.

The eight public controls correspond to components 1–8 by explained variance. Components 9 through `K_full` remain fixed at their projected portrait weights.

## Browser reconstruction contract

The browser does not need the entire fitted basis. Export the complete baseline plus only the data required for the eight live deltas.

For adjustable raw weights `w'_1 … w'_8`, render:

`r_display = clamp(r_baseline + sum((w'_i - w_i) * u_i for i = 1..8), 0, 1)`

This formula is the central invariant. It ensures that:

- all untouched sliders reproduce the full `K_full`-component baseline exactly;
- hidden components remain present and unchanged;
- moving one slider changes only its eigenface direction; and
- reset returns exactly to the original projection.

Implement this calculation in a framework-independent TypeScript module using typed arrays. Render the 128 × 128 luminance buffer to a canvas or equivalent pixel surface without network calls. A slider input should schedule at most one paint per animation frame while always using the newest state.

## Generated asset contract

The offline pipeline should produce a compact, versioned manifest and binary numeric payloads rather than large decimal JSON arrays.

### Manifest fields

The exact serialization may follow the generated site scaffold, but the logical manifest must contain:

- schema version;
- dataset name and official source URL;
- selection seed and selected-record provenance-manifest path;
- image width, height, grayscale transform, and flattening order;
- training sample count;
- `K_full` and cumulative explained variance at 128, 256, 512, and `K_full`;
- baseline reconstruction payload path and checksum;
- mean-face display asset path and checksum;
- component records 1–8, each with eigenvalue, baseline raw weight, baseline standardized weight, raw eigenvector payload path, display-thumbnail path, and checksums;
- portrait preprocessing metadata sufficient to reproduce its alignment without embedding the original image;
- pipeline version and deterministic seed.

### Numeric payloads

- Store the baseline reconstruction and eight raw eigenvectors as little-endian Float32 values.
- Validate payload lengths before use: the baseline and every eigenvector must contain exactly 16,384 finite values.
- Generate the average-face display image from `clamp(mu, 0, 1)`.
- Generate each eigenface thumbnail with a symmetric per-component mapping around mid-gray:
  `thumbnail_i = 0.5 + 0.5 * u_i / max(abs(u_i))`.
- Do not use independent positive and negative min/max scaling, because that would obscure the zero point.
- Preserve binary numerical eigenvectors separately from their display-normalized thumbnails.

## State model

- Immutable loaded state: manifest, baseline reconstruction, eigenvectors 1–8, eigenvalues, and baseline weights.
- Mutable UI state: eight adjusted raw weights and the currently revealed tile, if any.
- Derived state: standardized display values and the current reconstructed luminance buffer.
- Do not duplicate reconstruction state across the stage and component grid. Both consume the same canonical values.
- Do not persist adjusted face weights across sessions unless a later product requirement explicitly requests it.

## Visual language

- Background: continuous near-black presentation stage.
- Type: system Apple stack, near-white primary text, cool muted secondary labels, tabular numeric values.
- Cyan: focus, live slider tracks, and active component state only.
- Warm gold: a single restrained emphasis associated with the reconstruction or reset-to-baseline state.
- Images: crisp square grayscale surfaces with hairline framing and no decorative colorization.
- Motion: short opacity/transform transitions for disclosure and focus; no entrance animation and no easing on direct slider movement.
- Keep the title large and confident but subordinate the hero's footprint to the working surface.

## Accessibility requirements

- Use a true `h1` for the title and meaningful figure captions for the reconstruction, mean, and components.
- Use native `input type="range"` controls with programmatic labels such as `Adjust principal component 1 weight`.
- Announce numeric values without causing a live-region utterance for every pointer movement.
- Support arrow keys, Home, End, Page Up, and Page Down through native range behavior.
- Make slider controls discoverable by focus without requiring hover.
- On touch, tapping a different component dismisses the prior overlay and reveals the new one.
- Provide visible focus rings, sufficient contrast, and targets of at least 44 CSS pixels where touch interaction is expected.
- Under `prefers-reduced-motion: reduce`, remove tile scaling and reduce disclosure transitions to effectively instantaneous state changes.

## Tests and acceptance criteria

### Offline pipeline

- The selected FFHQ IDs are identical across clean runs with seed `20260902`.
- The training tensor is `5000 × 16384`, finite, row-major, and within `[0, 1]`.
- The processed portrait is 128 × 128, finite, and uses the same grayscale transform and row-major convention.
- Eigenvalues are non-increasing and eigenvectors are orthonormal within an explicit floating-point tolerance.
- Component signs follow the canonical greatest-absolute-element rule.
- `K_full` follows the 128/256/512 checkpoint rule and the manifest records achieved explained variance.
- Recomputing the baseline from `mu`, `U_full`, and `w` agrees with the exported baseline within Float32 tolerance.

### Browser math

- Baseline weights render the exported baseline exactly within Float32 tolerance; the result must not collapse to an eight-component reconstruction.
- Moving only component `i` by delta `d` changes the unclamped buffer by exactly `d * u_i` within tolerance.
- Hidden components 9 through `K_full` remain represented through the baseline and never change during interaction.
- Reset restores all eight original weights and the exact baseline buffer.
- Display conversion clamps values below 0 and above 1 without mutating the underlying baseline or component arrays.
- Corrupt, missing, non-finite, or incorrectly sized assets produce an accessible error state rather than a blank or distorted canvas.

### Interface

- The desktop first viewport contains the dominant reconstruction, top-left hero, and all nine component-grid tiles at the target reference size.
- Mean is first and noninteractive; PC 01–08 follow in correct variance order.
- Hover, keyboard focus, and touch can each reveal and operate every component slider.
- Tile enlargement does not cause layout shift or clipping of focus indicators.
- Reset is keyboard and touch accessible and visibly returns every readout to its baseline.
- Mobile layouts preserve content order and introduce no horizontal scrolling.
- Reduced-motion mode removes nonessential scaling and animation.
- The supplied portrait and its derivatives are absent from social-preview metadata and imagery.

### Final verification

Run the eventual repository's tests, lint checks, and production build. Check desktop, tablet, and mobile widths; keyboard-only navigation; touch interaction; reduced motion; and 200% zoom. Compare the rendered page with the sibling Dolly Zoom demo for visual hierarchy, alignment, typography, spacing, palette, and restraint while retaining the eigenfaces-specific layout defined here.
