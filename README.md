# Eigenfaces

**Face It: It’s Just Linear Algebra**

An interactive, non-commercial educational visualization of PCA face reconstruction. The interface reconstructs an aligned input portrait with up to 1,000 principal components and exposes the first eight component weights for direct exploration.

Live site: <https://csprofkgd.github.io/eigenfaces-linear-algebra/>

## Development

```bash
pnpm install
pnpm dev
```

Run the validation suite with `pnpm test`, `pnpm run lint`, and `pnpm run build`.

## Data and attribution

The eigenspace is derived from a deterministic 5,000-image sample of aligned 128 × 128 thumbnails from [Flickr-Faces-HQ (FFHQ)](https://github.com/NVlabs/ffhq-dataset). Images were converted to grayscale and used to derive the mean face, principal components, and reconstructions.

FFHQ is distributed under CC BY-NC-SA 4.0, and individual photographs retain their original licenses. See [`public/eigenfaces/ATTRIBUTION.txt`](public/eigenfaces/ATTRIBUTION.txt) for the citation and modification notice. This project is an educational PCA visualization, not recognition or biometric software.
