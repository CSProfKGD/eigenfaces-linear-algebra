import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

import { describe, expect, it } from 'vitest';

type Manifest = {
  defaultDimensions: number;
  maxDimensions: number;
  width: number;
  height: number;
  cumulativeExplainedVariance: number[];
};

const manifest = JSON.parse(
  readFileSync(resolve('public/eigenfaces/manifest.json'), 'utf8'),
) as Manifest;

describe('dimension-prefix assets', () => {
  it('defaults to 512 dimensions and includes all 1000 cached prefixes', () => {
    expect(manifest.defaultDimensions).toBe(512);
    expect(manifest.maxDimensions).toBe(1000);
    expect(manifest.cumulativeExplainedVariance).toHaveLength(1000);
    expect(gunzipSync(readFileSync(resolve('public/eigenfaces/prefix-reconstructions.delta.bin'))).length).toBe(
      1000 * manifest.width * manifest.height,
    );
  });

  it('uses the actual monotonically increasing cumulative variance curve', () => {
    for (let index = 1; index < manifest.cumulativeExplainedVariance.length; index += 1) {
      expect(manifest.cumulativeExplainedVariance[index]).toBeGreaterThanOrEqual(
        manifest.cumulativeExplainedVariance[index - 1],
      );
    }
    expect(manifest.cumulativeExplainedVariance[511]).toBeCloseTo(0.9512, 4);
    expect(manifest.cumulativeExplainedVariance[999]).toBeGreaterThan(
      manifest.cumulativeExplainedVariance[511],
    );
  });
});
