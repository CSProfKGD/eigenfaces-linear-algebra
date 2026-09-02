export function reconstructFace(
  baseline: Float32Array,
  vectors: Float32Array[],
  adjustedWeights: number[],
  baselineWeights: number[],
): Float32Array {
  if (
    vectors.length !== adjustedWeights.length ||
    adjustedWeights.length !== baselineWeights.length
  ) {
    throw new Error('Weight and component counts must match.');
  }
  if (vectors.some((vector) => vector.length !== baseline.length)) {
    throw new Error('Every component must match the baseline length.');
  }

  const output = new Float32Array(baseline);
  for (let component = 0; component < vectors.length; component += 1) {
    const delta = adjustedWeights[component] - baselineWeights[component];
    if (delta === 0) continue;
    const vector = vectors[component];
    for (let pixel = 0; pixel < output.length; pixel += 1) {
      output[pixel] += delta * vector[pixel];
    }
  }
  return output;
}
