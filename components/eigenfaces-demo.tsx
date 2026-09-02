'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import Image from 'next/image';

import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { reconstructFace } from '@/lib/eigenfaces';

type ComponentRecord = {
  index: number;
  eigenvalue: number;
  explainedVariance: number;
  baselineWeight: number;
  baselineZ: number;
  vector: string;
  thumbnail: string;
  labelTones: {
    name: 'dark' | 'light';
    variance: 'dark' | 'light';
  };
};

type Manifest = {
  dataset: string;
  sampleCount: number;
  width: number;
  height: number;
  kFull: number;
  defaultDimensions: number;
  maxDimensions: number;
  explainedVariance: Record<string, number>;
  cumulativeExplainedVariance: number[];
  baseline: string;
  prefixReconstructions: string;
  mean: string;
  reconstruction: string;
  components: ComponentRecord[];
};

type LoadedModel = {
  manifest: Manifest;
  baseline: Float32Array;
  prefixReconstructions: Uint8Array;
  vectors: Float32Array[];
};

function drawLuminance(canvas: HTMLCanvasElement, values: Float32Array, width: number, height: number) {
  const context = canvas.getContext('2d');
  if (!context) return;
  const image = context.createImageData(width, height);
  for (let index = 0; index < values.length; index += 1) {
    const luminance = Math.round(Math.min(1, Math.max(0, values[index])) * 255);
    const offset = index * 4;
    image.data[offset] = luminance;
    image.data[offset + 1] = luminance;
    image.data[offset + 2] = luminance;
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

async function fetchFloat32(url: string) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  return new Float32Array(await response.arrayBuffer());
}

async function fetchUint8(url: string) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  if (!url.endsWith('.bin')) return new Uint8Array(await response.arrayBuffer());
  const decompressed = response.body?.pipeThrough(new DecompressionStream('gzip'));
  if (!decompressed) throw new Error(`Could not decompress ${url}`);
  return new Uint8Array(await new Response(decompressed).arrayBuffer());
}

function decodePrefixDeltas(payload: Uint8Array, pixelCount: number) {
  for (let offset = pixelCount; offset < payload.length; offset += 1) {
    payload[offset] = (payload[offset] + payload[offset - pixelCount]) & 0xff;
  }
  return payload;
}

export function EigenfacesDemo() {
  const [model, setModel] = useState<LoadedModel | null>(null);
  const [zValues, setZValues] = useState<number[]>([]);
  const [dimensions, setDimensions] = useState(512);
  const [dimensionsOpen, setDimensionsOpen] = useState(false);
  const [activeTile, setActiveTile] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      try {
        const manifestResponse = await fetch('/eigenfaces/manifest.json');
        if (!manifestResponse.ok) throw new Error('The eigenspace is unavailable.');
        const manifest = (await manifestResponse.json()) as Manifest;
        const [baseline, prefixReconstructions, ...vectors] = await Promise.all([
          fetchFloat32(manifest.baseline),
          fetchUint8(manifest.prefixReconstructions),
          ...manifest.components.map((component) => fetchFloat32(component.vector)),
        ]);
        const expectedLength = manifest.width * manifest.height;
        if (
          baseline.length !== expectedLength ||
          prefixReconstructions.length !== expectedLength * manifest.maxDimensions ||
          vectors.some((vector) => vector.length !== expectedLength)
        ) {
          throw new Error('The eigenspace data has an unexpected size.');
        }
        decodePrefixDeltas(prefixReconstructions, expectedLength);
        if (!cancelled) {
          setModel({ manifest, baseline, prefixReconstructions, vectors });
          setZValues(manifest.components.map((component) => component.baselineZ));
          setDimensions(manifest.defaultDimensions);
        }
      } catch (reason) {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : 'The eigenspace is unavailable.');
        }
      }
    }
    void loadModel();
    return () => { cancelled = true; };
  }, []);

  const rawWeights = useMemo(() => {
    if (!model || zValues.length !== model.manifest.components.length) return [];
    return model.manifest.components.map((component, index) => {
      const deltaZ = zValues[index] - component.baselineZ;
      return component.baselineWeight + deltaZ * Math.sqrt(component.eigenvalue);
    });
  }, [model, zValues]);

  const selectedPrefix = useMemo(() => {
    if (!model) return null;
    if (dimensions === model.manifest.defaultDimensions) {
      return model.baseline;
    }
    const pixelCount = model.manifest.width * model.manifest.height;
    const offset = (dimensions - 1) * pixelCount;
    const result = new Float32Array(pixelCount);
    for (let pixel = 0; pixel < pixelCount; pixel += 1) {
      result[pixel] = model.prefixReconstructions[offset + pixel] / 255;
    }
    return result;
  }, [dimensions, model]);

  useEffect(() => {
    if (!model || !selectedPrefix || rawWeights.length === 0 || !canvasRef.current) return;
    const frame = requestAnimationFrame(() => {
      const values = reconstructFace(
        selectedPrefix,
        model.vectors,
        rawWeights,
        model.manifest.components.map((component) => component.baselineWeight),
        dimensions,
      );
      if (canvasRef.current) drawLuminance(canvasRef.current, values, model.manifest.width, model.manifest.height);
    });
    return () => cancelAnimationFrame(frame);
  }, [dimensions, model, rawWeights, selectedPrefix]);

  const reset = useCallback(() => {
    if (model) setZValues(model.manifest.components.map((component) => component.baselineZ));
  }, [model]);

  const hasChanges = Boolean(model && zValues.some((value, index) => Math.abs(value - model.manifest.components[index].baselineZ) > 0.001));
  const variance = model?.manifest.cumulativeExplainedVariance[dimensions - 1] ?? null;

  return (
    <main className="eigenfaces-page">
      <div className="ambient-glow" aria-hidden="true" />
      <section className="experience-grid" aria-label="Interactive eigenfaces reconstruction">
        <header className="hero-block">
          <h1>Eigenfaces</h1>
          <p>Face It: It’s Just Linear Algebra</p>
        </header>

        <figure className="reconstruction-figure">
          <figcaption className="figure-topline">
            <span className="eyebrow">Reconstruction</span>
          </figcaption>

          <div className="reconstruction-stage">
            {!model && !error && <Image className="reconstruction-fallback" src="/eigenfaces/reconstruction.png" alt="Grayscale preview of the reconstructed input portrait" width={128} height={128} priority unoptimized />}
            <canvas ref={canvasRef} className={model ? 'reconstruction-canvas is-ready' : 'reconstruction-canvas'} width={model?.manifest.width ?? 128} height={model?.manifest.height ?? 128} role="img" aria-label="PCA reconstruction of the supplied portrait" />
            <div className="scan-line" aria-hidden="true" />
            {error && <div className="model-error" role="alert"><span>Eigenspace unavailable</span><small>{error}</small></div>}
          </div>

          <div className="reconstruction-footer">
            <div className="dimensions-metric">
              <span className="footer-kicker">Dimensions</span>
              <div
                className={`dimensions-control${dimensionsOpen ? ' is-open' : ''}`}
                onBlur={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget)) setDimensionsOpen(false);
                }}
              >
                <button
                  type="button"
                  className="dimensions-trigger"
                  aria-expanded={dimensionsOpen}
                  aria-label={`${dimensions} ${dimensions === 1 ? 'dimension' : 'dimensions'}. Adjust reconstruction dimensions`}
                  onFocus={() => setDimensionsOpen(true)}
                  onClick={() => setDimensionsOpen((open) => !open)}
                >
                  {dimensions}
                </button>
                <div className="dimensions-slider-shell">
                  <span>1</span>
                  <Slider
                    aria-label="Number of principal components used in the reconstruction"
                    min={1}
                    max={model?.manifest.maxDimensions ?? 1000}
                    step={1}
                    value={[dimensions]}
                    disabled={!model}
                    onValueChange={(next) => {
                      const nextValue = Array.isArray(next) ? next[0] : next;
                      setDimensions(Math.round(nextValue));
                    }}
                  />
                  <span>{model?.manifest.maxDimensions ?? 1000}</span>
                </div>
              </div>
            </div>
            <div className="variance-metric"><span className="footer-kicker">Variance retained</span><strong>{variance ? `${(variance * 100).toFixed(1)}%` : '—'}</strong></div>
            <Button type="button" variant="ghost" size="sm" className="reset-button" onClick={reset} disabled={!model || !hasChanges}>
              <RotateCcw aria-hidden="true" /> Reset weights
            </Button>
          </div>
        </figure>

        <section className="basis-section" aria-labelledby="basis-title">
            <div className="basis-heading"><h2 id="basis-title">Principal components</h2></div>
            <div className="component-grid">
              <figure className="component-tile mean-tile">
                <Image src="/eigenfaces/mean.png" alt="Average face across the FFHQ training sample" width={128} height={128} unoptimized />
                <figcaption><span>Mean</span><small>μ</small></figcaption>
              </figure>

              {(model?.manifest.components ?? []).map((component, index) => {
                const value = zValues[index] ?? component.baselineZ;
                return (
                  <figure
                    className={`component-tile control-tile${activeTile === index ? ' is-active' : ''}`}
                    key={component.index}
                    onPointerLeave={(event) => {
                      if (event.pointerType !== 'mouse') return;
                      const focusedElement = document.activeElement;
                      if (focusedElement instanceof HTMLElement && event.currentTarget.contains(focusedElement)) {
                        focusedElement.blur();
                      }
                      setActiveTile(null);
                    }}
                  >
                    <Image src={component.thumbnail} alt={`Eigenface for principal component ${component.index}`} width={128} height={128} unoptimized />
                    <figcaption>
                      <span className={`label-${component.labelTones.name}`}>PC {String(component.index).padStart(2, '0')}</span>
                      <small className={`label-${component.labelTones.variance}`}>{(component.explainedVariance * 100).toFixed(1)}%</small>
                    </figcaption>
                    <button
                      type="button"
                      className="tile-activator"
                      aria-label={`Reveal principal component ${component.index} weight control`}
                      aria-expanded={activeTile === index}
                      onClick={() => {
                        if (window.matchMedia('(hover: none)').matches) setActiveTile(index);
                      }}
                    />
                    <div className="weight-control">
                      <div className="weight-readout"><span>Weight</span><output>{value >= 0 ? '+' : ''}{value.toFixed(2)}σ</output></div>
                      <Slider
                        aria-label={`Adjust principal component ${component.index} weight`}
                        min={component.baselineZ - 3}
                        max={component.baselineZ + 3}
                        step={0.01}
                        value={[value]}
                        onValueChange={(next) => {
                          const nextValue = Array.isArray(next) ? next[0] : next;
                          setZValues((current) => current.map((item, itemIndex) => itemIndex === index ? nextValue : item));
                        }}
                      />
                    </div>
                  </figure>
                );
              })}

              {!model && Array.from({ length: 8 }, (_, index) => (
                <div className="component-tile loading-tile" key={index} aria-hidden="true">
                  <div className="loading-face" /><span>PC {String(index + 1).padStart(2, '0')}</span>
                </div>
              ))}
            </div>
        </section>
      </section>

    </main>
  );
}
