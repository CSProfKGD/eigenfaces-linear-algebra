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
};

type Manifest = {
  dataset: string;
  sampleCount: number;
  width: number;
  height: number;
  kFull: number;
  explainedVariance: Record<string, number>;
  baseline: string;
  mean: string;
  reconstruction: string;
  components: ComponentRecord[];
};

type LoadedModel = {
  manifest: Manifest;
  baseline: Float32Array;
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

export function EigenfacesDemo() {
  const [model, setModel] = useState<LoadedModel | null>(null);
  const [zValues, setZValues] = useState<number[]>([]);
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
        const [baseline, ...vectors] = await Promise.all([
          fetchFloat32(manifest.baseline),
          ...manifest.components.map((component) => fetchFloat32(component.vector)),
        ]);
        const expectedLength = manifest.width * manifest.height;
        if (baseline.length !== expectedLength || vectors.some((vector) => vector.length !== expectedLength)) {
          throw new Error('The eigenspace data has an unexpected size.');
        }
        if (!cancelled) {
          setModel({ manifest, baseline, vectors });
          setZValues(manifest.components.map((component) => component.baselineZ));
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

  useEffect(() => {
    if (!model || rawWeights.length === 0 || !canvasRef.current) return;
    const frame = requestAnimationFrame(() => {
      const values = reconstructFace(
        model.baseline,
        model.vectors,
        rawWeights,
        model.manifest.components.map((component) => component.baselineWeight),
      );
      if (canvasRef.current) drawLuminance(canvasRef.current, values, model.manifest.width, model.manifest.height);
    });
    return () => cancelAnimationFrame(frame);
  }, [model, rawWeights]);

  const reset = useCallback(() => {
    if (model) setZValues(model.manifest.components.map((component) => component.baselineZ));
  }, [model]);

  const hasChanges = Boolean(model && zValues.some((value, index) => Math.abs(value - model.manifest.components[index].baselineZ) > 0.001));
  const variance = model
    ? model.manifest.explainedVariance[String(model.manifest.kFull)] ?? model.manifest.explainedVariance['512']
    : null;

  return (
    <main className="eigenfaces-page">
      <div className="ambient-glow" aria-hidden="true" />
      <section className="experience-grid" aria-label="Interactive eigenfaces reconstruction">
        <figure className="reconstruction-figure">
          <figcaption className="figure-topline">
            <span className="eyebrow">Reconstruction</span>
            <span className="model-status">{model ? `${model.manifest.kFull} dimensions` : 'Building eigenspace'}</span>
          </figcaption>

          <div className="reconstruction-stage">
            {!model && !error && <Image className="reconstruction-fallback" src="/eigenfaces/reconstruction.png" alt="Grayscale preview of the reconstructed input portrait" width={128} height={128} priority unoptimized />}
            <canvas ref={canvasRef} className={model ? 'reconstruction-canvas is-ready' : 'reconstruction-canvas'} width={model?.manifest.width ?? 128} height={model?.manifest.height ?? 128} role="img" aria-label="PCA reconstruction of the supplied portrait" />
            <div className="scan-line" aria-hidden="true" />
            {error && <div className="model-error" role="alert"><span>Eigenspace unavailable</span><small>{error}</small></div>}
          </div>

          <div className="reconstruction-footer">
            <div><span className="footer-kicker">Projected from</span><strong>{model ? model.manifest.sampleCount.toLocaleString() : '5,000'} faces</strong></div>
            <div><span className="footer-kicker">Variance retained</span><strong>{variance ? `${(variance * 100).toFixed(1)}%` : '—'}</strong></div>
            <Button type="button" variant="ghost" size="sm" className="reset-button" onClick={reset} disabled={!model || !hasChanges}>
              <RotateCcw aria-hidden="true" /> Reset weights
            </Button>
          </div>
        </figure>

        <header className="hero-block">
          <h1>Eigenfaces</h1>
          <p>Face It: It’s Just Linear Algebra</p>
        </header>

        <section className="basis-section" aria-labelledby="basis-title">
            <div className="basis-heading"><h2 id="basis-title">Principal components</h2><p>Hover. Focus. Reshape.</p></div>
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
                  >
                    <Image src={component.thumbnail} alt={`Eigenface for principal component ${component.index}`} width={128} height={128} unoptimized />
                    <figcaption><span>PC {String(component.index).padStart(2, '0')}</span><small>{(component.explainedVariance * 100).toFixed(1)}%</small></figcaption>
                    <button
                      type="button"
                      className="tile-activator"
                      aria-label={`Reveal principal component ${component.index} weight control`}
                      aria-expanded={activeTile === index}
                      onClick={() => setActiveTile(index)}
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
