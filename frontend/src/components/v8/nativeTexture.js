// Missing model dimensions mean no texture, never a fabricated 1×1 field.
export const positiveDimension = value => Number.isSafeInteger(value) && value > 0 && value <= 0x7fffffff;

export function nativeGrid(count) {
  if (!positiveDimension(count)) return [0, 0];
  const width = Math.ceil(Math.sqrt(count * 1.6));
  return [width, Math.ceil(count / width)];
}

export function nativeTexture(width, height, fill) {
  // Guard at the allocation boundary too: asynchronous callers may not yet have a shape.
  if (!positiveDimension(width) || !positiveDimension(height)) return null;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) return null;
  const pixels = context.createImageData(canvas.width, canvas.height);
  fill(pixels.data);
  context.putImageData(pixels, 0, 0);
  return canvas;
}
