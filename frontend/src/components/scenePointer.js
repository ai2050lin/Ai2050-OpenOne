import { useEffect, useRef } from 'react';

export const SCENE_MOUSE_HINT = '左键拖动平移 · 右键拖动旋转 · 滚轮缩放';

export function panScene(view, dx, dy) {
  return { ...view, panX: (view.panX ?? 0) + dx, panY: (view.panY ?? 0) + dy };
}

export function panFixedCanvas(view, dx, dy, canvas) {
  const bounds = canvas.getBoundingClientRect();
  return panScene(view, dx * canvas.width / bounds.width, dy * canvas.height / bounds.height);
}

export function pitchedScreenY(y, depth, pitch, verticalScale, depthScale) {
  const cos = Math.cos(pitch), sin = Math.sin(pitch);
  return -y * (verticalScale * cos - depthScale * sin) + depth * (depthScale * cos + verticalScale * sin);
}

export function useScenePointer({ onPan, onRotate, onPick, onHover, onLeave, touchPan = false }) {
  const drag = useRef(null);
  const cancel = event => {
    if (drag.current?.pointerId === event.pointerId) drag.current = null;
  };

  return {
    onPointerDown(event) {
      if (event.isPrimary === false || (event.button !== 0 && event.button !== 2) || drag.current) return;
      const { pointerId, clientX, clientY, button, pointerType } = event;
      drag.current = { pointerId, x: clientX, y: clientY, moved: false, button, target: event.target,
        rotate: button === 2 || (pointerType === 'touch' && !touchPan) };
      event.currentTarget.setPointerCapture(pointerId);
    },
    onPointerMove(event) {
      const active = drag.current;
      if (!active) { onHover?.(event); return; }
      if (active.pointerId !== event.pointerId) return;
      const dx = event.clientX - active.x, dy = event.clientY - active.y;
      if (!active.moved && Math.abs(dx) + Math.abs(dy) <= 4) return;
      if (!active.moved) onLeave?.();
      active.moved = true;
      active.x = event.clientX;
      active.y = event.clientY;
      if (active.rotate) onRotate(dx, dy);
      else onPan(dx, dy, event.currentTarget);
    },
    onPointerUp(event) {
      const active = drag.current;
      if (!active || active.pointerId !== event.pointerId) return;
      drag.current = null;
      const distance = Math.abs(event.clientX - active.x) + Math.abs(event.clientY - active.y);
      if (!active.moved && distance <= 4 && active.button === 0 && event.button === 0) onPick?.(event, active.target);
    },
    onPointerCancel: cancel,
    onLostPointerCapture: cancel,
    onPointerLeave: onLeave,
    onContextMenu: event => event.preventDefault(),
  };
}

export function useSceneWheel(ref, onWheel, enabled = true) {
  const handler = useRef(onWheel);
  useEffect(() => { handler.current = onWheel; }, [onWheel]);
  useEffect(() => {
    const element = enabled ? ref.current : null;
    const wheel = event => { event.preventDefault(); handler.current(event.deltaY); };
    element?.addEventListener('wheel', wheel, { passive: false });
    return () => element?.removeEventListener('wheel', wheel);
  }, [ref, enabled]);
}
